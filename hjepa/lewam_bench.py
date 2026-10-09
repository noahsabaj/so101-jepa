"""Training speed of a LeWAM recipe (PLAN.md A25): seconds per batch under speed variants, with LeWAM's own
batch code (third_party/lewam/scripts/train_lewam.py run_batch), on batches already on the GPU (the loader is
timed apart).

    sh scripts/uvr python hjepa/lewam_bench.py MODEL DATA.h5 VARIANTS [STEPS]

MODEL: a config in hjepa/config/lewam/. VARIANTS: comma-separated, each a +-joined set of
  fp32 | bf16           autocast precision
  cl                    channels_last encoder
  bench                 cudnn (MIOpen) benchmark: search the conv algorithms
  tf32                  TF32 matmuls and convs
  cenc                  torch.compile the encoder
  closs                 torch.compile the predictor loss (model.loss)
  ctrunk                torch.compile the encoder's ResNet trunk only (in place: checkpoint keys unchanged)
  gtrunk                ctrunk with CUDA graphs (mode reduce-overhead)
  fused                 fused AdamW
  live                  batches from the data loader in the loop (as in training), not cached on the GPU
  workersN              with live: N loader workers (default: the config's)
  ram                   with live: frames from the uncompressed copy in RAM (data.source decomp, load_in_ram;
                        LeWAM's scripts/decompress_h5.py writes it), not read from the h5 per sample
  prof                  also profile 10 steps: the top GPU kernels by their own time
  loader8, loader16 ... the data loader alone, with that many workers (batches per second)
e.g. fp32,bf16,bf16+bench,bf16+cl+bench,bf16+cenc+fused. Prints one line per variant: s/batch and the epoch
time it gives (train + val batches of one pass, val at 1/3 of a train batch).
"""

import os
import sys
import time
from contextlib import nullcontext
from functools import partial

import torch
from omegaconf import OmegaConf

from lewam_common import ROOT, trainer_namespace

os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data"))
T = trainer_namespace("lewam_bench", nosync=os.environ.get("LEWAM_NOSYNC", "1") == "1")


def load_cfg(model):
    parts, name = [], model
    while name:  # follow the defaults chain: MODEL -> its parent ... -> gr/tc -> base
        path = ROOT / "hjepa" / "config" / "lewam" / f"{name}.yaml"
        if not path.exists():
            path = ROOT / "third_party" / "lewam" / "configs" / "train" / f"{name}.yaml"
        c = OmegaConf.load(path)
        parent = next((d for d in c.pop("defaults", []) if d != "_self_"), None)
        parts.insert(0, c)
        name = parent
    return OmegaConf.merge(*parts)


def make_data(cfg, data, workers):
    d, m = cfg.data, cfg.model
    frames = T["load_frames"](data, list(d.views), d.source, d.decomp_dir, d.load_in_ram)
    actions, _ = T["read_actions"](data)
    starts, ep_starts, to_terminal = T["get_start_indices"](data, d.frameskip, d.terminal_state)
    ds = T["TrajectoryDataset"](frames, actions, starts, ep_starts, to_terminal, torch.arange(starts.shape[0]),
                                m.history_len, m.history_stride, m.num_actions_pred, m.num_states_pred, d.frameskip,
                                m.goal_type, m.H_max)
    kw = dict(num_workers=workers, pin_memory=True)
    if workers:
        kw.update(prefetch_factor=d.prefetch_factor, persistent_workers=True)
    return T["make_loader"](ds, cfg.train.batch_size, 0, **kw), actions, starts.shape[0]


def bench_loader(cfg, data, workers, steps):
    loader, _, _ = make_data(cfg, data, workers)
    it = iter(loader)
    for _ in range(5):
        next(it)
    t0 = time.perf_counter()
    for _ in range(steps):
        next(it)
    return (time.perf_counter() - t0) / steps


def bench_model(cfg, flags, batches, actions, steps, live=None):
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = "bench" in flags
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = "tf32" in flags
    model = T["build_model"](T["model_config"](cfg, actions.shape[1])).cuda()
    if "cl" in flags:
        model.encoder.to(memory_format=torch.channels_last)
    if "cenc" in flags:
        model.encoder = torch.compile(model.encoder)
    if "ctrunk" in flags:
        model.encoder.trunk.compile()
    if "gtrunk" in flags:
        model.encoder.trunk.compile(mode="reduce-overhead")
    if "closs" in flags:
        model.loss = torch.compile(model.loss)
    sigreg = T["SIGReg"]().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
                            fused="fused" in flags)
    autocast = partial(torch.autocast, device_type="cuda", dtype=torch.bfloat16) if "bf16" in flags else nullcontext

    def step(batch):
        with autocast():
            loss, _, _ = T["run_batch"](batch, model, None, sigreg, cfg, "cuda", train=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    get = (lambda i: next(live)) if live is not None else (lambda i: batches[i % len(batches)])
    warm = 15 if flags & {"cenc", "closs", "ctrunk", "gtrunk"} else 5
    t0 = time.perf_counter()
    for i in range(warm):
        step(get(i))
    torch.cuda.synchronize()
    t_warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(steps):
        step(get(i))
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - t0) / steps
    if "prof" in flags:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(10):
                step(get(i))
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=30, max_name_column_width=70),
              flush=True)
    return seconds, t_warm


if __name__ == "__main__":
    model_name, data, variants, *rest = sys.argv[1:]
    steps = int(rest[0]) if rest else 40
    cfg = load_cfg(model_name)
    cfg.dataset_path = data
    batches = actions = None
    for v in variants.split(","):
        flags = set(v.split("+"))
        loaders = [f for f in flags if f.startswith("loader")]
        if loaders:
            s = bench_loader(cfg, data, int(loaders[0][6:]), steps)
            print(f"[bench] {v}: {s:.4f} s/batch ({1 / s:.1f} batches/s) from the loader alone", flush=True)
            continue
        if batches is None:
            loader, actions, n_starts = make_data(cfg, data, cfg.data.num_workers)
            it = iter(loader)
            batches = [tuple(x.cuda() for x in next(it)) for _ in range(8)]
            n_train = round(n_starts * cfg.data.train_split / cfg.train.batch_size)
            n_val = round(n_starts * (1 - cfg.data.train_split) / cfg.train.batch_size)
        live = None
        if "live" in flags:
            workers = next((int(f[7:]) for f in flags if f.startswith("workers")), cfg.data.num_workers)
            live_cfg = cfg.copy()
            if "ram" in flags:
                live_cfg.data.source, live_cfg.data.load_in_ram = "decomp", True
            live = iter(make_data(live_cfg, data, workers)[0])
        try:
            s, t_warm = bench_model(cfg, flags, batches, actions, steps, live)
            epoch = s * (n_train + n_val / 3)
            print(f"[bench] {v}: {s:.4f} s/batch, epoch {epoch:.0f} s ({cfg.train.epochs} epochs "
                  f"{epoch * cfg.train.epochs / 3600:.2f} h), warmup {t_warm:.0f} s, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        except Exception as e:  # one variant failing does not stop the rest
            print(f"[bench] {v}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

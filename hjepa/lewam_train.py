"""Train a LeWAM model (PLAN.md A25) with LeWAM's own trainer (third_party/lewam/scripts/train_lewam.py),
its encoder sized for our 64 px views (hjepa/lewam_common.py).

    sh scripts/uvr python hjepa/lewam_train.py MODEL DATA.h5 [key=value ...]

MODEL is a config in hjepa/config/lewam/ (it extends LeWAM's gr or tc config); DATA.h5 is in LeWAM's layout
(hjepa/lewam_data.py). The run goes to data/ckpts/lewam/MODEL/seed42/ (lewam_best.pt, lewam_config.json,
lewam_full.pt for resuming).
Warm start: `init_from: PATH` in the MODEL config loads a LeWAM checkpoint (e.g. one of the authors') before
training; tensors whose shape differs (the action input and output for another action size, the view
embedding for another number of cameras) keep their fresh initialization. A resumed run loads its own state.
Data-parallel on W GPUs (hjepa/lewam_common.py Ddp; the batch is then W x train.batch_size):
    torchrun --nproc_per_node W hjepa/lewam_train.py MODEL DATA.h5 [key=value ...]
"""

import os
import sys
import time

import torch
from omegaconf import OmegaConf

from lewam_common import AFTER_CROP, COMPILED, GN_ATEN, ROOT, after_crop, lewam_model, trainer_namespace


def speedups():
    """CPU-side speed for LeWAM's trainer, whose step is bound by Python dispatch on a loaded node (profiled
    on an MI355X: the AdamW step was 28% of the main thread). LEWAM_FUSED=0 keeps LeWAM's foreach AdamW.
    Also prints the seconds per batch every LEWAM_TIMER (default 200) train batches: LeWAM logs per epoch."""
    if os.environ.get("LEWAM_FUSED", "1") == "1" and torch.cuda.is_available():
        class FusedAdamW(torch.optim.AdamW):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, fused=True, **kwargs)

            def load_state_dict(self, state_dict):  # a resumed foreach run: its groups and step counters
                super().load_state_dict(state_dict)
                for group in self.param_groups:
                    group["fused"], group["foreach"] = True, None
                    for p in group["params"]:
                        s = self.state.get(p, {})
                        if "step" in s:
                            s["step"] = s["step"].to(device=p.device, dtype=torch.float32)

        torch.optim.AdamW = FusedAdamW
    if os.environ.get("LEWAM_COMPILE", "0") == "1" and torch.cuda.is_available():
        # torch.compile the encoder after its crop as one graph (lewam_common.after_crop; checkpoint keys
        # unchanged), SIGReg and the predictor loss. Not the crop: compiled (unfold + gather), it crashed on ROCm
        # (illegal memory access). LEWAM_GRAPHS=1: all three also as CUDA graphs (mode reduce-overhead): the CPU
        # time a step fell 26 -> 5 ms at batch 8 on noah-pc (~1,000 kernel launches a step; the loss alone ~700).
        # Inductor's own GroupNorm backward gave wrong gradients (lewam_common.GN_ATEN): GroupNorm runs as ATen's
        # kernels when compiled. Off by default until the whole compiled step matches float64
        # (hjepa/jobs/step_ref.py); runs before 2026-10-10 15:10 EDT with the encoder compiled are suspect.
        import torch._dynamo as dynamo  # not `import torch._dynamo`: that makes torch a local name here

        dynamo.config.cache_size_limit = 32  # one graph per batch shape (train, val, last batch) and mode
        graphs = os.environ.get("LEWAM_GRAPHS", "0")  # 1 | 0 | a comma list of enc, sig, loss (debug)
        parts = {"enc", "sig", "loss"} if graphs == "1" else set() if graphs == "0" else set(graphs.split(","))
        if os.environ.get("LEWAM_FALLBACK_RANDOM") == "1":
            import torch._inductor.config as inductor_config
            inductor_config.fallback_random = True
        part_mode = {k: "reduce-overhead" if k in parts else None for k in ("enc", "sig", "loss")}
        mode = part_mode["enc"]
        GN_ATEN["on"] = True
        encode = torch.compile(after_crop, fullgraph=True, dynamic=False, mode=mode)
        # a CUDA graph's output lives in its pool until the graph's next run: the copy keeps the latents valid
        # (without it, validation read latents a later run had overwritten)
        AFTER_CROP["fn"] = (lambda encoder, pixels: encode(encoder, pixels).clone()) if mode else encode
        COMPILED["on"], COMPILED["mode"] = True, part_mode["sig"]  # SIGReg (lewam_common.fast)
        mode = part_mode["loss"]
        print(f"[lewam] graphs on: {sorted(parts)}, fallback_random {os.environ.get('LEWAM_FALLBACK_RANDOM') == '1'}",
              flush=True)
        build = lewam_model.build_model

        def build_compiled(cfg):
            model = build(cfg)
            loss = torch.compile(model.loss, mode=mode)
            # the copies again: the trainer keeps the loss tensors past the next batch (its val sums read them later)
            model.loss = (lambda *args, **kwargs: tuple(t.clone() for t in loss(*args, **kwargs))) if mode else loss
            return model

        lewam_model.build_model = build_compiled
    every, clip = int(os.environ.get("LEWAM_TIMER", "200")), torch.nn.utils.clip_grad_norm_
    state = {"n": 0, "t": time.perf_counter()}

    def timed_clip(*args, **kwargs):  # called once per train batch
        state["n"] += 1
        if every and state["n"] % every == 0 and os.environ.get("RANK", "0") == "0":
            now = time.perf_counter()
            print(f"[lewam] batch {state['n']}: {(now - state['t']) / every:.4f} s/batch", flush=True)
            state["t"] = now
        return clip(*args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = timed_clip


def warm_start(path):
    build = lewam_model.build_model

    def build_and_load(cfg):
        model = build(cfg)
        own, src = model.state_dict(), torch.load(path, map_location="cpu")
        keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"[warm start] {path}: {len(keep)} of {len(own)} tensors loaded; fresh: "
              f"{sorted(k for k in own if k not in keep)}", flush=True)
        return model

    lewam_model.build_model = build_and_load


def config_value(config_dir, name, key):
    """`key` of our config `name`, or else of the configs in its defaults (the last one first, as Hydra composes)."""
    cfg = OmegaConf.load(config_dir / f"{name}.yaml")
    if key in cfg:
        return cfg[key]
    for parent in reversed(cfg.get("defaults", [])):
        if isinstance(parent, str) and (config_dir / f"{parent}.yaml").exists():
            value = config_value(config_dir, parent, key)
            if value is not None:
                return value
    return None


if __name__ == "__main__":
    model, data, *overrides = sys.argv[1:]
    config_dir = ROOT / "hjepa" / "config" / "lewam"
    init = config_value(config_dir, model, "init_from")
    if init:
        warm_start(ROOT / init)
    speedups()
    os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data"))
    run_dir = ROOT / "data" / "ckpts" / "lewam" / model / "seed42"
    rank_suffix = "" if os.environ.get("RANK", "0") == "0" else os.environ["RANK"]  # torchrun: one hydra dir a rank
    sys.argv = [sys.argv[0], "--config-dir", str(config_dir), "--config-name", model,
                f"dataset_path={data}", f"run_name={model}", f"run_dir={run_dir}",
                f"hydra.run.dir={run_dir}/hydra{rank_suffix}", *overrides]
    sys.argv[0] = str(ROOT / "third_party" / "lewam" / "scripts" / "train_lewam.py")
    trainer_namespace("__main__", nosync=os.environ.get("LEWAM_NOSYNC", "1") == "1")

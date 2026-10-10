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

from lewam_common import ROOT, lewam_model, trainer_namespace


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
    if os.environ.get("LEWAM_COMPILE", "1") == "1" and torch.cuda.is_available():
        # torch.compile the encoder's ResNet trunk (in place: checkpoint keys unchanged) and the predictor loss:
        # 15% faster on an MI355X. Not the whole encoder: its random crop (unfold + gather) crashed the compiled
        # graph on ROCm (illegal memory access).
        build = lewam_model.build_model

        def build_compiled(cfg):
            model = build(cfg)
            if hasattr(model.encoder, "trunk"):
                model.encoder.trunk.compile()
            model.loss = torch.compile(model.loss)
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

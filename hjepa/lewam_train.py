"""Train a LeWAM model (PLAN.md A25) with LeWAM's own trainer (third_party/lewam/scripts/train_lewam.py),
its encoder sized for our 64 px views (hjepa/lewam_common.py).

    sh scripts/uvr python hjepa/lewam_train.py MODEL DATA.h5 [key=value ...]

MODEL is a config in hjepa/config/lewam/ (it extends LeWAM's gr or tc config); DATA.h5 is in LeWAM's layout
(hjepa/lewam_data.py). The run goes to data/ckpts/lewam/MODEL/seed42/ (lewam_best.pt, lewam_config.json,
lewam_full.pt for resuming).
Warm start: `init_from: PATH` in the MODEL config loads a LeWAM checkpoint (e.g. one of the authors') before
training; tensors whose shape differs (the action input and output for another action size, the view
embedding for another number of cameras) keep their fresh initialization. A resumed run loads its own state.
"""

import os
import runpy
import sys

import torch
from omegaconf import OmegaConf

from lewam_common import ROOT, lewam_model


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


if __name__ == "__main__":
    model, data, *overrides = sys.argv[1:]
    config_dir = ROOT / "hjepa" / "config" / "lewam"
    init = OmegaConf.load(config_dir / f"{model}.yaml").get("init_from")
    if init:
        warm_start(ROOT / init)
    os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data"))
    run_dir = ROOT / "data" / "ckpts" / "lewam" / model / "seed42"
    sys.argv = [sys.argv[0], "--config-dir", str(config_dir), "--config-name", model,
                f"dataset_path={data}", f"run_name={model}", f"run_dir={run_dir}",
                f"hydra.run.dir={run_dir}/hydra", *overrides]
    runpy.run_path(str(ROOT / "third_party" / "lewam" / "scripts" / "train_lewam.py"), run_name="__main__")

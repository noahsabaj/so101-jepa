"""DSReg diagnostic (hjepa/dsreg.py run) on a released LeWAM checkpoint (PLAN.md A27 test bed): does a rotation of
its latent (192-d, from a 32-keypoint ResNet-18) put single factors of OGBench Cube into single latents?

    python hjepa/lewam_dsreg.py extract CUBE.h5 OUT.h5 [EPISODES]   # a subset: EPISODES (100) spread evenly (CPU)
    python hjepa/lewam_dsreg.py run RUN SUBSET.h5 OUT.json [K] [POOL] [ANCHORS] [NEIGHBOURS]

RUN: a checkpoint folder in $STABLEWM_HOME/checkpoints (lewam-cube); PYTHONPATH=third_party/lewam. Factors: the 6
arm joints, the gripper opening, the block's position, sin and cos of 4 x its yaw (it looks the same every 90
degrees). Observations: the 224 px frame average-pooled by POOL (default 14: 16 x 16 x 3).
"""
import sys
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import torch

NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "grip", "block_x", "block_y", "block_z", "yaw4_sin", "yaw4_cos"]
KEYS = ["ep_idx", "proprio_joint_pos", "proprio_gripper_opening", "privileged_block_0_pos", "privileged_block_0_yaw"]


def extract(src, dst, episodes=100):
    f = h5py.File(src, "r")
    eps = np.unique(np.linspace(0, len(f["ep_len"]) - 1, int(episodes)).astype(int))
    spans = [(int(f["ep_offset"][e]), int(f["ep_len"][e])) for e in eps]
    rows = np.concatenate([np.arange(o, o + n) for o, n in spans])
    with h5py.File(dst, "w") as g:
        for key in KEYS:
            g[key] = f[key][rows]
        px = g.create_dataset("pixels", (len(rows), *f["pixels"].shape[1:]), np.uint8,
                              chunks=(1, *f["pixels"].shape[1:]), **hdf5plugin.Zstd())
        i = 0
        for o, n in spans:
            px[i:i + n] = f["pixels"][o:o + n]
            i += n
    print(f"{len(eps)} episodes, {len(rows)} frames -> {dst}", flush=True)


def run(run_name, subset, out, kdim=32, pool=14, m=1024, k=128):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dsreg
    from lewam.eval.loaders import load_lewam

    pool = int(pool)
    model, cfg = load_lewam(run_name)
    model = model.cuda().eval()
    mean = torch.tensor(cfg.get("input_norm_mean", [0.485, 0.456, 0.406]), device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("input_norm_std", [0.229, 0.224, 0.225]), device="cuda").view(1, 3, 1, 1)
    f = h5py.File(subset, "r")
    lat, obs = [], []
    with torch.no_grad():
        for i in range(0, len(f["pixels"]), 128):
            px = torch.from_numpy(f["pixels"][i:i + 128]).cuda().permute(0, 3, 1, 2).float() / 255.0
            lat.append(model.encode((px - mean) / std).float().cpu())
            obs.append(torch.nn.functional.avg_pool2d(px, pool).flatten(1).cpu())
    lat, x = torch.cat(lat).numpy(), torch.cat(obs).numpy().astype(np.float64)
    yaw = f["privileged_block_0_yaw"][:, 0]
    y = np.concatenate([f["proprio_joint_pos"][:], f["proprio_gripper_opening"][:], f["privileged_block_0_pos"][:],
                        np.sin(4 * yaw)[:, None], np.cos(4 * yaw)[:, None]], 1).astype(np.float64)
    dsreg.run(lat, x, y, f["ep_idx"][:], NAMES, out, kdim, m, k, info=dict(run=run_name, data=subset, pool=pool))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    (extract if cmd == "extract" else run)(*args)

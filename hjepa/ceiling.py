"""Image ceiling: how well can a supervised CNN find the cube and the grasp point in our frames?

    sh scripts/uvr python hjepa/ceiling.py TRAIN.h5 VAL.h5 OUT.json [epochs]

The upper bound for any encoder at this resolution (64x128: scene | wrist). A small CNN (no global
pooling, so it keeps where things are) regresses the cube and the grasp point from one frame;
trained on every 3rd train frame, scored on the val episodes: RMS error (cm), also on the frames
where the cube rests on the table (only vision can show it there). CPU is enough.
"""

import json
import os
import sys
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch import nn

CUBE_HALF = 0.0125  # sim/scene.py
torch.set_num_threads(int(float(os.environ.get("FLEET_CPUS", os.cpu_count()))))


def load(path, step):
    f = h5py.File(path, "r")
    n = len(f["pixels"])
    px, k = np.empty((len(range(0, n, step)), *f["pixels"].shape[1:]), np.uint8), 0
    for i in range(0, n, 3000 * step):  # one copy in memory: fill block by block
        blk = f["pixels"][i:i + 3000 * step][::step]
        px[k:k + len(blk)] = blk
        k += len(blk)
    y = np.concatenate([f["cube_pos"][::step], f["ee"][::step]], 1).astype(np.float32)
    return px, y


def net():
    def block(i, o, k):
        return [nn.Conv2d(i, o, k, 2, k // 2), nn.BatchNorm2d(o), nn.ReLU()]
    return nn.Sequential(*block(3, 32, 5), *block(32, 64, 3), *block(64, 128, 3), *block(128, 128, 3),
                         nn.Flatten(), nn.Linear(128 * 4 * 8, 256), nn.ReLU(), nn.Linear(256, 6))


def batches(px, y, idx, bs):
    for i in range(0, len(idx), bs):
        j = np.sort(idx[i:i + bs])
        yield torch.from_numpy(px[j]).permute(0, 3, 1, 2).float() / 255.0, torch.from_numpy(y[j])


def main(train, val, out, epochs=6):
    t0 = time.time()
    px, y = load(train, 3)
    vpx, vy = load(val, 1)
    mu, sd = y.mean(0), y.std(0)
    model, rng = net(), np.random.default_rng(0)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(px) // 256 + 1))
    print(f"loaded {len(px)} train and {len(vpx)} val frames in {time.time() - t0:.0f} s", flush=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in batches(px, (y - mu) / sd, rng.permutation(len(px)), 256):
            loss = ((model(xb) - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {ep + 1}: train loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
    model.eval()
    with torch.no_grad():
        pred = np.concatenate([model(xb).numpy() for xb, _ in batches(vpx, vy, np.arange(len(vpx)), 512)]) * sd + mu
    err = pred - vy
    rms = lambda e: round(float(np.sqrt((e ** 2).sum(1).mean()) * 100), 2)
    table = vy[:, 2] < CUBE_HALF + 0.003
    report = dict(train_frames=len(px), val_frames=len(vpx), epochs=epochs,
                  cube_rms_cm=rms(err[:, :3]), cube_on_table_rms_cm=rms(err[table, :3]),
                  cube_on_table_xy_rms_cm=rms(err[table, :2]), grasp_point_rms_cm=rms(err[:, 3:]),
                  cube_rel_grasp_rms_cm=rms(err[:, :3] - err[:, 3:]),
                  spread_cm=dict(cube=rms(vy[:, :3] - vy[:, :3].mean(0)), grasp_point=rms(vy[:, 3:] - vy[:, 3:].mean(0))))
    print(json.dumps(report, indent=1))
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4], *(int(a) for a in sys.argv[4:5]))

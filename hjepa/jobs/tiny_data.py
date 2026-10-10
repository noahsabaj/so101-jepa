"""A tiny LeWAM-layout h5 for tests: 40 episodes of 60 frames, 224 px, one camera. A coloured square moves by the
2-d action, so the action is learnable from the frames.   python hjepa/jobs/tiny_data.py OUT.h5"""
import sys

import h5py
import numpy as np

E, T, S = 40, 60, 224
rng = np.random.default_rng(0)
pixels, action = np.zeros((E * T, S, S, 3), np.uint8), np.zeros((E * T, 2), np.float32)
for e in range(E):
    p, colour = rng.uniform(40, 184, 2), rng.integers(64, 256, 3)
    for t in range(T):
        y, x = p.astype(int)
        pixels[e * T + t, y - 12:y + 12, x - 12:x + 12] = colour
        action[e * T + t] = a = rng.normal(0, 3, 2)
        p = np.clip(p + a, 20, 204)
with h5py.File(sys.argv[1], "w") as f:
    f["pixels"], f["action"] = pixels, action
    f["ep_offset"], f["ep_len"] = np.arange(E) * T, np.full(E, T)

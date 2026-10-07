"""Mix the community SO-100/SO-101 episodes into the sim training set (PLAN.md A24).

    uv run python hjepa/mix_community.py SIM_TRAIN.h5 COMMUNITY_TRAIN.h5 OUT.h5

The two sources differ in layout; the mix keeps each source's own meaning and adds no rule about
what the cameras show:
- pixels: a community frame (one camera, 64x64) fills the left half of a 64x128 frame and the
  right half is black; sim frames (scene | wrist) stay as they are. The model learns that a black
  wrist half means a one-camera robot.
- action and proprio: each source is z-scored with its own statistics (community-1 already is,
  per dataset), so neither source dominates the training normalizer.
Columns: pixels, action, proprio, ep_len, ep_offset, ep_idx, source (0 sim, 1 community).
"""

import sys
import time

import h5py
import hdf5plugin
import numpy as np

IMAGE_COMPRESSION = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
BLOCK = 20000  # frames per copy


def main(sim, community, out):
    t0 = time.time()
    with h5py.File(sim, "r") as s, h5py.File(community, "r") as c, h5py.File(out, "w") as f:
        n_s, n_c = len(s["action"]), len(c["action"])
        n = n_s + n_c
        f.create_dataset("pixels", (n, 64, 128, 3), np.uint8, chunks=(10, 64, 128, 3), compression=IMAGE_COMPRESSION)
        for k in ("action", "proprio"):
            f.create_dataset(k, (n, 6), np.float32, chunks=(1000, 6))
        f.create_dataset("ep_idx", (n,), np.int32, chunks=(1000,))
        f.create_dataset("source", (n,), np.int8, chunks=(1000,))
        stats = {k: (s[k][:].mean(0), s[k][:].std(0, ddof=1) + 1e-6) for k in ("action", "proprio")}
        for i in range(0, n_s, BLOCK):
            j = min(n_s, i + BLOCK)
            f["pixels"][i:j] = s["pixels"][i:j]
            for k, (mu, sd) in stats.items():
                f[k][i:j] = (s[k][i:j] - mu) / sd
        f["ep_idx"][:n_s], f["source"][:n_s] = s["ep_idx"][:], 0
        e_s = len(s["ep_len"])
        for i in range(0, n_c, BLOCK):
            j = min(n_c, i + BLOCK)
            px = np.zeros((j - i, 64, 128, 3), np.uint8)
            px[:, :, :64] = c["pixels"][i:j]
            f["pixels"][n_s + i:n_s + j] = px
            for k in ("action", "proprio"):
                f[k][n_s + i:n_s + j] = c[k][i:j]
            print(f"FLEET_PROGRESS {j}/{n_c} community frames ({time.time() - t0:.0f} s)", flush=True)
        f["ep_idx"][n_s:], f["source"][n_s:] = c["ep_idx"][:] + e_s, 1
        f["ep_len"] = np.concatenate([s["ep_len"][:], c["ep_len"][:]]).astype(np.int32)
        f["ep_offset"] = np.concatenate([s["ep_offset"][:], c["ep_offset"][:] + n_s]).astype(np.int64)
    print(f"wrote {out}: {n_s} sim + {n_c} community frames in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])

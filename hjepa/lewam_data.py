"""Our HDF5 data in LeWAM's layout (PLAN.md A25).

    sh scripts/uvr python hjepa/lewam_data.py IN.h5 OUT.h5

LeWAM reads one image column per camera (channels last, uint8), action, ep_len and ep_offset. Our frame
is the scene and wrist views side by side (64 x 128); OUT gets them as pixels_scene and pixels_wrist
(64 x 64 each), uncompressed with one frame per chunk, so LeWAM's loader (data.source=h5) reads a frame
without decompressing. The action is copied (6 joint-target changes, rad); proprio is left out (vision
only, A17). OUT is written under a temporary name and renamed when complete.
"""

import os
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from lewam_common import VIEWS, split_views

ROWS = 4096


def main(src, out):
    with h5py.File(src, "r") as f, h5py.File(out + ".tmp", "w") as g:
        n, h, w, c = f["pixels"].shape
        cols = [g.create_dataset(v, (n, h, w // 2, c), np.uint8, chunks=(1, h, w // 2, c)) for v in VIEWS]
        for i in range(0, n, ROWS):
            for col, view in zip(cols, split_views(f["pixels"][i:i + ROWS])):
                col[i:i + len(view)] = view
            print(f"lewam_data: {min(i + ROWS, n)}/{n}", flush=True)
        for k in ("action", "ep_len", "ep_offset"):
            g.create_dataset(k, data=f[k][:])
    os.replace(out + ".tmp", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])

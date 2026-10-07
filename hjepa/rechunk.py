"""Rewrite an H-JEPA HDF5 file with small image chunks, in place.

    sh scripts/uvr python hjepa/rechunk.py DATA.h5 [FRAMES]   # default 10 frames per chunk

The training loader reads short random clips. With 100-frame chunks each clip decompresses
1-2 whole chunks (up to 5 MB); with 10-frame chunks, about 0.5 MB.
"""

import os
import sys
from pathlib import Path

import h5py
import hdf5plugin

BLOSC = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
BLOCK = 10_000  # frames copied at a time


def rechunk(path, frames=10):
    path = Path(path)
    tmp = path.with_suffix(".rechunk.tmp")
    with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
        for key, d in src.items():
            if d.ndim == 4:  # images (N, H, W, 3)
                out = dst.create_dataset(key, d.shape, d.dtype, chunks=(frames, *d.shape[1:]), compression=BLOSC)
                for i in range(0, len(d), BLOCK):
                    out[i:i + BLOCK] = d[i:i + BLOCK]
                    print(f"FLEET_PROGRESS {min(i + BLOCK, len(d))}/{len(d)} {key}", flush=True)
            else:
                src.copy(d, dst, name=key)
        dst.attrs.update(src.attrs)
    os.replace(tmp, path)


if __name__ == "__main__":
    rechunk(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)

"""The first N episodes of a dataset, as a new dataset (PLAN.md A18: data scaling).

    uv run python hjepa/subset.py DATA.h5 OUT.h5 N
"""

import sys

import h5py
import hdf5plugin

IMAGE_COMPRESSION = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)


def main(data, out, n):
    n = int(n)
    with h5py.File(data, "r") as f, h5py.File(out, "w") as g:
        end = int(f["ep_offset"][n - 1] + f["ep_len"][n - 1])
        for k in f:
            rows = n if k in ("ep_len", "ep_offset") else end
            chunks = tuple(min(c, rows if i == 0 else c) for i, c in enumerate(f[k].chunks))
            kw = dict(IMAGE_COMPRESSION) if f[k].ndim == 4 else {}  # pixels: the Blosc filter of sim/collect.py
            g.create_dataset(k, data=f[k][:rows], chunks=chunks, **kw)
    print(f"wrote {out}: {n} episodes, {end} frames", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])

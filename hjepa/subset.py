"""The first N episodes of a dataset, as a new dataset (PLAN.md A18: data scaling).

    uv run python hjepa/subset.py DATA.h5 OUT.h5 N
"""

import sys

import h5py
import hdf5plugin  # noqa: F401


def main(data, out, n):
    n = int(n)
    with h5py.File(data, "r") as f, h5py.File(out, "w") as g:
        end = int(f["ep_offset"][n - 1] + f["ep_len"][n - 1])
        for k in f:
            rows = n if k in ("ep_len", "ep_offset") else end
            g.create_dataset(k, data=f[k][:rows], chunks=f[k].chunks, compression=f[k].compression,
                             compression_opts=f[k].compression_opts)
    print(f"wrote {out}: {n} episodes, {end} frames", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])

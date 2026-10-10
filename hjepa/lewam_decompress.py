"""LeWAM's scripts/decompress_h5.py with many processes (PLAN.md A27): the same files (<decomp_dir>/<stem>/<column>.npy,
channels first, and meta.json), each process decompressing its own blocks of rows into the one memory-mapped file.
LeWAM's script runs in one process: ~300 GB of gzip-compressed frames (Cube, Push-T) would take hours of a lease.

    python hjepa/lewam_decompress.py DATA.h5 DECOMP_DIR [--views pixels] [--workers 64]
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  the swm h5 files compress frames with plugin filters
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "lewam"))
from lewam.train.datasets import H5Frames  # noqa: E402

ROWS = 1024  # rows a task


def work(args):
    dataset_path, column, out_path, shape, channels_last, start = args
    out = np.load(out_path, mmap_mode="r+")
    with h5py.File(dataset_path, "r") as f:
        rows = f[column][start:min(start + ROWS, shape[0])]
    out[start:start + len(rows)] = rows.transpose(0, 3, 1, 2) if channels_last else rows
    out.flush()
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_path")
    ap.add_argument("decomp_dir")
    ap.add_argument("--views", default="pixels")
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count()))
    ap.add_argument("--max_rows", type=int, default=0, help="test only: the first rows (the trainer refuses the copy)")
    args = ap.parse_args()

    out_dir = Path(args.decomp_dir) / Path(args.dataset_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [H5Frames(args.dataset_path, v) for v in args.views.split(",") if v]
    if args.max_rows:
        for f in frames:
            f.shape = (min(args.max_rows, f.shape[0]), *f.shape[1:])
    needed, free = sum(int(np.prod(f.shape)) for f in frames), shutil.disk_usage(out_dir).free
    print(f"[decompress] {[(f.img_column, tuple(f.shape)) for f in frames]}: {needed / 1e9:.1f} GB in {out_dir} "
          f"({free / 1e9:.1f} GB free), {args.workers} processes", flush=True)
    assert needed < free, "not enough disk"
    for f in frames:
        final, partial = out_dir / f"{f.img_column}.npy", out_dir / f"{f.img_column}.npy.partial"
        if final.exists() and tuple(np.load(final, mmap_mode="r").shape) == tuple(f.shape):
            print(f"[decompress] {final} exists", flush=True)
            continue
        np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=tuple(f.shape)).flush()
        tasks = [(args.dataset_path, f.img_column, str(partial), tuple(f.shape), f.channels_last, start)
                 for start in range(0, f.shape[0], ROWS)]
        t, done = time.time(), 0
        with ProcessPoolExecutor(args.workers) as pool:
            for i, n in enumerate(pool.map(work, tasks, chunksize=4)):
                done += n
                if i % 200 == 0:
                    print(f"[decompress] {f.img_column}: {done}/{f.shape[0]} ({done / (time.time() - t):.0f} "
                          f"frames/s)", flush=True)
        assert done == f.shape[0], (done, f.shape)
        os.replace(partial, final)
        print(f"[decompress] wrote {final} in {time.time() - t:.0f} s", flush=True)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"views": {}}
    meta["source"] = os.path.abspath(args.dataset_path)
    meta["views"].update({f.img_column: list(f.shape) for f in frames})
    meta_path.write_text(json.dumps(meta, indent=1))
    print("[decompress] done", flush=True)


if __name__ == "__main__":
    main()

"""Check data/community/community_{train,test}.h5 against community_meta.json; save a grid of 64 random frames.

Needs h5py, hdf5plugin, numpy and pillow. Run from the project root on samsung-1: python community/verify.py
"""
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the Blosc filter)
import numpy as np
from PIL import Image, ImageDraw

OUT = Path("data/community")
FLOATS = ("proprio", "action", "state_raw", "action_raw")


def check(f, split, meta):
    """Assert shapes, dtypes, finite values, episode index consistency and the per-dataset counts and stats."""
    n, lens, offs = len(f["pixels"]), f["ep_len"][:], f["ep_offset"][:]
    cols = {k: f[k][:] for k in (*FLOATS, "dataset_id", "ep_idx")}
    assert f["pixels"].shape == (n, 64, 64, 3) and f["pixels"].dtype == np.uint8
    for k in FLOATS:
        assert cols[k].shape == (n, 6) and cols[k].dtype == np.float32 and np.isfinite(cols[k]).all(), k
    assert cols["dataset_id"].shape == cols["ep_idx"].shape == (n,)
    assert cols["dataset_id"].dtype == cols["ep_idx"].dtype == lens.dtype == np.int32 and offs.dtype == np.int64
    assert offs[0] == 0 and (offs[1:] == offs[:-1] + lens[:-1]).all() and lens.sum() == n and lens.min() >= 16
    assert (cols["ep_idx"] == np.repeat(np.arange(len(lens)), lens)).all()
    ds = cols["dataset_id"]
    assert (ds[offs] == ds[offs + lens - 1]).all() and (np.diff(ds) >= 0).all()  # whole episodes, by dataset
    ep_ds = ds[offs]
    clip = meta["z_clip"]
    for i in np.unique(ds):
        info, rows = meta["datasets"][str(i)], ds == i
        assert info["split"] == split and info["episodes"]["kept"] == (ep_ds == i).sum() and info["steps"] == rows.sum()
        s, st, dt = cols["state_raw"][rows], info["state_stats"], info["delta_stats"]
        delta = (cols["action_raw"][rows] - s + 180.0) % 360.0 - 180.0  # wrapped, as in build.py
        assert np.allclose(cols["proprio"][rows], np.clip((s - st["mean"]) / np.array(st["std"]), -clip, clip), atol=1e-3)
        assert np.allclose(cols["action"][rows], np.clip((delta - dt["mean"]) / np.array(dt["std"]), -clip, clip), atol=1e-3)
    counts = dict(datasets=len(np.unique(ds)), episodes=len(lens), steps=n)
    assert counts == meta["splits"][split], (counts, meta["splits"][split])
    print(f"{split}: {counts}, episode steps min/median/max {lens.min()}/{int(np.median(lens))}/{lens.max()}")
    for k in FLOATS:
        print(f"  {k:10s} min {cols[k].min(0).round(1)} max {cols[k].max(0).round(1)}")


def frame_health(f, rng, k=5_000):
    """Count blank (near-constant) and green-dominant frames (as failed decodes look) among k random ones."""
    idx = np.sort(rng.choice(len(f["pixels"]), min(k, len(f["pixels"])), replace=False))
    px = f["pixels"][idx].reshape(len(idx), -1, 3).astype(np.float32)
    rgb = px.mean(1)
    blank = int((px.std((1, 2)) < 3).sum())
    green = int((rgb[:, 1] - rgb[:, [0, 2]].mean(1) > 60).sum())
    print(f"  {len(idx)} random frames: {blank} blank, {green} green-dominant, mean RGB {rgb.mean(0).round(1)}")


def main():
    meta = json.loads((OUT / "community_meta.json").read_text())
    rng = np.random.default_rng(0)
    tiles = []
    for split, k in (("train", 56), ("test", 8)):
        with h5py.File(OUT / f"community_{split}.h5", "r") as f:
            check(f, split, meta)
            frame_health(f, rng)
            idx = np.sort(rng.choice(len(f["pixels"]), k, replace=False))
            mark = "T" if split == "test" else ""
            tiles += [(px, f"{d}{mark}") for px, d in zip(f["pixels"][idx], f["dataset_id"][idx])]
    failed = {i: d["error"] for i, d in meta["datasets"].items() if "error" in d}
    print(f"failed datasets: {failed or 'none'}")
    grid = Image.new("RGB", (8 * 128, 8 * 142), "white")
    draw = ImageDraw.Draw(grid)
    for j, (px, label) in enumerate(tiles):
        x, y = j % 8 * 128, j // 8 * 142
        grid.paste(Image.fromarray(px).resize((128, 128), Image.NEAREST), (x, y))
        draw.text((x + 4, y + 129), f"dataset {label}", fill="black")
    grid.save(OUT / "sample_grid.png")
    print(f"saved {OUT / 'sample_grid.png'} (T = test split)")


if __name__ == "__main__":
    main()

"""Build a 5 Hz, 64x64 training set from LeRobot's community SO-100/SO-101 data.

Source: lerobot/community_dataset_v3 on Hugging Face (Apache-2.0, LeRobot format v3.0).
Needs huggingface_hub, pyarrow, av, h5py, hdf5plugin, numpy and pillow.
Run from the project root on samsung-1, one step after the other:

    python community/build.py sample     # community/sample.json: dataset, camera and split
    python community/build.py download   # data/community/raw/ (~105 GB); a re-run skips finished files
    python community/build.py convert    # data/community_{train,test}.h5 (where the configs read them) + data/community/community_meta.json

Step k of an episode covers source frames [6k, 6k+6) (30 fps -> 5 Hz). pixels and state_raw come from
frame 6k, action_raw (the commanded target) from frame 6k+5 or the episode's last frame.
proprio = z(state_raw) and action = z(action_raw - state_raw), with each dataset's own mean and std.
Datasets in EXCLUDE (data errors) are left out.
"""
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from itertools import groupby
from pathlib import Path

import av
import h5py
import hdf5plugin
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

os.environ.setdefault("HF_HOME", "data/community/hf")  # keep Hugging Face's caches in the project
from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.utils import disable_progress_bars  # noqa: E402

REPO = "lerobot/community_dataset_v3"
REVISION = "ab92ac3fa4d0c08336e21b8e51cb9dc279535539"  # main on 2026-10-06
HERE = Path(__file__).parent
OUT = Path("data/community")
RAW, SHARDS = OUT / "raw", OUT / "shards"
MIN_EPISODES = 15_000
STRIDE = 6  # source frames per step: 30 fps -> 5 Hz
SIZE = 64  # image side
MIN_STEPS = 16  # 3.2 s: one H-JEPA clip (level 2: 4 steps x frameskip 5)
# Per-dataset z-scores. A joint that hardly moves in a dataset has a tiny std, and its rare moves
# then get z of 30 or more, so the std has a floor (degrees; gripper units), and z is clipped.
STATE_STD_FLOOR, DELTA_STD_FLOOR, Z_CLIP = 5.0, 1.0, 10.0
# A shard is reused only if it was made with these constants.
SETTINGS = dict(stride=STRIDE, size=SIZE, min_steps=MIN_STEPS, state_std_floor=STATE_STD_FLOOR,
                delta_std_floor=DELTA_STD_FLOOR, z_clip=Z_CLIP)
# Source datasets with data errors, found by the first build (2026-10-06) and left out of the files.
EXCLUDE = {
    "Ryosei2/0704_donut3": "100% black frames",
    "cerealkiller2527/so101_rayban_quality_assessment_001": "37% black frames",
    "Ryosei2/0704_donut2": "20% black frames",
    "Cornito/so101_chess4": "joints and video frozen",
    "luriss/so100_pick_and_place_box_task_0428": "shoulder_lift wraps past 360 degrees",
    "luriss/so100_pick_and_place_box_task_0430": "shoulder_lift wraps past 360 degrees",
    "romanovvv1987/so100_test3": "shoulder_lift wraps past 360 degrees",
    "hhh6/so101_test8": "shoulder_lift wraps past 360 degrees",
    "kantine/flip_A1": "42% of steps have |action - state| > 45: leader and follower calibrations differ",
    "Hugo-Castaing/LAR_DEMO": "42% of steps have |action - state| > 45: leader and follower calibrations differ",
}


def load_sample():
    return json.loads((HERE / "sample.json").read_text())


def sample():
    """Shuffle the selection (seed 0), take datasets until >= 15,000 episodes, hold out ~10% of them."""
    index = {d["dataset"]: d for d in json.loads((HERE / "cds_v3_index.json").read_text())}
    pairs = json.loads((HERE / "selection_all.json").read_text())
    picked, episodes = [], 0
    for i in np.random.default_rng(0).permutation(len(pairs)):
        picked.append(pairs[i])
        episodes += index[pairs[i][0]]["episodes"]
        if episodes >= MIN_EPISODES:
            break
    test = set(np.random.default_rng(0).choice(len(picked), round(0.1 * len(picked)), replace=False).tolist())
    rows = sorted(
        (dict(dataset=d, camera=c, split="test" if i in test else "train",
              episodes=index[d]["episodes"], video_gb=index[d]["video_gb"][c])
         for i, (d, c) in enumerate(picked)),
        key=lambda r: r["dataset"])
    (HERE / "sample.json").write_text(json.dumps(rows, indent=1) + "\n")
    for split in ("train", "test"):
        part = [r for r in rows if r["split"] == split]
        print(f"{split}: {len(part)} datasets, {sum(r['episodes'] for r in part)} episodes")
    video = sum(r["video_gb"] for r in rows)
    other = sum(index[r["dataset"]]["gb"] - sum(index[r["dataset"]]["video_gb"].values()) for r in rows)
    print(f"expected download: {video + other:.1f} GB ({video:.1f} GB video, {other:.2f} GB data+meta)")


def raw_bytes():
    return sum(f.stat().st_size for f in RAW.rglob("*") if f.is_file()) if RAW.exists() else 0


def download(batch=40):
    """Fetch meta/, data/ and the chosen camera's videos/ of each sampled dataset; re-runs skip finished files."""
    disable_progress_bars()
    rows = load_sample()
    start, t0 = raw_bytes(), time.time()
    for i in range(0, len(rows), batch):
        patterns = [p for r in rows[i:i + batch] for p in
                    (f"{r['dataset']}/meta/**", f"{r['dataset']}/data/**", f"{r['dataset']}/videos/{r['camera']}/**")]
        for attempt in range(5):  # anonymous requests can hit Hub rate limits: wait and resume
            try:
                snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=RAW,
                                  allow_patterns=patterns, max_workers=8)
                break
            except Exception as e:
                print(f"batch {i // batch}: {e!r}; retry in 2 min", flush=True)
                time.sleep(120)
        else:
            raise RuntimeError(f"batch {i // batch} failed 5 times")
        done = raw_bytes()
        print(f"FLEET_PROGRESS {min(i + batch, len(rows))}/{len(rows)}  raw {done / 1e9:.1f} GB, "
              f"{(done - start) / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)


def decode_steps(container, out, t0, t1, fps):
    """Fill out with 64x64 RGB frames 0, 6, 12, ... of the video segment [t0, t1) s; False if any is missing."""
    stream = container.streams.video[0]
    seen = np.zeros(len(out), bool)
    try:
        container.seek(round(t0 / stream.time_base), stream=stream)  # the keyframe at or before t0
        for frame in container.decode(stream):
            k = round((frame.time - t0) * fps)
            if k > STRIDE * (len(out) - 1) or frame.time >= t1 - 0.5 / fps:
                break
            if k >= 0 and k % STRIDE == 0:
                out[k // STRIDE] = np.asarray(frame.to_image().resize((SIZE, SIZE), Image.Resampling.BOX))
                seen[k // STRIDE] = True
    except av.error.FFmpegError:
        return False
    return seen.all()


def wrap(d):
    """Angle differences into [-180, 180) degrees: a reading that wrapped past 360 (59 train steps
    of the 2026-10-06 build had |action - state| > 180) is not a 360-degree move."""
    return (d + 180.0) % 360.0 - 180.0


def zscore(x, floor):
    mean, std = x.mean(0), np.maximum(x.std(0), floor)
    z = np.clip((x - mean) / std, -Z_CLIP, Z_CLIP)
    return z.astype(np.float32), dict(mean=mean.tolist(), std=std.tolist())


def normalize(s, a):
    """proprio and action columns (and their stats) from raw states and leader actions of one dataset."""
    proprio, state_stats = zscore(s, STATE_STD_FLOOR)
    action, delta_stats = zscore(wrap(a - s), DELTA_STD_FLOOR)
    return proprio, action, state_stats, delta_stats


def convert_dataset(i, row):
    """Convert one dataset's episodes to 5 Hz steps in shards/<i>.npz and return its summary."""
    shard = SHARDS / f"{i:04d}.npz"
    if shard.exists():  # written by an earlier run that stopped before the merge; reused if made the same way
        with np.load(shard) as z:
            info = json.loads(str(z["info"]))
        if info.get("settings") == SETTINGS:
            return info
    root, cam, v = RAW / row["dataset"], row["camera"], f"videos/{row['camera']}/"
    info = json.loads((root / "meta/info.json").read_text())
    fps = info["fps"]
    assert fps == 30, fps
    eps = pa.concat_tables([
        pq.read_table(p, columns=["episode_index", v + "chunk_index", v + "file_index",
                                  v + "from_timestamp", v + "to_timestamp"])
        for p in sorted(root.glob("meta/episodes/*/*.parquet"))]).to_pylist()
    data = pa.concat_tables([pq.read_table(p, columns=["episode_index", "frame_index", "observation.state", "action"])
                             for p in sorted(root.glob("data/*/*.parquet"))])
    state, action = (np.asarray(data[k].combine_chunks().flatten(), np.float64).reshape(len(data), 6)
                     for k in ("observation.state", "action"))
    ep, frame = data["episode_index"].to_numpy(), data["frame_index"].to_numpy()
    order = np.lexsort((frame, ep))
    ep_sorted = ep[order]

    drop, todo = dict(short=0, data=0, nonfinite=0, video=0), []
    for e in sorted(eps, key=lambda e: e["episode_index"]):
        k = e["episode_index"]
        # The episode's data rows decide its length (some meta/episodes "length" values are stale).
        ix = order[np.searchsorted(ep_sorted, k):np.searchsorted(ep_sorted, k, "right")]
        n = -(-len(ix) // STRIDE)
        if n < MIN_STEPS:
            drop["short"] += 1
        elif (frame[ix] != np.arange(len(ix))).any():
            drop["data"] += 1
        else:
            s, a = state[ix[::STRIDE]], action[ix[np.minimum(np.arange(n) * STRIDE + STRIDE - 1, len(ix) - 1)]]
            if np.isfinite(s).all() and np.isfinite(a).all():
                path = root / info["video_path"].format(
                    video_key=cam, chunk_index=e[v + "chunk_index"], file_index=e[v + "file_index"])
                todo.append((path, e, s, a))
            else:
                drop["nonfinite"] += 1

    pixels, kept, m, video = np.empty((sum(len(t[2]) for t in todo), SIZE, SIZE, 3), np.uint8), [], 0, None
    for path, group in groupby(todo, key=lambda t: t[0]):
        group = list(group)
        try:
            container = av.open(str(path))
        except av.error.FFmpegError:
            drop["video"] += len(group)
            continue
        with container:
            cc = container.streams.video[0].codec_context
            cc.thread_count = 1  # one decoder thread per worker process
            video = video or f"{cc.name} {cc.width}x{cc.height}"
            for _, e, s, a in group:
                if decode_steps(container, pixels[m:m + len(s)], e[v + "from_timestamp"], e[v + "to_timestamp"], fps):
                    kept.append((s, a))
                    m += len(s)
                else:
                    drop["video"] += 1

    summary = dict(name=row["dataset"], camera=cam, split=row["split"], fps=fps, settings=SETTINGS,
                   robot_type=info.get("robot_type"), state_names=info["features"]["observation.state"].get("names"), video=video,
                   episodes=dict(source=len(eps), kept=len(kept), **drop), steps=m)
    arrays = {}
    if kept:
        s, a = (np.concatenate(x) for x in zip(*kept))
        proprio, act, summary["state_stats"], summary["delta_stats"] = normalize(s, a)
        arrays = dict(pixels=pixels[:m], proprio=proprio, action=act, state_raw=s.astype(np.float32),
                      action_raw=a.astype(np.float32), ep_len=np.array([len(k[0]) for k in kept], np.int32))
    SHARDS.mkdir(parents=True, exist_ok=True)
    tmp = shard.with_suffix(".tmp.npz")
    np.savez(tmp, info=np.array(json.dumps(summary)), **arrays)
    os.replace(tmp, shard)
    return summary


def merge(rows, summaries):
    """Write community_{train,test}.h5 (ordered by dataset, then episode) and community_meta.json; drop the shards."""
    blosc = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    splits = {}
    for split in ("train", "test"):
        ids = [i for i in sorted(summaries) if rows[i]["split"] == split and summaries[i]["steps"]]
        n = sum(summaries[i]["steps"] for i in ids)
        n_ep = sum(summaries[i]["episodes"]["kept"] for i in ids)
        path = OUT.parent / f"community_{split}.h5"
        with h5py.File(path.with_suffix(".tmp"), "w") as f:
            def column(name, shape, dtype, chunk_rows=1000, **kw):
                f.create_dataset(name, shape, dtype, chunks=(min(chunk_rows, shape[0]),) + shape[1:], **kw)
            column("pixels", (n, SIZE, SIZE, 3), np.uint8, 10, compression=blosc)  # 10 frames: cheap random clips
            for k in ("proprio", "action", "state_raw", "action_raw"):
                column(k, (n, 6), np.float32)
            column("dataset_id", (n,), np.int32)
            column("ep_idx", (n,), np.int32)
            column("ep_len", (n_ep,), np.int32)
            column("ep_offset", (n_ep,), np.int64)
            row = ep = 0
            for i in ids:
                with np.load(SHARDS / f"{i:04d}.npz") as z:
                    lens = z["ep_len"]
                    m, e = int(lens.sum()), len(lens)
                    for k in ("pixels", "proprio", "action", "state_raw", "action_raw"):
                        f[k][row:row + m] = z[k]
                f["dataset_id"][row:row + m] = i
                f["ep_idx"][row:row + m] = np.repeat(np.arange(ep, ep + e), lens)
                f["ep_len"][ep:ep + e] = lens
                f["ep_offset"][ep:ep + e] = row + np.cumsum(lens) - lens
                row, ep = row + m, ep + e
        os.replace(path.with_suffix(".tmp"), path)
        splits[split] = dict(datasets=len(ids), episodes=n_ep, steps=n)
        print(f"{path}: {len(ids)} datasets, {n_ep} episodes, {n} steps, {path.stat().st_size / 1e9:.2f} GB")
    meta = dict(source=REPO, revision=REVISION, license="Apache-2.0", step_hz=5, frame_stride=STRIDE,
                image_size=SIZE, min_steps=MIN_STEPS, state_std_floor=STATE_STD_FLOOR,
                delta_std_floor=DELTA_STD_FLOOR, z_clip=Z_CLIP, splits=splits,
                datasets={str(i): summaries[i] for i in sorted(summaries)})
    (OUT / "community_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    shutil.rmtree(SHARDS)


def convert():
    """Convert the datasets in parallel worker processes (largest first), then merge their shards."""
    rows, summaries, t0 = load_sample(), {}, time.time()
    for i, r in enumerate(rows):
        if r["dataset"] in EXCLUDE:
            summaries[i] = dict(name=r["dataset"], camera=r["camera"], split=r["split"],
                                excluded=EXCLUDE[r["dataset"]], steps=0)
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        futures = {pool.submit(convert_dataset, i, r): i
                   for i, r in sorted(enumerate(rows), key=lambda t: -t[1]["video_gb"]) if i not in summaries}
        for done, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            try:
                summaries[i] = future.result()
            except BrokenProcessPool:
                raise
            except Exception as e:  # a broken dataset: record why and go on
                summaries[i] = dict(name=rows[i]["dataset"], camera=rows[i]["camera"], split=rows[i]["split"],
                                    error=repr(e), steps=0)
            s = summaries[i]
            print(f"FLEET_PROGRESS {done}/{len(rows)}  {s['name']}: {s.get('episodes', s.get('error'))}, "
                  f"{s['steps']} steps, {time.time() - t0:.0f} s", flush=True)
    merge(rows, summaries)
    print(f"convert: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    {"sample": sample, "download": download, "convert": convert}[sys.argv[1]]()

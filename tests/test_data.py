"""Data pipeline and job scripts: community shard identity, dataset merge and shard counts, self-play
round failures and device lists, lane cancellation, artifact waiting (codex audit findings 5, 14, 15, 16, 17, 18, 19, 22).

The shell tests run the real scripts in a copy of the tree with a fake `uv` (no simulation, no GPU)."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

import collect

ROOT = Path(__file__).resolve().parents[1]
SH = shutil.which("sh")
needs_sh = pytest.mark.skipif(SH is None, reason="needs a POSIX sh")


def _load_build():
    stub = None
    try:  # build.py imports av (video decoding); these tests never decode
        import av  # noqa: F401
    except ImportError:
        stub = types.ModuleType("av")
        stub.error = types.SimpleNamespace(FFmpegError=Exception)
        sys.modules["av"] = stub
    spec = importlib.util.spec_from_file_location("community_build", ROOT / "community" / "build.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:  # other modules guard `import av` with ImportError: the stub must not outlive build.py's import
        if stub is not None and sys.modules.get("av") is stub:
            del sys.modules["av"]
    return module


build = _load_build()


# --- 5: community shards ---------------------------------------------------------------------

def _row(dataset="a/b", split="train", camera="cam"):
    return dict(dataset=dataset, camera=camera, split=split, episodes=1, video_gb=0.0)


def _write_shard(path, row, steps=20):
    info = dict(name=row["dataset"], camera=row["camera"], split=row["split"], steps=steps,
                episodes=dict(kept=1), identity=build.identity(row))
    arrays = dict(pixels=np.zeros((steps, build.SIZE, build.SIZE, 3), np.uint8),
                  proprio=np.zeros((steps, 6), np.float32), action=np.zeros((steps, 6), np.float32),
                  state_raw=np.zeros((steps, 6), np.float32), action_raw=np.zeros((steps, 6), np.float32),
                  ep_len=np.array([steps], np.int32))
    np.savez(path, info=np.array(json.dumps(info)), **arrays)
    return info


@pytest.fixture
def shards(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "SHARDS", tmp_path / "shards")
    monkeypatch.setattr(build, "RAW", tmp_path / "raw")  # empty: converting a dataset for real fails
    monkeypatch.setattr(build, "OUT", tmp_path / "data" / "community")
    (tmp_path / "shards").mkdir()
    (tmp_path / "data" / "community").mkdir(parents=True)
    return tmp_path / "shards"


def test_shard_is_reused_only_for_the_same_identity(shards):
    row = _row(split="train")
    info = _write_shard(shards / "0000.npz", row)
    assert build.convert_dataset(0, row) == info
    for other in (_row(split="test"), _row(dataset="c/d"), _row(camera="other")):
        with pytest.raises(Exception):  # not reused: it tries to convert from the (missing) raw files
            build.convert_dataset(0, other)


def test_shard_from_another_revision_is_not_reused(shards, monkeypatch):
    row = _row()
    _write_shard(shards / "0000.npz", row)
    monkeypatch.setattr(build, "REVISION", "0" * 40)
    with pytest.raises(Exception):
        build.convert_dataset(0, row)


def test_merge_checks_shard_identity(shards):
    old, new = _row(split="train"), _row(split="test")  # the sample moved dataset 0 from train to test
    train = _row("c/d", split="train")
    info = _write_shard(shards / "0000.npz", old)
    other = _write_shard(shards / "0001.npz", train)
    with pytest.raises(RuntimeError, match="not the shard"):
        build.merge([new, train], {0: dict(info, split="test"), 1: other})
    assert not (shards.parent / "data" / "community_test.h5").exists()


def test_merge_of_matching_shards_writes_the_split(shards):
    row, train = _row(split="test"), _row("c/d", split="train")
    info = _write_shard(shards / "0000.npz", row)
    build.merge([row, train], {0: info, 1: _write_shard(shards / "0001.npz", train)})
    with h5py.File(shards.parent / "data" / "community_test.h5") as f:
        assert f["ep_len"].shape == (1,)


# --- 15: collect.py merge --------------------------------------------------------------------

def _episode(n=4):
    return dict(pixels=np.zeros((n, 2, 2, 3), np.uint8), proprio=np.zeros((n, 6), np.float32),
                action=np.ones((n, 6), np.float32))


def _dataset(path, episodes):
    with h5py.File(path, "w") as f:
        for i in range(episodes):
            if i == 0:
                collect.create(f, _episode())
            collect.append(f, _episode())
    return str(path)


def _count(path):
    with h5py.File(path, "r") as f:
        return f["ep_len"].shape[0], f["action"].shape[0]


def test_merge_refuses_an_input_as_output(tmp_path):
    a, b = _dataset(tmp_path / "a.h5", 2), _dataset(tmp_path / "b.h5", 3)
    with pytest.raises(ValueError, match="also an input"):
        collect.merge(a, [b, a])
    assert _count(a) == (2, 8) and _count(b) == (3, 12)  # nothing was truncated


def test_merge_publishes_only_a_whole_file(tmp_path):
    a, b = _dataset(tmp_path / "a.h5", 2), _dataset(tmp_path / "b.h5", 3)
    out = str(tmp_path / "out.h5")
    collect.merge(out, [a, b])
    assert _count(out) == (5, 20)
    assert not os.path.exists(out + ".tmp")


def test_failed_merge_leaves_no_output(tmp_path):
    a = _dataset(tmp_path / "a.h5", 2)
    out = tmp_path / "out.h5"
    with pytest.raises(Exception):
        collect.merge(str(out), [a, str(tmp_path / "missing.h5")])
    assert not out.exists()


def test_count_reports_episodes(tmp_path):
    assert collect.count(_dataset(tmp_path / "a.h5", 3)) == 3


# --- shell tests: a copy of the tree with a fake uv -------------------------------------------

def _tree(tmp_path, scripts, collect_py=None):
    for name in ("scripts/gpu.sh", "scripts/uvr", *scripts):
        dst = tmp_path / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, dst)
    (tmp_path / "sim").mkdir(exist_ok=True)
    if collect_py:
        (tmp_path / "sim" / "collect.py").write_text(collect_py)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    return bin_dir


def _env(bin_dir, **extra):
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"], FAKE_PY=sys.executable,
               SO101_GPU="cuda", MUJOCO_GL="egl")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("GPUS", None)
    env.update(extra)
    return env


def _fake_uv(bin_dir, body):
    uv = bin_dir / "uv"
    uv.write_text("#!/bin/sh\nshift; shift  # run python\n" + body)
    uv.chmod(0o755)


FAKE_DATASET_COLLECT = '''
import sys
import h5py
import numpy as np

def write(path, n):
    with h5py.File(path, "w") as f:
        f["ep_len"] = np.full(n, 3, np.int32)
        f["ep_offset"] = np.arange(n, dtype=np.int64) * 3
        f["action"] = np.zeros((3 * n, 6), np.float32)

def count(path):
    with h5py.File(path, "r") as f:
        return f["ep_len"].shape[0]

if sys.argv[1] == "shard":
    write(sys.argv[2], int(sys.argv[4]))
elif sys.argv[1] == "count":
    print(*(count(p) for p in sys.argv[2:]))
else:
    write(sys.argv[2], sum(count(p) for p in sys.argv[3:]))
'''


# --- 14: make_dataset.sh ---------------------------------------------------------------------

def _make_dataset(tmp_path, bin_dir, train):
    return subprocess.run([SH, "sim/make_dataset.sh", str(train), "0", "2"], cwd=tmp_path, env=_env(bin_dir),
                          capture_output=True, text=True, timeout=120)


@needs_sh
def test_make_dataset_reruns_with_another_size(tmp_path):
    bin_dir = _tree(tmp_path, ["sim/make_dataset.sh"], FAKE_DATASET_COLLECT)
    _fake_uv(bin_dir, 'case "$1" in -c) exit 0 ;; esac  # import mujoco\nexec "$FAKE_PY" "$@"\n')
    out = tmp_path / "data" / "so101_train.h5"
    done = _make_dataset(tmp_path, bin_dir, 30)  # shards of 25 and 5
    assert done.returncode == 0, done.stderr
    assert _count_ep(out) == 30
    done = _make_dataset(tmp_path, bin_dir, 50)  # the 5-episode tail must be made again, as 25
    assert done.returncode == 0, done.stderr
    assert _count_ep(out) == 50
    done = _make_dataset(tmp_path, bin_dir, 30)  # fewer: the 25-episode tail is not a 5-episode shard
    assert done.returncode == 0, done.stderr
    assert _count_ep(out) == 30


def _count_ep(path):
    with h5py.File(path, "r") as f:
        return f["ep_len"].shape[0]


# --- 16, 17: self_improve.sh -----------------------------------------------------------------

FAKE_MERGE_COLLECT = '''
import sys
open("../merged.txt", "w").write(" ".join(sys.argv[2:]))
'''

FAKE_UV_PLAY = '''case "$1" in
  sim/self_play.py)  # ckpt eval goals out first n
    out=$5 first=$6 n=$7
    echo "$CUDA_VISIBLE_DEVICES" > "$out/gpu_$first.txt"
    [ -z "$FAKE_FAIL" ] || exit 3
    i=0
    while [ "$i" -lt "$n" ]; do : > "$out/$((first + i)).h5"; i=$((i + 1)); done ;;
  *) exec "$FAKE_PY" "$@" ;;
esac
'''


def _self_improve(tmp_path, bin_dir, **env):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "base.h5").write_text("")
    return subprocess.run([SH, "hjepa/self_improve.sh", "ckpt", "base", "round", "4", "2", "100"], cwd=tmp_path,
                          env=_env(bin_dir, **env), capture_output=True, text=True, timeout=120)


@needs_sh
def test_self_improve_merges_the_new_episodes(tmp_path):
    bin_dir = _tree(tmp_path, ["hjepa/self_improve.sh"], FAKE_MERGE_COLLECT)
    _fake_uv(bin_dir, FAKE_UV_PLAY)
    done = _self_improve(tmp_path, bin_dir)
    assert done.returncode == 0, done.stderr
    merged = (tmp_path / "merged.txt").read_text().split()
    assert merged[0] == "../data/round.h5" and merged[1] == "../data/base.h5" and len(merged) == 6


@needs_sh
def test_self_improve_fails_when_every_worker_fails(tmp_path):
    bin_dir = _tree(tmp_path, ["hjepa/self_improve.sh"], FAKE_MERGE_COLLECT)
    _fake_uv(bin_dir, FAKE_UV_PLAY)
    done = _self_improve(tmp_path, bin_dir, FAKE_FAIL="1")
    assert done.returncode != 0
    assert not (tmp_path / "merged.txt").exists()  # no merge: the output would be the base data alone


@needs_sh
def test_self_improve_splits_comma_separated_devices(tmp_path):
    bin_dir = _tree(tmp_path, ["hjepa/self_improve.sh"], FAKE_MERGE_COLLECT)
    _fake_uv(bin_dir, FAKE_UV_PLAY)
    done = _self_improve(tmp_path, bin_dir, CUDA_VISIBLE_DEVICES="2,3")
    assert done.returncode == 0, done.stderr
    seen = {(p.read_text().strip()) for p in (tmp_path / "data" / "self" / "round").glob("gpu_*.txt")}
    assert seen == {"2", "3"}


@needs_sh
def test_self_improve_refuses_the_same_input_and_output(tmp_path):
    bin_dir = _tree(tmp_path, ["hjepa/self_improve.sh"], FAKE_MERGE_COLLECT)
    _fake_uv(bin_dir, FAKE_UV_PLAY)
    (tmp_path / "data").mkdir()
    done = subprocess.run([SH, "hjepa/self_improve.sh", "ckpt", "same", "same", "4", "2", "100"], cwd=tmp_path,
                          env=_env(bin_dir), capture_output=True, text=True, timeout=120)
    assert done.returncode != 0


# --- 18: run_lanes.sh ------------------------------------------------------------------------

@needs_sh
@pytest.mark.skipif(shutil.which("pgrep") is None, reason="needs pgrep")
def test_run_lanes_forwards_cancellation_to_descendants(tmp_path):
    _tree(tmp_path, ["scripts/run_lanes.sh"])
    marker = f"{os.getpid()}.5"  # a sleep time that only this test uses
    (tmp_path / "lanes.txt").write_text(f'0 | sh -c "sleep {marker}; echo x" && echo y\n1 | sleep {marker}\n')
    runner = subprocess.Popen([SH, "scripts/run_lanes.sh", "lanes.txt"], cwd=tmp_path)
    try:
        for _ in range(50):
            time.sleep(0.2)
            if len(subprocess.run(["pgrep", "-f", f"sleep {marker}"], capture_output=True, text=True).stdout.split()) >= 2:
                break
        runner.terminate()
        runner.wait(timeout=30)
        time.sleep(0.5)
        left = subprocess.run(["pgrep", "-f", f"sleep {marker}"], capture_output=True, text=True).stdout.split()
        assert left == []
    finally:
        subprocess.run(["pkill", "-f", f"sleep {marker}"])
        runner.kill()


# --- 19: wait_for ----------------------------------------------------------------------------

def _wait_for(tmp_path, *args):
    return subprocess.run([SH, str(ROOT / "scripts" / "wait_for"), *args], cwd=tmp_path, capture_output=True,
                          text=True, timeout=60)


@needs_sh
def test_wait_for_accepts_a_published_file(tmp_path):
    (tmp_path / "a.pt").write_text("x")
    assert _wait_for(tmp_path, "a.pt", "0").returncode == 0


@needs_sh
def test_wait_for_does_not_accept_a_file_still_being_written(tmp_path):
    (tmp_path / "a.pt").write_text("x")
    (tmp_path / "a.pt.tmp").write_text("x")
    lanes = tmp_path / "outputs" / "lanes"
    lanes.mkdir(parents=True)
    (lanes / "0.end").write_text("1")  # the producer lane has ended
    done = _wait_for(tmp_path, "a.pt", "0")
    assert done.returncode == 1 and "ended without" in done.stderr


@needs_sh
def test_wait_for_stops_when_the_producer_process_is_gone(tmp_path):
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    done = _wait_for(tmp_path, "missing.pt", "9", "", str(gone.pid))
    assert done.returncode == 1 and "ended without" in done.stderr


# --- 22: check_planner.sh --------------------------------------------------------------------

@needs_sh
def test_check_planner_runs_every_check_and_reports_failure(tmp_path):
    bin_dir = _tree(tmp_path, ["hjepa/check_planner.sh"])
    _fake_uv(bin_dir, 'echo "$@" >> calls.txt\ncase "$*" in *fast*) exit 1 ;; esac\n')
    done = subprocess.run([SH, "hjepa/check_planner.sh"], cwd=tmp_path, env=_env(bin_dir), capture_output=True,
                          text=True, timeout=120)
    assert done.returncode != 0
    assert len((tmp_path / "calls.txt").read_text().splitlines()) == 3  # the check after the failure still ran

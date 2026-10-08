"""Make SO-101 episodes and write them in the H-JEPA HDF5 format.

    collect.py shard OUT.h5 FIRST_SEED N [expert|play]   # N episodes, seeds FIRST_SEED.., one process
    collect.py merge OUT.h5 SHARD.h5 ...                  # concatenate shards into one file

Policies: expert (scripted pick-and-place, hand-written waypoints) or play (no task knowledge: hold a
random joint target, uniform in the joint range, for 1 to 15 steps; PLAN.md A19).

Columns per step (5 Hz): pixels (64x128x3 uint8: scene | wrist), proprio (6 joint angles, rad),
action (6 joint-target changes, rad), and the true state for tests: ee (grasp point, m), cube_pos,
cube_quat, phase (index into PHASES). Plus ep_len, ep_offset, ep_idx (H-JEPA layout).
"""

import sys
import time

import h5py
import hdf5plugin
import numpy as np

from env import MAX_ACTION, SO101Env
from expert import GRASP_POINT, Expert

STEPS = 300  # 60 s per episode
PHASES = ["start", "approach", "descend", "grasp", "lift", "carry", "lower", "release", "retreat", "reach", "free"]
IMAGE_COMPRESSION = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)


class Play:
    """Motor babbling: no task, no waypoints. A random joint target (uniform in the joint range) is
    held for a random 1 to 15 steps; each step moves toward it as fast as the action limit allows."""

    phase = "free"

    def __init__(self, env, rng):
        self.env, self.rng, self.left = env, rng, 0

    def act(self):
        if self.left == 0:
            self.target = self.rng.uniform(self.env.ctrl_lo, self.env.ctrl_hi)
            self.left = int(self.rng.integers(1, 16))
        self.left -= 1
        return np.clip(self.target - self.env.joints(), -MAX_ACTION, MAX_ACTION)


def episode(env, seed, policy="expert"):
    obs = env.reset(seed)
    if policy not in ("expert", "play"):
        raise ValueError(f"unknown policy {policy!r}: expert or play")
    expert = (Play if policy == "play" else Expert)(env, np.random.default_rng([seed, 1]))  # not env.reset's stream
    cols = {k: [] for k in ["pixels", "proprio", "action", "ee", "cube_pos", "cube_quat", "phase"]}
    for _ in range(STEPS):
        action = expert.act()
        pos, rot = env.site_pose()
        cube = env.cube_pose()
        cols["pixels"].append(obs["pixels"])
        cols["proprio"].append(obs["proprio"])
        cols["action"].append(action.astype(np.float32))
        cols["ee"].append((pos + rot @ GRASP_POINT).astype(np.float32))
        cols["cube_pos"].append(cube[:3].astype(np.float32))
        cols["cube_quat"].append(cube[3:].astype(np.float32))
        cols["phase"].append(PHASES.index(expert.phase))
        obs = env.step(action)
    return {k: np.asarray(v) for k, v in cols.items()}


def create(f, sample):
    for k, v in sample.items():
        image = v.ndim == 4
        f.create_dataset(k, shape=(0, *v.shape[1:]), maxshape=(None, *v.shape[1:]), dtype=v.dtype,
                         chunks=(10 if image else 1000, *v.shape[1:]),  # 10 frames: cheap random clips
                         compression=IMAGE_COMPRESSION if image else None)
    f.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32)
    f.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)
    f.create_dataset("ep_idx", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1000,))


def append(f, ep):
    n, start, e = len(ep["action"]), f["action"].shape[0], f["ep_len"].shape[0]
    for k, v in {**ep, "ep_idx": np.full(n, e, np.int32)}.items():
        f[k].resize(start + n, axis=0)
        f[k][start:] = v
    for k, v in (("ep_len", n), ("ep_offset", start)):
        f[k].resize(e + 1, axis=0)
        f[k][e] = v


def shard(out, first_seed, n, policy="expert"):
    env = SO101Env(randomize=True)
    t0 = time.perf_counter()
    with h5py.File(out, "w") as f:
        for i in range(n):
            ep = episode(env, first_seed + i, policy)
            if i == 0:
                create(f, ep)
            append(f, ep)
            print(f"FLEET_PROGRESS {i + 1}/{n} ({(i + 1) * STEPS / (time.perf_counter() - t0):.0f} steps/s)", flush=True)
    env.renderer.close()


def merge(out, shards):
    with h5py.File(out, "w") as f:
        for i, path in enumerate(shards):
            with h5py.File(path, "r") as s:
                for e, (start, n) in enumerate(zip(s["ep_offset"][:], s["ep_len"][:])):
                    ep = {k: s[k][start:start + n] for k in s if k not in ("ep_len", "ep_offset", "ep_idx")}
                    if i == 0 and e == 0:
                        create(f, ep)
                    append(f, ep)
            print(f"merged {path}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "shard":
        shard(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), *sys.argv[5:6])
    else:
        merge(sys.argv[2], sys.argv[3:])

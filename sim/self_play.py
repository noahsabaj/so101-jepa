"""Self-collected data (PLAN.md A19): the planner plays toward goals sampled from earlier data.

    sh scripts/uvr python sim/self_play.py CKPT EVAL_CONFIG GOALS.h5 OUT.h5 FIRST_SEED N

No expert and no task: each episode (300 steps, 60 s) starts from a random scene (the seed), and
every 50 steps the goal becomes a random frame of GOALS.h5 (its arm joints and cube pose, rendered
in this scene). The planner drives toward it; every step is recorded in the H-JEPA HDF5 format of
sim/collect.py, so the episodes merge with the training data. Successes and failures both teach
the world model; the value learns from them with hindsight goals.
"""

import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hjepa"))
from collect import PHASES, STEPS, append, create  # noqa: E402
from env import MAX_ACTION, SO101Env  # noqa: E402
from expert import GRASP_POINT  # noqa: E402
from planner import Planner  # noqa: E402

GOAL_EVERY = 50  # steps (10 s) per goal


def yaw(q):
    """Yaw of a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def goal_obs(env, joints, cube_pos, cube_quat):
    """Render the scene with the arm at `joints` and the cube at the goal pose, then restore it."""
    saved = env.get_state()
    env.data.qpos[env.qadr] = joints
    env.data.qvel[:] = 0.0
    env.set_cube(cube_pos[:2], yaw(cube_quat), cube_pos[2])
    mujoco.mj_forward(env.model, env.data)
    obs = env.observe()
    env.set_state(saved)
    return obs


def main(ckpt, eval_config, goals, out, first_seed, n):
    first_seed, n = int(first_seed), int(n)
    with h5py.File(goals, "r") as g:
        g_joints, g_pos, g_quat = g["proprio"][:], g["cube_pos"][:], g["cube_quat"][:]
    planner, env = Planner(ckpt, eval_config), SO101Env(randomize=True)
    t0 = time.perf_counter()
    with h5py.File(out, "w") as f:
        for i in range(n):
            seed = first_seed + i
            rng = np.random.default_rng(seed + 555)
            obs = env.reset(seed)
            planner.seed(seed)
            cols = {k: [] for k in ["pixels", "proprio", "action", "ee", "cube_pos", "cube_quat", "phase"]}
            buffer = []
            for step in range(STEPS):
                if step % GOAL_EVERY == 0:
                    k = int(rng.integers(len(g_joints)))
                    goal, buffer = goal_obs(env, g_joints[k], g_pos[k], g_quat[k]), []
                    print(f"episode {i + 1}/{n} step {step} ({time.perf_counter() - t0:.0f} s)", flush=True)
                if not buffer:
                    buffer = list(planner.plan(obs, goal, steps_taken=step % GOAL_EVERY, eval_budget=GOAL_EVERY))
                action = np.clip(np.asarray(buffer.pop(0), np.float32), -MAX_ACTION, MAX_ACTION)  # what env.step runs
                pos, rot = env.site_pose()
                cube = env.cube_pose()
                cols["pixels"].append(obs["pixels"])
                cols["proprio"].append(obs["proprio"])
                cols["action"].append(action)
                cols["ee"].append((pos + rot @ GRASP_POINT).astype(np.float32))
                cols["cube_pos"].append(cube[:3].astype(np.float32))
                cols["cube_quat"].append(cube[3:].astype(np.float32))
                cols["phase"].append(PHASES.index("free"))
                obs = env.step(action)
            ep = {k: np.asarray(v) for k, v in cols.items()}
            if i == 0:
                create(f, ep)
            append(f, ep)
            print(f"FLEET_PROGRESS {i + 1}/{n} ({(i + 1) * STEPS / (time.perf_counter() - t0):.1f} steps/s)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:7])

"""Closed-loop trials in the sim: the planner drives the SO-101 to a goal observation.

    uv run python sim/closed_loop.py TASK CKPT EVAL_CONFIG FIRST_SEED N OUT.jsonl

TASK reach (rung 1): the goal is the arm at a random reachable pose (the cube stays where it is).
    Success: the grasp point ends within 1 cm of the goal's. Budget 100 steps (20 s).
TASK pick (rung 2): the goal is the cube at a new place (6 cm or more away) and the arm at rest;
    no sub-goals. Success: the cube ends within 2 cm (xy) of the goal place, resting on the table.
    Budget 300 steps (60 s).
Seeds 1000-4999 are for tuning, 5000 and higher for the test (PLAN.md, rule 6).
One JSON line per trial.
"""

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hjepa"))
from env import REST, SO101Env, sample_cube_xy  # noqa: E402
from expert import CLOSED, DOWN, GRASP_POINT, OPEN, ik, jaw_dir_for  # noqa: E402
from planner import Planner  # noqa: E402
from scene import CUBE_HALF  # noqa: E402

BUDGET = {"reach": 100, "pick": 300}


def grasp_point(env):
    pos, rot = env.site_pose()
    return pos + rot @ GRASP_POINT


def goal_observation(env, joints, cube_xy=None, cube_yaw=None):
    """Render the goal state (arm at `joints`, cube moved if given), then restore the state."""
    saved = env.get_state()
    env.data.qpos[env.qadr] = joints
    env.data.qvel[:] = 0.0
    if cube_xy is not None:
        env.set_cube(cube_xy, cube_yaw)
    mujoco.mj_forward(env.model, env.data)
    goal = dict(obs=env.observe(), grasp_point=grasp_point(env), cube=env.cube_pose()[:3])
    env.set_state(saved)
    return goal


def make_trial(env, task, seed):
    rng = np.random.default_rng(seed + 777)
    env.reset(seed)
    if task == "reach":
        xy, z = sample_cube_xy(rng), rng.uniform(0.04, 0.20)
        q, _ = ik(env, [*xy, z], DOWN, jaw_dir_for(rng.uniform(-1.0, 1.0)), REST)
        q[5] = rng.uniform(CLOSED, OPEN)
        return goal_observation(env, q)
    cube = env.cube_pose()[:2]
    target = sample_cube_xy(rng)
    while np.linalg.norm(target - cube) < 0.06:
        target = sample_cube_xy(rng)
    return goal_observation(env, REST, target, rng.uniform(-np.pi / 4, np.pi / 4))


def run(task, planner, env, seed):
    goal = make_trial(env, task, seed)
    planner.seed(seed)
    obs, buffer, budget = env.observe(), [], BUDGET[task]
    max_cube_z, t0 = 0.0, time.perf_counter()
    for step in range(budget):
        if not buffer:
            buffer = list(planner.plan(obs, goal["obs"], steps_taken=step, eval_budget=budget))
        obs = env.step(buffer.pop(0))
        max_cube_z = max(max_cube_z, float(env.cube_pose()[2]))
    result = dict(task=task, seed=seed, seconds=round(time.perf_counter() - t0, 1))
    if task == "reach":
        err = float(np.linalg.norm(grasp_point(env) - goal["grasp_point"]))
        result.update(error_cm=round(err * 100, 2), success=err < 0.01)
    else:
        cube = env.cube_pose()[:3]
        err = float(np.linalg.norm(cube[:2] - goal["cube"][:2]))
        resting = cube[2] < CUBE_HALF + 0.003
        result.update(error_cm=round(err * 100, 2), resting=bool(resting), success=bool(err < 0.02 and resting),
                      lifted=max_cube_z > CUBE_HALF + 0.02)
    return result


def main(task, ckpt, eval_config, first_seed, n, out):
    planner, env = Planner(ckpt, eval_config), SO101Env(randomize=True)
    with open(out, "a") as fh:
        for i in range(int(n)):
            r = run(task, planner, env, int(first_seed) + i)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"FLEET_PROGRESS {i + 1}/{n} {r}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:7])

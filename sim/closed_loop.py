"""Closed-loop trials in the sim: the planner drives the SO-101 to a goal observation.

    sh scripts/uvr python sim/closed_loop.py TASK CKPT EVAL_CONFIG FIRST_SEED N OUT.jsonl

TASK reach (rung 1): the goal is the arm at a random reachable pose (the cube stays where it is).
    Success: the grasp point ends within 1 cm of the goal's. Budget 100 steps (20 s).
TASK pick (rung 2): the goal is the cube at a new place (6 cm or more away) and the arm at rest;
    no sub-goals. Success: the cube ends within 2 cm (xy) of the goal place, resting on the table.
    Budget 300 steps (60 s). Measured too, outside the fixed pass mark (rule 5): picked_up (the cube
    was off the table, held by the arm), released (no arm contact at the end), on_table, settled
    (the cube moved under 2 mm in the last second and is still), and success_strict (all of them).
Seeds 1000-4999 are for tuning, 5000 and higher for the test (PLAN.md, rule 6).
One JSON line per trial, with the identity of the run: run_id is a hash of the task, the bytes of
the checkpoint (and of the normalizer, train config and value head the planner reads) and the
resolved planner config. Seeds of this run_id already in OUT are skipped, so a stopped run continues
where it ended; if OUT holds rows of another run, the run stops: results of two runs never mix.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

try:
    import fcntl  # Linux: lock the shared output file while appending
except ImportError:
    fcntl = None
import mujoco
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hjepa"))
from env import MAX_ACTION, REST, SO101Env, sample_cube_xy  # noqa: E402
from expert import CLOSED, DOWN, GRASP_POINT, OPEN, ik, jaw_dir_for  # noqa: E402
from planner import Planner  # noqa: E402
from scene import CUBE_HALF  # noqa: E402
from value import sha256_file  # noqa: E402

BUDGET = {"reach": 100, "pick": 300}
SETTLE = 5  # steps (1 s) over which a placed cube must not move


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


def cube_contacts(env):
    """Whether the cube touches the table, and whether it touches the arm (any other geom)."""
    n, cube, table = env.data.ncon, env.model.geom("cube").id, env.model.geom("table").id
    g1, g2 = env.data.contact.geom1[:n], env.data.contact.geom2[:n]
    other = np.concatenate([g2[g1 == cube], g1[g2 == cube]])
    return bool((other == table).any()), bool((other != table).any())


def pick_outcome(env, goal, picked_up, track):
    """The pick result at the end of a trial. `track`: the cube position after each step."""
    cube = env.cube_pose()[:3]
    err = float(np.linalg.norm(cube[:2] - goal["cube"][:2]))
    resting = cube[2] < CUBE_HALF + 0.003
    on_table, touched = cube_contacts(env)
    last = np.asarray(track[-SETTLE:])
    speed = float(np.linalg.norm(env.data.qvel[env.model.joint("cube").dofadr[0]:][:3]))
    settled = len(last) == SETTLE and float(np.linalg.norm(np.ptp(last, 0))) < 0.002 and speed < 0.01
    success = bool(err < 0.02 and resting)  # the fixed pass mark (rule 5)
    return dict(error_cm=round(err * 100, 2), resting=bool(resting), success=success,
                lifted=bool(track) and bool(max(p[2] for p in track) > CUBE_HALF + 0.02),
                picked_up=bool(picked_up), released=not touched, on_table=on_table, settled=bool(settled),
                success_strict=bool(success and picked_up and not touched and on_table and settled))


def run(task, planner, env, seed):
    goal = make_trial(env, task, seed)
    planner.seed(seed)
    planner.reset()
    obs, buffer, budget = env.observe(), [], BUDGET[task]
    track, picked_up, replans, nonfinite, t0 = [], False, 0, False, time.perf_counter()
    for step in range(budget):
        if not buffer:
            try:
                buffer = list(planner.plan(obs, goal["obs"], steps_taken=step, eval_budget=budget))
            except FloatingPointError:  # no finite plan: the trial ends here and is scored as it stands
                nonfinite = True
                break
            replans += 1
        action = np.clip(np.asarray(buffer.pop(0), np.float32), -MAX_ACTION, MAX_ACTION)  # what env.step runs
        planner.record(obs, action)
        obs = env.step(action)
        if task == "pick":
            track.append(env.cube_pose()[:3])
            on_table, touched = cube_contacts(env)
            picked_up |= touched and not on_table and track[-1][2] > CUBE_HALF + 0.01  # held off the table
    result = dict(task=task, seed=seed, seconds=round(time.perf_counter() - t0, 1), replans=replans,
                  nonfinite_plan=nonfinite)
    if task == "reach":
        err = float(np.linalg.norm(grasp_point(env) - goal["grasp_point"]))
        result.update(error_cm=round(err * 100, 2), success=err < 0.01)
    else:
        result.update(pick_outcome(env, goal, picked_up, track))
    return result


def _sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def run_identity(task, ckpt, eval_config, overrides=()):
    """The fields that tell this run's rows from another's (see the module docstring)."""
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "hjepa/config/eval" / f"{eval_config}.yaml"),
                          OmegaConf.from_dotlist(list(overrides)))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg.pop("policy", None)  # the checkpoint's path: its bytes are hashed below
    files = {"ckpt": Path(ckpt)}
    for name in ("normalizer.pt", "config.yaml") + (("value.pt",) if cfg.get("cost") == "value" else ()):
        files[name] = Path(ckpt).parent / name
    hashes = {k: sha256_file(p) if p.exists() else None for k, p in files.items()}
    ident = dict(task=task, budget=BUDGET[task], planner_config=cfg, files=hashes)
    return dict(run_id=_sha256(ident)[:16], ckpt_sha256=hashes["ckpt"], planner_config_sha256=_sha256(cfg),
                eval_config=eval_config)


def done_seeds(out, run_id):
    """Seeds of this run already in OUT. Rows of another run (or with no run_id) stop the run."""
    seeds = set()
    for line in open(out) if Path(out).exists() else ():
        try:
            r = json.loads(line)
        except ValueError:  # a line cut short by a stop
            continue
        if r.get("run_id") != run_id:
            raise SystemExit(f"{out} holds a row of another run (run_id {r.get('run_id')}, seed {r.get('seed')}); "
                             f"this run is {run_id}. Use a new output file.")
        seeds.add(r["seed"])
    return seeds


def main(task, ckpt, eval_config, first_seed, n, out):
    ident = run_identity(task, ckpt, eval_config)
    done = done_seeds(out, ident["run_id"])
    planner, env = Planner(ckpt, eval_config), SO101Env(randomize=True)
    if Path(out).exists() and Path(out).stat().st_size and not Path(out).read_bytes().endswith(b"\n"):
        with open(out, "a") as fh:  # end a line cut short by a stop, so the next record starts a line
            fh.write("\n")
    with open(out, "a") as fh:
        for i in range(int(n)):
            if int(first_seed) + i in done:
                continue
            r = dict(run(task, planner, env, int(first_seed) + i), **ident)
            if fcntl:  # several processes append to one file (run_phase0.sh): one whole line at a time
                fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if fcntl:
                fcntl.flock(fh, fcntl.LOCK_UN)
            print(f"FLEET_PROGRESS {i + 1}/{n} {r}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:7])

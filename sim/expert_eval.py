"""Fast test of the scripted expert: episodes without rendering, then pick-and-place outcomes.

    expert_eval.py reach                         # IK error of the grasp pose by cube distance
    expert_eval.py run OUT.json N FIRST_SEED [key=value ...]   # Expert options, e.g. grasp_dz=0.002
"""

import json
import sys

import numpy as np

from collect import PHASES, STEPS
from env import SO101Env
from expert import DOWN, GRASP_DZ, GRASP_POINT, Expert, ik, jaw_dir_for
from scene import CUBE_HALF
from stats import pick_outcomes, summary


def reach_map():
    env = SO101Env(randomize=False)
    env.reset(0)
    for r in np.arange(0.12, 0.35, 0.02):
        errs = [ik(env, [r * np.cos(a), r * np.sin(a), CUBE_HALF + GRASP_DZ], DOWN, jaw_dir_for(yaw), env.joints())[1]
                for a in (0.0, 0.5, 1.0) for yaw in (0.0, 0.7)]
        print(f"r {r:.2f} m: IK error max {1000 * max(errs):.1f} mm", flush=True)


def run(out, n, first_seed, opts):
    env = SO101Env(randomize=True)
    env.render = lambda: np.zeros((1, 1, 3), np.uint8)  # the expert needs no pixels
    outcomes = []
    for seed in range(first_seed, first_seed + n):
        env.reset(seed)
        expert = Expert(env, np.random.default_rng(seed), **opts)
        phase, cube, ee = [], [], []
        for _ in range(STEPS):
            action = expert.act()
            pos, rot = env.site_pose()
            phase.append(PHASES.index(expert.phase))
            cube.append(env.cube_pose()[:3])
            ee.append(pos + rot @ GRASP_POINT)
            env.step(action)
        outcomes += pick_outcomes(np.array(phase), np.array(cube), np.array(ee))
        print(f"FLEET_PROGRESS {seed - first_seed + 1}/{n}", flush=True)
    with open(out, "w") as f:
        json.dump({"opts": opts, "seeds": [first_seed, n], "outcomes": outcomes}, f)
    print(json.dumps(summary(outcomes)))


if __name__ == "__main__":
    if sys.argv[1] == "reach":
        reach_map()
    else:
        opts = {k: float(v) for k, v in (a.split("=") for a in sys.argv[5:])}
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), opts)

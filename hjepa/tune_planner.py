"""Planner settings by search, not by hand (PLAN.md A22).

    sh scripts/uvr python hjepa/tune_planner.py CKPT EVAL_CONFIG TASK TRIALS SETTINGS OUT.json [FIRST]

Random search over the CEM settings: horizon, samples, kept samples, rounds and the replanning
interval, drawn from wide ranges (numpy seed 0; settings FIRST.. of the sequence, so several
processes can split the search, each with its own OUT.json). Each setting plays TRIALS closed-loop trials of TASK (reach or
pick) on the tuning seeds 1000.. (rule 6: never the test seeds 5000..), the same seeds for every
setting. Score: success rate, then mean error. OUT.json lists every setting, best first.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from closed_loop import run  # noqa: E402
from env import SO101Env  # noqa: E402
from planner import Planner  # noqa: E402

SPACE = {  # name: (override key, choices)
    "horizon": ("plan_config.horizon", [2, 3, 5, 8, 12, 16]),
    "receding": ("plan_config.receding_horizon", [1, 2, 4]),
    "samples": ("solver.num_samples", [64, 128, 300, 600, 1000, 2000]),
    "kept": ("solver.topk", [0.03, 0.1, 0.2]),  # share of the samples
    "rounds": ("solver.n_steps", [5, 10, 30, 60]),
}


def settings(n, first=0):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(first + n):
        s = {k: v[int(rng.integers(len(v)))] for k, (_, v) in SPACE.items()}
        s["receding"] = min(s["receding"], s["horizon"])
        s["kept"] = max(2, int(round(s["kept"] * s["samples"])))
        out.append(s)
    return out[first:]


def main(ckpt, eval_config, task, trials, n, out, first=0):
    trials, n, first = int(trials), int(n), int(first)
    env, report = SO101Env(randomize=True), []
    for s in settings(n, first):
        overrides = [f"{SPACE[k][0]}={v}" for k, v in s.items()]
        planner, t0 = Planner(ckpt, eval_config, overrides=overrides), time.time()
        results = [run(task, planner, env, 1000 + i) for i in range(trials)]
        row = dict(s, success=float(np.mean([r["success"] for r in results])),
                   error_cm=round(float(np.mean([r["error_cm"] for r in results])), 2),
                   seconds_per_trial=round((time.time() - t0) / trials, 1))
        report.append(row)
        print(row, flush=True)
        report.sort(key=lambda r: (-r["success"], r["error_cm"]))
        with open(out, "w") as fh:
            json.dump(dict(ckpt=ckpt, eval_config=eval_config, task=task, trials=trials, settings=report), fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:8])

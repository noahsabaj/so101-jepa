"""Planning speed: seconds per plan in fp32, bf16 and fp16 autocast, bf16 + torch.compile and bf16 +
CUDA graphs (compile mode reduce-overhead), and how far each plan moves from the fp32 plan (same seed,
same observations).

    sh scripts/uvr python hjepa/bench_plan.py CKPT EVAL_CONFIG PLANS OUT.json

Observations and goals: the first frames of reach trials on the tuning seeds 1000.. (rule 6). The
first 3 plans of each mode are warm-up (compilation) and are not timed.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from closed_loop import make_trial  # noqa: E402
from env import SO101Env  # noqa: E402
from planner import Planner  # noqa: E402

MODES = {
    "fp32": [], "bf16": ["precision=bf16"], "fp16": ["precision=fp16"],
    "bf16_compile": ["precision=bf16", "compile=true"], "fp16_compile": ["precision=fp16", "compile=true"],
    "bf16_graphs": ["precision=bf16", "compile=reduce-overhead"],
}
WARMUP = 3


def main(ckpt, eval_config, plans, out):
    plans, env = int(plans), SO101Env(randomize=True)
    cases = []
    for i in range(WARMUP + plans):
        goal = make_trial(env, "reach", 1000 + i)
        cases.append((env.observe(), goal["obs"]))
    report, ref = {"ckpt": ckpt, "eval_config": eval_config, "plans": plans, "gpu": torch.cuda.get_device_name()}, None
    for mode, overrides in MODES.items():
        planner, acts = Planner(ckpt, eval_config, overrides=overrides), []
        for i, (obs, goal) in enumerate(cases):
            if i == WARMUP:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            planner.seed(i)
            acts.append(planner.plan(obs, goal))
        torch.cuda.synchronize()
        row = {"seconds_per_plan": round((time.perf_counter() - t0) / plans, 4)}
        acts = np.stack(acts[WARMUP:])
        if ref is None:
            ref = acts
        else:
            row["mean_abs_diff_rad"] = round(float(np.abs(acts - ref).mean()), 5)
            row["fp32_mean_abs_rad"] = round(float(np.abs(ref).mean()), 5)
        report[mode] = row
        print(mode, row, flush=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])

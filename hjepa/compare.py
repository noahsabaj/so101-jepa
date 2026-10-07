"""Paired comparisons (rule 6) of results on the same test trials.

    sh scripts/uvr python hjepa/compare.py BASE OTHER...

Closed loop (*.jsonl of sim/closed_loop.py): for each file, the successes with a 95% Wilson
interval, the median error and the rung's pass mark. For each OTHER against BASE, on the seeds
that both ran: the success difference with a 95% bootstrap interval, McNemar's exact p (from the
seeds where only one succeeded) and the median paired error difference.
Offline (*.json of hjepa/offline.py): each test's mean and 95% interval, and for each OTHER the
paired mean difference against BASE over the same cases, with a 95% interval.
"""

import json
import sys
from math import comb, sqrt
from pathlib import Path

import numpy as np

PASS = {"reach": (90, 100), "pick": (30, 50)}  # rungs 1 and 2: successes needed of the test trials
RNG = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p + z * z / (2 * n)) / d - h, (p + z * z / (2 * n)) / d + h


def boot(d, reps=10000):
    m = d[RNG.integers(0, len(d), (reps, len(d)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def trials(path):
    rows = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:  # a blank line, or a line cut short by a stop
            continue
        rows[r["seed"]] = r  # a repeated seed keeps its last run
    return rows


def closed_loop(paths):
    runs = [trials(p) for p in paths]
    print(f"{'file':44} {'n':>4} {'success':>9} {'95% CI':>12} {'median err':>11}  pass mark")
    for p, r in zip(paths, runs):
        task = next(iter(r.values()))["task"]
        k, n = sum(x["success"] for x in r.values()), len(r)
        lo, hi = wilson(k, n)
        need, full = PASS[task]
        verdict = f"{need}/{full}: " + ("incomplete" if n < full else "PASS" if k >= need else "fail")
        med = np.median([x["error_cm"] for x in r.values()])
        print(f"{Path(p).name:44} {n:4} {k:4}/{n:<4} [{lo:4.0%},{hi:4.0%}] {med:8.2f} cm  {verdict}")
    base = runs[0]
    for p, r in zip(paths[1:], runs[1:]):
        if {x["task"] for x in r.values()} != {x["task"] for x in base.values()}:
            print(f"{Path(p).name}: not the task of {Path(paths[0]).name}, no paired test")
            continue
        seeds = sorted(set(base) & set(r))
        a = np.array([base[s]["success"] for s in seeds], float)
        b = np.array([r[s]["success"] for s in seeds], float)
        lo, hi = boot(b - a)
        derr = np.median([r[s]["error_cm"] - base[s]["error_cm"] for s in seeds])
        only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        print(f"{Path(p).name} - {Path(paths[0]).name}, {len(seeds)} seeds: success {(b - a).mean():+.0%}"
              f" [{lo:+.0%}, {hi:+.0%}], McNemar p = {mcnemar(only_a, only_b):.3f} (only base {only_a},"
              f" only other {only_b}); median error difference {derr:+.2f} cm")


def offline(paths):
    reps = [json.load(open(p)) for p in paths]
    tests = [k for k in reps[0] if k.startswith(("expert_rank_k", "plan_cosine_k")) and not k.endswith("_cases")]
    names = [Path(p).stem.removeprefix("offline_") for p in paths]
    print(f"{'test':16} " + " ".join(f"{n:>24}" for n in names))
    for t in tests:
        print(f"{t:16} " + " ".join(f"{r[t][0]:>15.3f} +- {r[t][1]:.3f}" for r in reps))
    for n, r in zip(names[1:], reps[1:]):
        for t in tests:
            if f"{t}_cases" not in r or f"{t}_cases" not in reps[0]:
                print(f"{n} - {names[0]} {t}: no per-case values (unpaired only)")
                continue
            d = np.array(r[f"{t}_cases"]) - np.array(reps[0][f"{t}_cases"])
            h = 1.96 * d.std(ddof=1) / sqrt(len(d))
            print(f"{n} - {names[0]} {t}: {d.mean():+.3f} [{d.mean() - h:+.3f}, {d.mean() + h:+.3f}] ({len(d)} cases)")


if __name__ == "__main__":
    files = sys.argv[1:]
    (closed_loop if files[0].endswith(".jsonl") else offline)(files)

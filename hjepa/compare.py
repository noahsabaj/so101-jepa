"""Paired comparisons (rule 6) of results on the same test trials.

    sh scripts/uvr python hjepa/compare.py [--unpaired] BASE OTHER...

Closed loop (*.jsonl of sim/closed_loop.py): for each file, the successes with a 95% Wilson
interval, the median error, the rung's pass mark and (pick) the success_strict count. The verdict
is given only on exactly the test seeds of the rung (5000 to 5000 + trials - 1, rule 6) and one
task; any other set of seeds is reported with no verdict. For each OTHER against BASE, on the seeds
that both ran: the success difference with a 95% bootstrap interval, McNemar's exact p (from the
seeds where only one succeeded) and the median paired error difference.
Offline (*.json of hjepa/offline.py or hjepa/jump_test.py): each test's mean and 95% interval, and
for each OTHER the paired mean difference against BASE, with a 95% interval, over the cases joined
on their (episode, frame) ids. Reports without case ids or data fingerprints, or of different data,
are not paired: the comparison stops, unless --unpaired asks for the unpaired difference of means.
"""

import json
import sys
from math import comb, sqrt
from pathlib import Path

import numpy as np

PASS = {"reach": (90, 100), "pick": (30, 50)}  # rungs 1 and 2: successes needed of the test trials
TEST_FIRST = 5000  # the test seeds of a rung: TEST_FIRST .. TEST_FIRST + trials - 1 (rule 6)
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
    runs = {r.get("run_id") for r in rows.values()}
    if len(runs) > 1:
        raise SystemExit(f"{path} mixes rows of {len(runs)} runs ({sorted(map(str, runs))}): no result")
    return rows


def verdict(rows):
    """The rung verdict, given only on exactly the rung's test seeds of one task."""
    tasks = {x["task"] for x in rows.values()}
    if len(tasks) != 1 or next(iter(tasks)) not in PASS:
        return f"no verdict: tasks {sorted(tasks)}"
    task = tasks.pop()
    need, full = PASS[task]
    test, seeds = set(range(TEST_FIRST, TEST_FIRST + full)), set(rows)
    if not seeds <= test:
        return f"{need}/{full}: no verdict, {len(seeds - test)} seeds outside the test seeds {TEST_FIRST}-{TEST_FIRST + full - 1}"
    if seeds != test:
        return f"{need}/{full}: incomplete ({len(seeds)} of {full} test seeds)"
    key = "success_strict" if task == "pick" else "success"  # rung 2: a real pick (Noah, 2026-10-08)
    if any(key not in rows[s] for s in test):
        return f"{need}/{full}: no verdict, rows without {key} (run before 2026-10-08)"
    return f"{need}/{full}: " + ("PASS" if sum(rows[s][key] for s in test) >= need else "fail")


def closed_loop(paths):
    runs = [trials(p) for p in paths]
    print(f"{'file':44} {'n':>4} {'success':>9} {'95% CI':>12} {'median err':>11}  pass mark  [strict]")
    for p, r in zip(paths, runs):
        k, n = sum(x["success"] for x in r.values()), len(r)
        lo, hi = wilson(k, n)
        med = np.median([x["error_cm"] for x in r.values()])
        strict = ""
        if all("success_strict" in x for x in r.values()):
            strict = f"  [strict {sum(x['success_strict'] for x in r.values())}/{n}]"
        print(f"{Path(p).name:44} {n:4} {k:4}/{n:<4} [{lo:4.0%},{hi:4.0%}] {med:8.2f} cm  {verdict(r)}{strict}")
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


def paired(base, other, t):
    """The per-case values of test t in both reports on the same cases (joined on case_ids).
    Raises ValueError when the reports cannot be paired."""
    for r in (base, other):
        if "case_ids" not in r or "data_fingerprint" not in r or f"{t}_cases" not in r:
            raise ValueError("no case_ids, data_fingerprint or per-case values")
    if base["data_fingerprint"] != other["data_fingerprint"]:
        raise ValueError("different data (data_fingerprint)")
    ids = []
    for r in (base, other):  # a test may use the first cases only (offline.py: plan_cosine)
        vals = r[f"{t}_cases"]
        if len(vals) > len(r["case_ids"]):
            raise ValueError("more values than case_ids")
        ids.append([tuple(c) for c in r["case_ids"][:len(vals)]])
    a, b = base[f"{t}_cases"], other[f"{t}_cases"]
    if ids[0] == ids[1]:
        return np.asarray(a, float), np.asarray(b, float)
    if len(set(ids[0])) < len(ids[0]) or len(set(ids[1])) < len(ids[1]):
        raise ValueError("different case lists with repeated cases: no join")
    da, db = dict(zip(ids[0], a)), dict(zip(ids[1], b))
    common = [c for c in ids[0] if c in db]
    if not common:
        raise ValueError("no common cases")
    return np.array([da[c] for c in common], float), np.array([db[c] for c in common], float)


def summary(r, t):
    """[mean, 95% half-width] of test t."""
    if isinstance(r.get(t), list):
        return r[t]
    x = np.asarray(r[f"{t}_cases"], float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / sqrt(len(x)))]


def offline(paths, unpaired=False):
    reps = [json.load(open(p)) for p in paths]
    tests = [k for k in reps[0] if k.startswith(("expert_rank_k", "plan_cosine_k")) and not k.endswith("_cases")]
    tests += [k[:-6] for k in reps[0] if k.endswith("_cases") and k[:-6] not in tests]
    names = [Path(p).stem.removeprefix("offline_") for p in paths]
    print(f"{'test':16} " + " ".join(f"{n:>24}" for n in names))
    for t in tests:
        print(f"{t:16} " + " ".join("{:>15.3f} +- {:.3f}".format(*summary(r, t)) for r in reps))
    refused = False
    for n, r in zip(names[1:], reps[1:]):
        for t in tests:
            if unpaired:
                (m0, h0), (m1, h1) = summary(reps[0], t), summary(r, t)
                h = sqrt(h0 ** 2 + h1 ** 2)
                print(f"{n} - {names[0]} {t}: {m1 - m0:+.3f} [{m1 - m0 - h:+.3f}, {m1 - m0 + h:+.3f}] (unpaired)")
                continue
            try:
                a, b = paired(reps[0], r, t)
            except ValueError as err:
                print(f"{n} - {names[0]} {t}: no paired test: {err} (--unpaired for an unpaired difference)")
                refused = True
                continue
            d = b - a
            h = 1.96 * d.std(ddof=1) / sqrt(len(d))
            print(f"{n} - {names[0]} {t}: {d.mean():+.3f} [{d.mean() - h:+.3f}, {d.mean() + h:+.3f}] ({len(d)} cases)")
    if refused:
        sys.exit(1)


if __name__ == "__main__":
    unpaired = "--unpaired" in sys.argv[1:]
    files = [a for a in sys.argv[1:] if a != "--unpaired"]
    if files[0].endswith(".jsonl"):
        closed_loop(files)
    else:
        offline(files, unpaired)

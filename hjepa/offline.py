"""Offline tests of a trained world model on held-out episodes (no environment).

    sh scripts/uvr python hjepa/offline.py CKPT EVAL_CONFIG DATA.h5 OUT.json [cases]

1. Expert rank (k = 1, 5, 15 steps ahead; 0.2, 1, 3 s): the goal is the latent of the frame k
   steps later. Level 1 predicts the next latent for the expert's move and for 100 random moves
   (moves of other held-out steps). The rank is the share of random moves whose prediction is
   closer to the goal than the expert's (0 is perfect, 0.5 is chance). "Closer" is the planner's
   cost: the latent distance, or minus the learned value (cost: value).
2. Plan direction (k = 5, 15): the goal is frame t + k; the planner plans its configured horizon
   (5 steps, 1 s; also for k = 15) toward it. The score is the cosine between its first move and the
   expert's move (normalized units; 0 is chance). As in closed_loop.py, the planner starts each case
   from a reset with the case's preceding real frames and actions recorded (up to its history).
A non-finite cost counts against the model: an expert's makes its rank 1, a random move's counts as
closer ("*_nonfinite" counts them). The report keeps the identity of the cases: "case_ids" ([episode,
frame], in the order of every "*_cases" array; plan_cosine uses the first cases only) and
"data_fingerprint" (size and sha256 of the first and last MB of DATA.h5).
"""

import json
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from planner import Planner, StrideEmbed
from data import file_fingerprint  # noqa: E402  (H-JEPA's data.py, on the path from planner)

RANKS_K, PLAN_K, DECOYS = (1, 5, 15), (5, 15), 100


def load(path, rng, cases, max_k):
    """Random (episode, t) cases with 3 steps of history and max_k steps ahead: (offset, t) picks and
    [episode, t] ids."""
    f = h5py.File(path, "r")
    lens, offs = f["ep_len"][:], f["ep_offset"][:]
    eps = np.flatnonzero(lens > max_k + 4)
    picks, ids = [], []
    for _ in range(cases):
        e = rng.choice(eps)
        picks.append((offs[e], int(rng.integers(3, lens[e] - max_k))))
        ids.append([int(e), picks[-1][1]])
    return f, picks, ids


def mean_ci(x):
    x = np.asarray(x, float)
    return [round(float(x.mean()), 4), round(float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), 4)]


@torch.no_grad()
def expert_rank(planner, f, picks, k, decoy_actions):
    level1 = planner.model.get_level(1) if hasattr(planner.model, "get_level") else planner.model
    embed = StrideEmbed(1, planner.kmax) if planner.kmax > 1 else level1.action_embed  # a time-step model: stride 1
    ranks, nonfinite = [], 0
    for off, t in picks:
        rows = slice(off + t - 3, off + t + 1)
        obs = [dict(pixels=p, proprio=q) for p, q in zip(f["pixels"][rows], f["proprio"][rows])]
        hist = torch.stack([planner.encode(o) for o in obs])[None]  # (1, 4, D)
        goal = planner.encode(dict(pixels=f["pixels"][off + t + k], proprio=f["proprio"][off + t + k]))
        acts = planner.normalize_action(f["action"][off + t - 3:off + t + 1])  # (4, 6), last = expert
        cands = np.concatenate([acts[-1:], decoy_actions[np.random.default_rng(t).choice(len(decoy_actions), DECOYS)]])
        act = np.repeat(acts[None], len(cands), axis=0)
        act[:, -1] = cands
        act = torch.as_tensor(act, dtype=torch.float32, device="cuda")
        pred = level1.predict(hist.expand(len(cands), *hist.shape[1:]), embed(act))[:, -1]
        if getattr(level1, "value_fn", None) is not None:  # the planner's learned cost (cost: value)
            cost = -level1.value_fn(pred, goal.expand_as(pred)).cpu().numpy()
        else:
            cost = ((pred - goal) ** 2).flatten(1).mean(1).cpu().numpy()
        ranks.append(rank_of_expert(cost))
        nonfinite += int(not np.isfinite(cost).all())
    return ranks, nonfinite


def rank_of_expert(cost):
    """Share of random moves (cost[1:]) closer than the expert's (cost[0]); a non-finite cost counts
    against the model (NaN compares false, which would make the rank look perfect)."""
    bad = ~np.isfinite(cost)
    return 1.0 if bad[0] else float(((cost[1:] < cost[0]) | bad[1:]).mean())


def record_history(planner, f, off, t):
    """Reset the planner and record the case's preceding real frames and actions (up to its history)."""
    planner.reset()
    for j in range(max(0, t - (planner.past.maxlen or 0)), t):
        planner.record(dict(pixels=f["pixels"][off + j], proprio=f["proprio"][off + j]), f["action"][off + j])


def plan_direction(planner, f, picks, k):
    cos = []
    for i, (off, t) in enumerate(picks):
        planner.seed(i)
        record_history(planner, f, off, t)
        obs = dict(pixels=f["pixels"][off + t], proprio=f["proprio"][off + t])
        goal = dict(pixels=f["pixels"][off + t + k], proprio=f["proprio"][off + t + k])
        a = planner.normalize_action(planner.plan(obs, goal, eval_budget=k)[0])
        e = planner.normalize_action(f["action"][off + t])
        cos.append(float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-8)))
    return cos


def main(ckpt, eval_config, data, out, cases=500):
    rng = np.random.default_rng(0)
    planner = Planner(ckpt, eval_config)
    f, picks, ids = load(data, rng, cases, max(RANKS_K))
    decoys = planner.normalize_action(f["action"][: min(len(f["action"]), 200000): 7])
    report = {"ckpt": ckpt, "planner": eval_config, "data": data, "cases": cases,
              "case_ids": ids, "data_fingerprint": file_fingerprint(data)}
    # The cases are the same for each model (fixed seeds): "_cases" keeps the values for paired tests.
    for k in RANKS_K:
        x, report[f"expert_rank_k{k}_nonfinite"] = expert_rank(planner, f, picks, k, decoys)
        report[f"expert_rank_k{k}"], report[f"expert_rank_k{k}_cases"] = mean_ci(x), [round(v, 4) for v in x]
        print(k, report[f"expert_rank_k{k}"], flush=True)
    for k in PLAN_K:
        x = plan_direction(planner, f, picks[: cases // 2], k)
        report[f"plan_cosine_k{k}"], report[f"plan_cosine_k{k}_cases"] = mean_ci(x), [round(v, 4) for v in x]
        print(k, report[f"plan_cosine_k{k}"], flush=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5], *(int(a) for a in sys.argv[5:6]))

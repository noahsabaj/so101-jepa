"""Offline tests of a trained world model on held-out episodes (no environment).

    uv run python hjepa/offline.py CKPT EVAL_CONFIG DATA.h5 OUT.json [cases]

1. Expert rank (k = 1, 5, 15 steps ahead; 0.2, 1, 3 s): the goal is the latent of the frame k
   steps later. Level 1 predicts the next latent for the expert's move and for 100 random moves
   (moves of other held-out steps). The rank is the share of random moves whose prediction is
   closer to the goal than the expert's (0 is perfect, 0.5 is chance).
2. Plan direction (k = 5, 15): the planner plans from frame t to frame t + k. The score is the
   cosine between its first move and the expert's move (normalized units; 0 is chance).
"""

import json
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from planner import Planner

RANKS_K, PLAN_K, DECOYS = (1, 5, 15), (5, 15), 100


def load(path, rng, cases, max_k):
    """Random (episode, t) cases with 3 steps of history and max_k steps ahead."""
    f = h5py.File(path, "r")
    lens, offs = f["ep_len"][:], f["ep_offset"][:]
    eps = np.flatnonzero(lens > max_k + 4)
    picks = []
    for _ in range(cases):
        e = rng.choice(eps)
        picks.append((offs[e], int(rng.integers(3, lens[e] - max_k))))
    return f, picks


def mean_ci(x):
    x = np.asarray(x, float)
    return [round(float(x.mean()), 4), round(float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), 4)]


@torch.no_grad()
def expert_rank(planner, f, picks, k, decoy_actions):
    level1 = planner.model.get_level(1) if hasattr(planner.model, "get_level") else planner.model
    ranks = []
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
        pred = level1.predict(hist.expand(len(cands), -1, -1), level1.action_embed(act))[:, -1]
        cost = ((pred - goal) ** 2).mean(-1).cpu().numpy()
        ranks.append(float((cost[1:] < cost[0]).mean()))
    return ranks


def plan_direction(planner, f, picks, k):
    cos = []
    for i, (off, t) in enumerate(picks):
        planner.seed(i)
        obs = dict(pixels=f["pixels"][off + t], proprio=f["proprio"][off + t])
        goal = dict(pixels=f["pixels"][off + t + k], proprio=f["proprio"][off + t + k])
        a = planner.normalize_action(planner.plan(obs, goal, eval_budget=k)[0])
        e = planner.normalize_action(f["action"][off + t])
        cos.append(float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-8)))
    return cos


def main(ckpt, eval_config, data, out, cases=500):
    rng = np.random.default_rng(0)
    planner = Planner(ckpt, eval_config)
    f, picks = load(data, rng, cases, max(RANKS_K))
    decoys = planner.normalize_action(f["action"][: min(len(f["action"]), 200000): 7])
    report = {"ckpt": ckpt, "planner": eval_config, "data": data, "cases": cases}
    for k in RANKS_K:
        report[f"expert_rank_k{k}"] = mean_ci(expert_rank(planner, f, picks, k, decoys))
        print(k, report[f"expert_rank_k{k}"], flush=True)
    for k in PLAN_K:
        report[f"plan_cosine_k{k}"] = mean_ci(plan_direction(planner, f, picks[: cases // 2], k))
        print(k, report[f"plan_cosine_k{k}"], flush=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5], *(int(a) for a in sys.argv[5:6]))

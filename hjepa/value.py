"""A learned goal-reaching value for the planner, in place of the hand-set latent distance.

    sh scripts/uvr python hjepa/value.py CKPT EVAL_CONFIG TRAIN.h5 VAL.h5 [steps]

V(z, z_goal) estimates minus the discounted number of 0.2 s steps from state z to goal state z_goal
(latents of the frozen world model's encoder). It learns from the training episodes with no task
labels: a goal is the same frame, a later frame of the episode (geometric offset) or a random frame
(hindsight relabelling); the reward is -1 per step until the goal frame, and a step into the goal
frame ends the episode (no bootstrap from V(goal, goal), which last frames never learn). The fit is expectile
regression of two value heads against a slow target (the action-free value of HIQL and GCIQL, Park
et al. 2023; OGBench settings: discount 0.99, expectile 0.7, goals 0.2 / 0.5 / 0.3).

Writes value.pt (with encoder_ckpt_sha256, the sha256 of the checkpoint file it learned on) and
value.json beside the checkpoint, each through a .tmp file and a rename. With `cost: value` in the
eval config, the planner's cost is -V(last predicted latent, goal latent). Val metrics: the rank
correlation of V with the true steps to a later goal (average ranks for ties; null when V is
constant), and how often a state 5 steps nearer to the goal has the larger value.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata

from planner import MEAN, STD, Planner

GAMMA, EXPECTILE, TAU = 0.99, 0.7, 0.005
P_CUR, P_TRAJ = 0.2, 0.5  # goal: this frame, a later frame of the episode; else a random frame


class _MLPHead(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        layers, d = [], 2 * dim
        for _ in range(3):
            layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.GELU()]
            d = width
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, z, g):
        return self.net(torch.cat([z, g], -1)).squeeze(-1)


class _TokenHead(nn.Module):
    """A small transformer over the state tokens, the goal tokens and a value token."""

    def __init__(self, dim, num_tokens, width, depth=2, heads=4):
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.pos = nn.Parameter(0.02 * torch.randn(1, 2 * num_tokens + 1, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, depth)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, z, g):
        x = torch.cat([self.cls.expand(len(z), -1, -1), self.inp(z), self.inp(g)], 1) + self.pos
        return self.out(self.body(x)[:, 0]).squeeze(-1)


class GoalValue(nn.Module):
    """Two value heads on (z, z_goal): (B, D) vector latents or (B, N, D) token latents."""

    def __init__(self, dim, num_tokens=1, width=None):
        super().__init__()
        self.cfg = dict(dim=int(dim), num_tokens=int(num_tokens), width=width)
        make = (lambda: _MLPHead(dim, width or 512)) if num_tokens == 1 else (lambda: _TokenHead(dim, num_tokens, width or 128))
        self.heads = nn.ModuleList([make(), make()])

    def both(self, z, g):
        return [h(z.float(), g.float()) for h in self.heads]

    def forward(self, z, g):
        """The planning value: the mean of the two heads."""
        v1, v2 = self.both(z, g)
        return (v1 + v2) / 2


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def publish(path, write):
    """write(tmp) to `path`.tmp, then rename it to `path`: a reader never sees a partial file."""
    tmp = Path(f"{path}.tmp")
    write(tmp)
    os.replace(tmp, path)


def load_value(path, device="cuda"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    value = GoalValue(**blob["cfg"])
    value.load_state_dict(blob["state_dict"])
    return value.to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode_all(planner, path, batch=256):
    """Encoder latents (fp16, on the CPU) of every frame, and each frame's episode bounds."""
    level1 = planner.model.get_level(1) if hasattr(planner.model, "get_level") else planner.model
    f = h5py.File(path, "r")
    out = None  # filled in place: no list of batches and no second copy from concatenating them
    for i in range(0, len(f["pixels"]), batch):
        px = torch.from_numpy(f["pixels"][i:i + batch]).permute(0, 3, 1, 2).float() / 255.0
        pr = (torch.as_tensor(f["proprio"][i:i + batch]).float() - planner.proprio_mean) / planner.proprio_std
        o = level1.encode({"pixels": ((px - MEAN) / STD)[:, None].cuda(), "proprio": pr[:, None].cuda()})
        z = o["embed_0"][:, 0].half().cpu()
        if out is None:
            out = torch.empty((len(f["pixels"]), *z.shape[1:]), dtype=torch.float16)
        out[i:i + len(z)] = z
    off, n = f["ep_offset"][:].astype(np.int64), f["ep_len"][:].astype(np.int64)
    ep = np.repeat(np.arange(len(n)), n)
    return out, off[ep] + n[ep] - 1  # latents, last frame of each frame's episode


def sample(rng, z, last, starts, batch, device="cuda"):
    """(state, next state, goal) rows and the reward and continuation of each. starts: the frames
    that have a next frame in their episode. A step into the goal frame ends the episode: its
    target is -1, not -1 + GAMMA * V(goal, goal), which is never trained for last frames."""
    idx = rng.choice(starts, batch)
    u = rng.random(batch)
    later = np.minimum(idx + rng.geometric(1 - GAMMA, batch), last[idx])
    goal = np.where(u < P_CUR, idx, np.where(u < P_CUR + P_TRAJ, later, rng.integers(0, len(last), batch)))
    hit = torch.as_tensor(goal == idx, dtype=torch.float32, device=device)
    ends = torch.as_tensor((goal == idx) | (goal == idx + 1), dtype=torch.float32, device=device)
    rows = lambda r: z[torch.as_tensor(r)].to(device, non_blocking=True)  # noqa: E731
    return rows(idx), rows(idx + 1), rows(goal), hit - 1.0, 1.0 - ends


def expectile_loss(adv, diff):
    return (torch.abs(EXPECTILE - (adv < 0).float()) * diff ** 2).mean()


@torch.no_grad()
def evaluate(value, z, last, rng, n=4000):
    """Spearman correlation of -V with the true steps to a later goal of the same episode (1 to 100
    steps), and the share of pairs where the state 5 steps nearer to the goal has the larger value."""
    idx = rng.choice(np.flatnonzero(last - np.arange(len(last)) >= 10), n)
    k = np.minimum(rng.integers(6, 101, n), last[idx] - idx)
    g = z[torch.as_tensor(idx + k)].cuda()
    v_far = value(z[torch.as_tensor(idx)].cuda(), g).cpu().numpy()
    v_near = value(z[torch.as_tensor(idx + 5)].cuda(), g).cpu().numpy()
    rho = spearman(-v_far, k)
    return dict(spearman_steps=None if rho is None else round(rho, 3),
                nearer_has_larger_value=round(float((v_near > v_far).mean()), 3))


def spearman(x, y):
    """Spearman correlation with average ranks for ties; None (undefined) when x or y is constant."""
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def main(ckpt, eval_config, train, val, steps=50000):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ckpt_sha = sha256_file(ckpt)  # the encoder the value learns on: planner.py checks it at load
    planner = Planner(ckpt, eval_config)
    t0 = time.time()
    z, last = encode_all(planner, train)
    zv, lastv = encode_all(planner, val)
    print(f"encoded {len(z)} train and {len(zv)} val frames, latent {tuple(z.shape[1:])}, {time.time() - t0:.0f} s", flush=True)
    tokens = z.ndim == 3
    value = GoalValue(z.shape[-1], z.shape[1] if tokens else 1).cuda()
    target = GoalValue(**value.cfg).cuda().requires_grad_(False)
    target.load_state_dict(value.state_dict())
    opt = torch.optim.AdamW(value.parameters(), lr=3e-4)
    batch, report = (256 if tokens else 1024), {"ckpt": ckpt, "encoder_ckpt_sha256": ckpt_sha, "steps": steps, "log": []}
    starts = np.flatnonzero(np.arange(len(last)) < last)
    for step in range(1, steps + 1):
        s, s2, g, r, cont = sample(rng, z, last, starts, batch)
        with torch.no_grad():
            n1, n2 = target.both(s2, g)
            t1, t2 = target.both(s, g)
            adv = r + GAMMA * cont * torch.minimum(n1, n2) - (t1 + t2) / 2
        v1, v2 = value.both(s, g)
        loss = expectile_loss(adv, r + GAMMA * cont * n1 - v1) + expectile_loss(adv, r + GAMMA * cont * n2 - v2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, q in zip(target.parameters(), value.parameters()):
                p.lerp_(q, TAU)
        if step % 5000 == 0 or step == steps:
            m = dict(step=step, loss=round(float(loss), 4), **evaluate(value.eval(), zv, lastv, np.random.default_rng(1)))
            value.train()
            report["log"].append(m)
            print(f"FLEET_PROGRESS {step}/{steps}", m, flush=True)
    report.update(report["log"][-1])
    save(Path(ckpt).parent, value, report)
    print(json.dumps(report, indent=1))


def save(out, value, report):
    """value.json, then value.pt (the file the planner and job waiters look for), each published whole."""
    def write_json(p):
        with open(p, "w") as fh:
            json.dump(report, fh, indent=1)
    publish(out / "value.json", write_json)
    publish(out / "value.pt", lambda p: torch.save({"cfg": value.cfg, "state_dict": value.state_dict(), "report": report,
                                                    "encoder_ckpt_sha256": report["encoder_ckpt_sha256"]}, p))


if __name__ == "__main__":
    main(*sys.argv[1:5], *(int(a) for a in sys.argv[5:6]))

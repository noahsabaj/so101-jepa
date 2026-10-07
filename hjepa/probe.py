"""Linear probes: does the level-1 latent show the cube and the grasp point? (held-out episodes)

    sh scripts/uvr python hjepa/probe.py CKPT EVAL_CONFIG DATA.h5 OUT.json

Ridge regression from a feature set to a target. Fit on the first 80% of the episodes (the ridge
strength is selected on the last quarter of those), score on the other 20%: RMS error (cm) and R^2.
Features: the pixel latent (vision only), the full latent (what the planner compares), the joint
angles (no vision: a baseline) and none (the mean: the spread of the target). Targets: the cube,
the grasp point, the cube relative to the grasp point, and the cube on the frames where it rests on
the table (there, only vision can show it). Also: the share of each latent's variance that is
within episodes (near 0: the latent is mostly constant per episode, e.g. light and colours).
Token latents (SO-JEPA v6 and later) also get the attentive probe of hjepa/levjepa_probe.py (one
learned query attends over the patch tokens; every 3rd frame), the fair readout for tokens: where
the cube is lies in which token, which a linear map of all tokens reads badly.
"""

import json
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

from planner import MEAN, STD, Planner

CUBE_HALF = 0.0125  # sim/scene.py
LAMBDAS = 10.0 ** np.arange(-5, 3)
TOKEN_EVERY = 3  # frames the attentive probe uses (LeVJEPA: 2 x 196 x 1024 fp16 = 0.8 MB each)


@torch.no_grad()
def latents(planner, f, batch=256):
    level1 = planner.model.get_level(1) if hasattr(planner.model, "get_level") else planner.model
    out = {"pixel": [], "full": [], "tokens": []}
    for i in range(0, len(f["pixels"]), batch):
        px = torch.from_numpy(f["pixels"][i:i + batch]).permute(0, 3, 1, 2).float() / 255.0
        pr = (torch.as_tensor(f["proprio"][i:i + batch]).float() - planner.proprio_mean) / planner.proprio_std
        o = level1.encode({"pixels": ((px - MEAN) / STD)[:, None].cuda(), "proprio": pr[:, None].cuda()})
        pix = o.get("pixel_embed_0", o["embed_0"])  # a vision-only encoder has no fusion
        out["pixel"].append(pix[:, 0].flatten(1).float().cpu())
        out["full"].append(o["embed_0"][:, 0].flatten(1).float().cpu())
        if pix.dim() == 4:  # token latents: (B, T, N, D)
            out["tokens"].append(pix[:, 0].half().cpu())
    tokens = torch.cat(out.pop("tokens")) if out["tokens"] else None
    out.pop("tokens", None)
    return {k: torch.cat(v).numpy().astype(np.float64) for k, v in out.items()}, tokens


class Ridge:
    """Ridge regression on standardized features. One eigendecomposition of the Gram matrix serves
    every target and strength (token latents have 8,192 features)."""

    def __init__(self, x):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        self.a = (x - self.mu) / self.sd
        self.e, self.v = np.linalg.eigh(self.a.T @ self.a)

    def predict(self, y, xt, lam):
        ym = y.mean(0)
        w = self.v @ ((self.v.T @ (self.a.T @ (y - ym))) / (self.e + lam * len(self.a))[:, None])
        return ((xt - self.mu) / self.sd) @ w + ym


def within_share(x, ep):
    """Share of the feature variance that is within episodes (1: all of it, 0: constant per episode)."""
    within = sum(x[ep == e].var(0).sum() * (ep == e).sum() for e in np.unique(ep)) / len(ep)
    return round(float(within / (x.var(0).sum() + 1e-12)), 3)


def score(y, p):
    return dict(rms_cm=round(float(np.sqrt(((y - p) ** 2).sum(1).mean()) * 100), 2),
                r2=round(float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()), 3))


class AttentiveProbe(nn.Module):
    def __init__(self, n_tokens, dim=1024, width=256, out=6):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
        self.pos = nn.Parameter(0.02 * torch.randn(1, n_tokens, width))
        self.query = nn.Parameter(0.02 * torch.randn(1, 1, width))
        self.attn = nn.MultiheadAttention(width, 8, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, out))

    def forward(self, t):
        x = self.inp(t.float()) + self.pos
        return self.head(self.attn(self.query.expand(len(x), -1, -1), x, x)[0][:, 0])


def attentive(tokens, y, fit, sel, val, epochs=40):
    """Train on the sel frames, keep the epoch with the best val error, refit nothing (the val
    split is part of fit, as in probe.py's ridge selection). Returns predictions for all frames."""
    torch.manual_seed(0)
    mu, sd = y[sel].mean(0), y[sel].std(0)
    yt = torch.as_tensor((y - mu) / sd, dtype=torch.float32)
    probe = AttentiveProbe(tokens.shape[1], dim=tokens.shape[2], out=y.shape[1]).cuda()
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * (int(sel.sum()) // 128 + 1))
    idx_sel, best, best_state = np.flatnonzero(sel), np.inf, None
    predict = lambda rows: torch.cat([probe(tokens[rows[k:k + 512]].cuda()).cpu() for k in range(0, len(rows), 512)])  # noqa: E731
    for _ in range(epochs):
        probe.train()
        for b in np.array_split(np.random.permutation(idx_sel), len(idx_sel) // 128 + 1):
            loss = ((probe(tokens[b].cuda()) - yt[b].cuda()) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        probe.eval()
        with torch.no_grad():
            err = float(((predict(np.flatnonzero(val)) - yt[val]) ** 2).mean())
        if err < best:
            best, best_state = err, {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        return predict(np.arange(len(tokens))).numpy() * sd + mu


def main(ckpt, eval_config, data, out):
    planner, f = Planner(ckpt, eval_config), h5py.File(data, "r")
    ep, cube, gp = f["ep_idx"][:], f["cube_pos"][:].astype(np.float64), f["ee"][:].astype(np.float64)
    eps = np.unique(ep)
    fit, sel = np.isin(ep, eps[: int(0.8 * len(eps))]), np.isin(ep, eps[: int(0.6 * len(eps))])
    test, val = ~fit, fit & ~sel
    feats, tokens = latents(planner, f)
    if np.array_equal(feats["pixel"], feats["full"]):  # a vision-only model: one feature set
        feats["full"] = feats["pixel"]
    feats["joints"] = f["proprio"][:].astype(np.float64)
    feats["none"] = np.zeros((len(ep), 1))
    targets = {"cube": cube, "grasp_point": gp, "cube_rel_grasp": cube - gp}
    table = test & (cube[:, 2] < CUBE_HALF + 0.003)
    report = {"ckpt": ckpt, "data": data, "frames_fit": int(fit.sum()), "frames_test": int(test.sum()),
              "frames_test_cube_on_table": int(table.sum())}
    fitted = {}
    for fname, x in feats.items():
        if fname != "none":
            report[f"{fname}_within_episode_var_share"] = within_share(x, ep)
        if id(x) not in fitted:
            fitted[id(x)] = Ridge(x[sel]), Ridge(x[fit])
        r_sel, r_fit = fitted[id(x)]
        for tname, y in targets.items():
            errs = [((y[val] - r_sel.predict(y[sel], x[val], lam)) ** 2).sum(1).mean() for lam in LAMBDAS]
            lam = float(LAMBDAS[int(np.argmin(errs))])
            p = r_fit.predict(y[fit], x, lam)
            report[f"{fname}->{tname}"] = dict(score(y[test], p[test]), lam=lam)
            if tname == "cube":
                report[f"{fname}->cube_on_table"] = dict(score(y[table], p[table]), lam=lam)
            print(fname, tname, report[f"{fname}->{tname}"], flush=True)
    if tokens is not None:
        rows = np.flatnonzero(np.arange(len(ep)) % TOKEN_EVERY == 0)
        p = attentive(tokens[rows], np.concatenate([cube, gp], 1)[rows], fit[rows], sel[rows], val[rows])
        for tname, cols, mask in (("cube", slice(0, 3), test), ("grasp_point", slice(3, 6), test),
                                  ("cube_on_table", slice(0, 3), table)):
            y = cube if tname != "grasp_point" else gp
            report[f"attentive_tokens->{tname}"] = score(y[rows][mask[rows]], p[mask[rows], cols])
            print("attentive", tname, report[f"attentive_tokens->{tname}"], flush=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])

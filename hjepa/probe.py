"""Linear probes: does the level-1 latent show the cube and the grasp point? (held-out episodes)

    uv run python hjepa/probe.py CKPT EVAL_CONFIG DATA.h5 OUT.json

Ridge regression from a feature set to a target. Fit on the first 80% of the episodes (the ridge
strength is selected on the last quarter of those), score on the other 20%: RMS error (cm) and R^2.
Features: the pixel latent (vision only), the full latent (what the planner compares), the joint
angles (no vision: a baseline) and none (the mean: the spread of the target). Targets: the cube,
the grasp point, the cube relative to the grasp point, and the cube on the frames where it rests on
the table (there, only vision can show it).
"""

import json
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from planner import MEAN, STD, Planner

CUBE_HALF = 0.0125  # sim/scene.py
LAMBDAS = 10.0 ** np.arange(-5, 3)


@torch.no_grad()
def latents(planner, f, batch=256):
    level1 = planner.model.get_level(1) if hasattr(planner.model, "get_level") else planner.model
    out = {"pixel": [], "full": []}
    for i in range(0, len(f["pixels"]), batch):
        px = torch.from_numpy(f["pixels"][i:i + batch]).permute(0, 3, 1, 2).float() / 255.0
        pr = (torch.as_tensor(f["proprio"][i:i + batch]).float() - planner.proprio_mean) / planner.proprio_std
        o = level1.encode({"pixels": ((px - MEAN) / STD)[:, None].cuda(), "proprio": pr[:, None].cuda()})
        out["pixel"].append(o["pixel_embed_0"][:, 0].flatten(1).float().cpu())
        out["full"].append(o["embed_0"][:, 0].flatten(1).float().cpu())
    return {k: torch.cat(v).numpy().astype(np.float64) for k, v in out.items()}


def ridge(xf, yf, xt, lam):
    mu, sd = xf.mean(0), xf.std(0) + 1e-6
    a, b, ym = (xf - mu) / sd, (xt - mu) / sd, yf.mean(0)
    w = np.linalg.solve(a.T @ a + lam * len(a) * np.eye(a.shape[1]), a.T @ (yf - ym))
    return b @ w + ym


def score(y, p):
    return dict(rms_cm=round(float(np.sqrt(((y - p) ** 2).sum(1).mean()) * 100), 2),
                r2=round(float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()), 3))


def main(ckpt, eval_config, data, out):
    planner, f = Planner(ckpt, eval_config), h5py.File(data, "r")
    ep, cube, gp = f["ep_idx"][:], f["cube_pos"][:].astype(np.float64), f["ee"][:].astype(np.float64)
    eps = np.unique(ep)
    fit, sel = np.isin(ep, eps[: int(0.8 * len(eps))]), np.isin(ep, eps[: int(0.6 * len(eps))])
    test, val = ~fit, fit & ~sel
    feats = latents(planner, f)
    feats["joints"] = f["proprio"][:].astype(np.float64)
    feats["none"] = np.zeros((len(ep), 1))
    targets = {"cube": cube, "grasp_point": gp, "cube_rel_grasp": cube - gp}
    table = test & (cube[:, 2] < CUBE_HALF + 0.003)
    report = {"ckpt": ckpt, "data": data, "frames_fit": int(fit.sum()), "frames_test": int(test.sum()),
              "frames_test_cube_on_table": int(table.sum())}
    for fname, x in feats.items():
        for tname, y in targets.items():
            errs = [((y[val] - ridge(x[sel], y[sel], x[val], lam)) ** 2).sum(1).mean() for lam in LAMBDAS]
            lam = float(LAMBDAS[int(np.argmin(errs))])
            p = ridge(x[fit], y[fit], x, lam)
            report[f"{fname}->{tname}"] = dict(score(y[test], p[test]), lam=lam)
            if tname == "cube":
                report[f"{fname}->cube_on_table"] = dict(score(y[table], p[table]), lam=lam)
            print(fname, tname, report[f"{fname}->{tname}"], flush=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])

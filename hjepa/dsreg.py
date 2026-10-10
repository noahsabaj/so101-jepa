"""DSReg diagnostic (Zheng, Klindt, Balestriero, SchÃ¶lkopf 2026, arXiv 2610.09457): once the rotation
that a SIGReg JEPA leaves free is removed, do single latents of our encoder line up with single world
factors (joint angles, cube position and yaw)?

    sh scripts/uvr python hjepa/dsreg.py CKPT EVAL_CONFIG VAL.h5 OUT.json [K] [MAX_FRAMES] [POOL] [ANCHORS] [NEIGHBOURS]

The encoder's latent of each frame (token latents flattened, 128 x 64 = 8,192 for v6) is reduced to
its top K principal directions and whitened (h, K = 32 by default): the representation before any
rotation. DSReg (the paper's Algorithm 1) then fits one orthogonal W on the fit episodes:
  observations x: the frame average-pooled by POOL (default 8: 8 x 16 x 3 = 384 features; 2: 32 x 64 x 3)
  at m anchors, a ridge fit of x on h over the anchor's k nearest neighbours in h gives the local
  Jacobian B_a = dx/dh (K x 384)
  W minimizes the mean L1 norm of W^T B_a over anchors (the sparsest dependency of the pixels on the
  rotated latents h W), by gradient steps on W = expm(skew) from several random starts.
No labels enter the fit. The factors are used only to score the held-out episodes (the last 20%):
  dense R2     ridge from all K latents to each factor (a rotation cannot change it)
  MCC          the mean |correlation| of each factor with its own latent, after a one-to-one matching
               (Hungarian) of factors to latents: 1 = each factor is one latent
  single R2    for each factor, the R2 of the best single latent (few-shot readout)
for the whitened PCA basis, a random rotation (a generic mixed basis), FastICA, DSReg, and two checks:
  oracle             the rotation closest to the factors (their readout directions made orthonormal; the other
                     K - F directions rotated to minimize the criterion): the best MCC any rotation can reach,
                     and the criterion at the factor basis. Criterion oracle > dsreg: on this data the sparsest
                     basis is not the factor basis (the paper's condition fails). oracle < dsreg: the search missed it.
  dsreg_from_oracle  DSReg started at the oracle: does the criterion keep the factor basis or leave it?
The cube's yaw is scored as sin and cos of 4 x yaw (the cube looks the same every 90 degrees).
"""

import json
import sys
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA

FACTORS = ["j0", "j1", "j2", "j3", "j4", "j5", "cube_x", "cube_y", "cube_z", "yaw4_sin", "yaw4_cos"]


def factors(f, n):
    q = f["cube_quat"][:n].astype(np.float64)  # (w, x, y, z)
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    return np.concatenate([f["proprio"][:n].astype(np.float64), f["cube_pos"][:n].astype(np.float64),
                           np.sin(4 * yaw)[:, None], np.cos(4 * yaw)[:, None]], 1)


def pooled_pixels(f, n, pool=8, batch=2048):
    out = []
    for i in range(0, n, batch):
        px = torch.from_numpy(f["pixels"][i:min(i + batch, n)]).permute(0, 3, 1, 2).float() / 255.0
        out.append(torch.nn.functional.avg_pool2d(px, pool).flatten(1))  # (B, 3 x 64/pool x 128/pool)
    return torch.cat(out).numpy().astype(np.float64)


def local_jacobians(h, x, m, k, lam, rng):
    """Ridge fits of x on h over the k nearest neighbours of m anchors: B_a = dx/dh, (m, K, p)."""
    ht, xt = torch.as_tensor(h, dtype=torch.float32).cuda(), torch.as_tensor(x, dtype=torch.float32).cuda()
    anchors = rng.choice(len(h), m, replace=False)
    out = []
    for a in anchors:
        nb = torch.cdist(ht[a:a + 1], ht).topk(k, largest=False).indices[0]
        hn, xn = ht[nb] - ht[nb].mean(0), xt[nb] - xt[nb].mean(0)
        g = hn.T @ hn + lam * k * torch.eye(h.shape[1], device="cuda")
        out.append(torch.linalg.solve(g, hn.T @ xn))
    return torch.stack(out)


def fit_rotation(b, restarts, steps, seed, start=None):
    """W in O(K) minimizing mean_a ||W^T B_a||_1 (the paper's anchor-averaged L1 criterion). As the
    authors' code (github.com/kunwuz/dsreg, dsreg/rotation.py): the first start is the identity (the
    PCA basis is kept unless a rotation beats it), the others random; Adam with the gradient clipped at 10.
    start: a (K, K) orthonormal basis in place of the identity as the first start."""
    torch.manual_seed(seed)
    kdim = b.shape[1]
    eye = torch.eye(kdim, device="cuda", dtype=b.dtype) if start is None else         torch.as_tensor(start, device="cuda", dtype=b.dtype)
    best = (float((eye.T @ b).abs().sum((1, 2)).mean()), eye.cpu().numpy().astype(np.float64))
    for r in range(restarts):
        w0 = eye if r == 0 else torch.linalg.qr(torch.randn(kdim, kdim, device="cuda", dtype=b.dtype))[0]
        s = torch.zeros(kdim, kdim, device="cuda", dtype=b.dtype, requires_grad=True)
        opt = torch.optim.Adam([s], lr=0.01)
        for _ in range(steps):
            w = w0 @ torch.linalg.matrix_exp(s - s.T)
            loss = (w.T @ b).abs().sum((1, 2)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([s], max_norm=10.0)
            opt.step()
        with torch.no_grad():
            w = w0 @ torch.linalg.matrix_exp(s - s.T)
            loss = float((w.T @ b).abs().sum((1, 2)).mean())
        if loss < best[0]:
            best = (loss, w.cpu().numpy().astype(np.float64))
    return best


def oracle_rotation(h, y, b, restarts, steps, seed):
    """The rotation closest to the factors: the ridge readout directions of the standardized factors from h, made
    orthonormal (the polar factor: the closest orthonormal set), and the remaining K - F directions rotated by
    DSReg's search to minimize the criterion."""
    yz = (y - y.mean(0)) / y.std(0)
    a = np.linalg.solve(h.T @ h + 1e-3 * len(h) * np.eye(h.shape[1]), h.T @ yz)  # (K, F)
    u, _, vt = np.linalg.svd(a, full_matrices=True)
    qf, qc = u[:, :a.shape[1]] @ vt, u[:, a.shape[1]:]
    bc = torch.einsum("kc,mkp->mcp", torch.as_tensor(qc, dtype=b.dtype, device="cuda"), b)
    _, rc = fit_rotation(bc, restarts, steps, seed)
    return np.c_[qf, qc @ rc]


def scores(zfit, zte, yfit, yte, names=FACTORS):
    """Dense R2 per factor, the matched MCC and the best single-latent R2 per factor (held-out)."""
    def r2(pred, y):
        return 1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
    a = np.c_[zfit, np.ones(len(zfit))]
    w = np.linalg.lstsq(a.T @ a + 1e-3 * np.eye(a.shape[1]), a.T @ yfit, rcond=None)[0]
    dense = r2(np.c_[zte, np.ones(len(zte))] @ w, yte)
    zc, yc = zte - zte.mean(0), yte - yte.mean(0)
    corr = np.abs((zc / (zc.std(0) + 1e-12)).T @ (yc / (yc.std(0) + 1e-12)) / len(zte))  # (K, factors)
    rows, cols = linear_sum_assignment(-corr)
    matched = np.zeros(yte.shape[1])
    matched[cols] = corr[rows, cols]
    single = []
    for j in range(yte.shape[1]):
        best = -np.inf
        for i in range(zte.shape[1]):
            c = np.polyfit(zfit[:, i], yfit[:, j], 1)
            best = max(best, float(r2(np.polyval(c, zte[:, i])[:, None], yte[:, j:j + 1])[0]))
        single.append(best)
    return dict(dense_r2={f: round(float(v), 3) for f, v in zip(names, dense)},
                mcc=round(float(matched.mean()), 3), matched_corr={f: round(float(v), 3) for f, v in zip(names, matched)},
                single_r2={f: round(v, 3) for f, v in zip(names, single)},
                single_r2_mean=round(float(np.mean(single)), 3))


def main(ckpt, eval_config, data, out, kdim=32, max_frames=0, pool=8, m=1024, k=128, lam=1e-2, restarts=12,
         steps=3000):
    from planner import Planner
    from probe import latents

    t0, max_frames, pool = time.time(), int(max_frames), int(pool)
    f = h5py.File(data, "r")
    n = min(len(f["pixels"]), max_frames) if max_frames else len(f["pixels"])
    planner = Planner(ckpt, eval_config)
    with torch.no_grad():
        feats, _ = latents(planner, {k_: f[k_] for k_ in ("pixels", "proprio")} if not max_frames else
                           {k_: f[k_][:n] for k_ in ("pixels", "proprio")})
    lat = feats["pixel"].astype(np.float32)
    ep, y, x = f["ep_idx"][:n], factors(f, n), pooled_pixels(f, n, pool)
    run(lat, x, y, ep, FACTORS, out, kdim, m, k, lam, restarts, steps, t0=t0,
        info=dict(ckpt=ckpt, data=data, pool=pool))


def run(lat, x, y, ep, names, out, kdim=32, m=1024, k=128, lam=1e-2, restarts=12, steps=3000, t0=None, info=None):
    """The analysis on any model: latents lat (N, D), observations x (N, p), factors y (N, F) named names, episode
    ids ep (the last 20% of episodes held out for scoring); writes the report to out."""
    t0, kdim, m, k = t0 or time.time(), int(kdim), int(m), int(k)
    rng = np.random.default_rng(0)
    n = len(lat)
    eps = np.unique(ep)
    fit = np.isin(ep, eps[: int(0.8 * len(eps))])
    test = ~fit
    mu = lat[fit].mean(0)
    _, sv, vt = np.linalg.svd(lat[fit][rng.choice(fit.sum(), min(fit.sum(), 8000), replace=False)] - mu,
                              full_matrices=False)
    proj = vt[:kdim].T / (sv[:kdim] / np.sqrt(min(fit.sum(), 8000)))  # whitened top-K directions
    h = ((lat - mu) @ proj).astype(np.float64)
    xs = (x - x[fit].mean(0)) / (x[fit].std(0) + 1e-6)
    print(f"{len(lat)} frames, latent {lat.shape[1]} -> K {kdim} (variance kept "
          f"{float((sv[:kdim] ** 2).sum() / (sv ** 2).sum()):.3f}); {time.time() - t0:.0f} s", flush=True)
    b = local_jacobians(h[fit], xs[fit], m, k, lam, rng)
    loss, w = fit_rotation(b, restarts, steps, seed=0)
    w_oracle = oracle_rotation(h[fit], y[fit], b, max(restarts // 3, 1), steps, seed=1)
    _, w_from_oracle = fit_rotation(b, 1, steps, seed=2, start=w_oracle)
    random_rot = np.linalg.qr(rng.standard_normal((kdim, kdim)))[0]
    ica = FastICA(n_components=kdim, whiten="unit-variance", random_state=0, max_iter=1000).fit(h[fit])
    with torch.no_grad():
        l1 = {name: float((torch.as_tensor(r, dtype=torch.float32).cuda().T @ b).abs().sum((1, 2)).mean())
              for name, r in (("pca", np.eye(kdim)), ("random", random_rot), ("dsreg", w), ("oracle", w_oracle),
                              ("dsreg_from_oracle", w_from_oracle))}
    print("criterion", {name: round(v, 1) for name, v in l1.items()}, flush=True)
    bases = {"pca": h, "random_rotation": h @ random_rot, "fastica": ica.transform(h), "dsreg": h @ w,
             "oracle": h @ w_oracle, "dsreg_from_oracle": h @ w_from_oracle}
    report = dict(**(info or {}), frames=int(n), frames_test=int(test.sum()), latent_dim=int(lat.shape[1]),
                  K=kdim, variance_kept=round(float((sv[:kdim] ** 2).sum() / (sv ** 2).sum()), 3),
                  observation_dim=int(x.shape[1]), anchors=m, neighbours=k, ridge=lam,
                  restarts=restarts, steps=steps, l1_criterion=l1)
    for name, z in bases.items():
        report[name] = scores(z[fit], z[test], y[fit], y[test], names)
        print(name, "MCC", report[name]["mcc"], "single R2", report[name]["single_r2_mean"], flush=True)
    report["seconds"] = round(time.time() - t0)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:10])

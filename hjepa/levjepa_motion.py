"""A2 step 1b: does LeVJEPA used as a video encoder (clips, as it was trained) show motion that one
frame cannot show?

    uv run python hjepa/levjepa_motion.py VAL_224.h5 OUT.json [K]

For every 3rd val frame t that has K - 1 earlier frames in its episode, each view is encoded three
ways, and the latent of t is the last temporal slot (its CLS and its 196 patch tokens per view;
with block-causal attention the last slot sees the whole clip):
  frame   t alone (T = 1: the image-encoder use of step 1)
  static  t repeated K times (a clip with no motion: same length and slot as video)
  video   frames t-K+1 .. t (T = K: the use LeVJEPA was trained for; 5 Hz here, 7.5 fps there)
plus 'stack': the CLS tokens of the K single frames side by side (an image encoder with a
history, which is what our predictor sees).

Targets. Motion, which one frame cannot show: grasp-point velocity, joint velocities, the previous
action, cube velocity while the cube is lifted. State, as in step 1: grasp point, resting cube.
Readouts: ridge on CLS + mean token, and step 1's attentive probe on the patch tokens pooled 2x2
(7x7 per view, the same for every encoding; full tokens for 3 encodings do not fit in memory). The
motion advantage: video better than static and frame on the motion targets (paired over the 20 test
episodes, bootstrap 95% interval), and no loss on the state targets.
"""

import json
import sys
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from transformers import AutoModel

from levjepa_probe import MEAN, REPO, REVISION, STD
from probe import CUBE_HALF, LAMBDAS, Ridge, attentive

DT = 0.2  # s per step (5 Hz)


def views(px):
    """uint8 (n, 224, 448, 3) scene | wrist -> normalized float (n, 2, 3, 224, 224)."""
    x = px.permute(0, 3, 1, 2).float().div(255).unflatten(3, (2, 224)).permute(0, 3, 1, 2, 4)
    return (x - MEAN.cuda().unsqueeze(1)) / STD.cuda().unsqueeze(1)


@torch.no_grad()
def encode_clips(model, v, clips, batch=8, tokens=True):
    """v: (n, 2, 3, H, W); clips: (B, T) frame indices. The last slot's CLS and mean patch token
    (B, 2048 each) and its patch tokens pooled 2x2 (B, 98, 1024), fp16 on the CPU."""
    cls, mean, tok = [], [], []
    for i in range(0, len(clips), batch):
        c = clips[i:i + batch]
        x = v[c.flatten()].unflatten(0, c.shape).permute(0, 2, 3, 1, 4, 5).flatten(0, 1)  # (2B, 3, T, H, W)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(pixel_values=x).last_hidden_state.float().unflatten(0, (-1, 2))  # (B, 2, 1 + T*196, D)
        cls.append(h[:, :, 0].flatten(1).half().cpu())
        last = h[:, :, -196:]
        mean.append(last.mean(2).flatten(1).half().cpu())
        if tokens:
            grid = last.unflatten(2, (14, 14)).permute(0, 1, 4, 2, 3).flatten(0, 1)  # (2B, D, 14, 14)
            pooled = torch.nn.functional.avg_pool2d(grid, 2).flatten(2).transpose(1, 2)  # (2B, 49, D)
            tok.append(pooled.unflatten(0, (-1, 2)).flatten(1, 2).half().cpu())
    return torch.cat(cls), torch.cat(mean), (torch.cat(tok) if tokens else None)


def ridge_predict(x, ys, sel, val, fit):
    """Ridge predictions of every target in ys (name: array), the strength selected per target."""
    r_sel, r_fit = Ridge(x[sel]), Ridge(x[fit])
    out = {}
    for name, y in ys.items():
        errs = [((y[val] - r_sel.predict(y[sel], x[val], lam)) ** 2).sum(1).mean() for lam in LAMBDAS]
        out[name] = r_fit.predict(y[fit], x, float(LAMBDAS[int(np.argmin(errs))]))
    return out


def summary(y, p, mask, ep, scale):
    """RMS (in target units x scale), R^2, and per-episode squared errors for the paired test."""
    e2 = ((y - p) ** 2).sum(1)
    eps = np.unique(ep[mask])
    per_ep = np.array([e2[mask & (ep == e)].mean() for e in eps])
    r2 = 1 - e2[mask].sum() / ((y[mask] - y[mask].mean(0)) ** 2).sum()
    return dict(rms=round(float(np.sqrt(e2[mask].mean()) * scale), 3), r2=round(float(r2), 3)), per_ep


def paired(a, b, rng, n=2000):
    """Bootstrap 95% interval over episodes of RMS(a) - RMS(b) (negative: a is better)."""
    d = [np.sqrt(a[i].mean()) - np.sqrt(b[i].mean()) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return [round(float(np.sqrt(a.mean()) - np.sqrt(b.mean())), 4), [round(float(q), 4) for q in np.percentile(d, [2.5, 97.5])]]


def main(data, out, K=4):
    K, t0, rng = int(K), time.time(), np.random.default_rng(0)
    model = AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True).cuda().eval()
    f = h5py.File(data, "r")
    offs, lens = f["ep_offset"][:], f["ep_len"][:]
    rows, frame_cls = [], {}
    feats = {k: {"cls": [], "mean": [], "tok": []} for k in ("frame", "static", "video")}
    for e, (s, n) in enumerate(zip(offs, lens)):
        v = views(torch.from_numpy(f["pixels"][s:s + n]).cuda())
        ts = np.array([t for t in range(K - 1, n) if (s + t) % 3 == 0])
        frame_cls[e] = encode_clips(model, v, torch.arange(n)[:, None], tokens=False)[0]  # T = 1, every frame
        clips = {"frame": torch.as_tensor(ts)[:, None],
                 "static": torch.as_tensor(ts)[:, None].repeat(1, K),
                 "video": torch.as_tensor(ts)[:, None] + torch.arange(-K + 1, 1)[None]}
        for k, c in clips.items():
            for name, x in zip(("cls", "mean", "tok"), encode_clips(model, v, c)):
                feats[k][name].append(x)
        rows += [(e, s + t, t) for t in ts]
        print(f"FLEET_PROGRESS {e + 1}/{len(lens)} ({time.time() - t0:.0f} s)", flush=True)
    del model
    ep = np.array([r[0] for r in rows])
    g = np.array([r[1] for r in rows])  # global frame index of t
    stack = torch.stack([torch.cat([frame_cls[e][t - j] for j in range(K)]) for e, _, t in rows])
    ee, cube = f["ee"][:].astype(np.float64), f["cube_pos"][:].astype(np.float64)
    q, act = f["proprio"][:].astype(np.float64), f["action"][:].astype(np.float64)
    targets = {  # name: (values at the rows, unit scale, frames that count)
        "grasp_point_velocity": ((ee[g] - ee[g - 1]) / DT, 100, None),  # cm/s
        "joint_velocity": ((q[g] - q[g - 1]) / DT, 1, None),  # rad/s
        "previous_action": (act[g - 1], 1, None),  # rad
        "cube_velocity_lifted": ((cube[g] - cube[g - 1]) / DT, 100, cube[g, 2] > CUBE_HALF + 0.01),  # cm/s
        "grasp_point": (ee[g], 100, None),  # cm
        "cube_on_table": (cube[g], 100, cube[g, 2] < CUBE_HALF + 0.003),  # cm
    }
    eps = np.unique(ep)
    fit, sel = np.isin(ep, eps[: int(0.8 * len(eps))]), np.isin(ep, eps[: int(0.6 * len(eps))])
    test, val = ~fit, fit & ~sel
    report = {"encoder": f"{REPO}@{REVISION}", "data": data, "K": K, "rows": len(rows), "rows_test": int(test.sum())}
    x_sets = {f"{k}_cls_mean": torch.cat([torch.cat(feats[k]["cls"]), torch.cat(feats[k]["mean"])], 1) for k in feats}
    x_sets["stack_cls"] = torch.as_tensor(stack)
    x_sets["none"] = torch.zeros(len(rows), 1)
    per_ep = {}
    for xname, x in x_sets.items():
        preds = ridge_predict(x.double().numpy(), {t: targets[t][0] for t in targets}, sel, val, fit)
        for tname, (y, scale, m) in targets.items():
            mask = test if m is None else test & m
            report[f"ridge_{xname}->{tname}"], per_ep[(xname, tname)] = summary(y, preds[tname], mask, ep, scale)
            print(xname, tname, report[f"ridge_{xname}->{tname}"], flush=True)
    ys = np.concatenate([targets[t][0] for t in targets], 1)
    ys_mu, ys_sd = ys[fit].mean(0), ys[fit].std(0) + 1e-9
    cols = np.cumsum([0] + [targets[t][0].shape[1] for t in targets])
    for k in ("frame", "static", "video"):
        tok = torch.cat(feats[k]["tok"])
        feats[k]["tok"] = None
        p = attentive(tok, (ys - ys_mu) / ys_sd, fit, sel, val) * ys_sd + ys_mu
        for i, (tname, (y, scale, m)) in enumerate(targets.items()):
            mask = test if m is None else test & m
            report[f"attentive_{k}->{tname}"], per_ep[(f"att_{k}", tname)] = summary(y, p[:, cols[i]:cols[i + 1]], mask, ep, scale)
            print("attentive", k, tname, report[f"attentive_{k}->{tname}"], flush=True)
        del tok
    for tname in targets:  # the paired tests: video against static and against frame
        for a, b in (("video", "static"), ("video", "frame"), ("video", "stack")):
            for kind, ka, kb in (("ridge", f"{a}_cls_mean", "stack_cls" if b == "stack" else f"{b}_cls_mean"),
                                 ("attentive", f"att_{a}", None if b == "stack" else f"att_{b}")):
                if kb is not None:
                    report[f"paired_{kind}_{a}_minus_{b}->{tname}"] = paired(per_ep[(ka, tname)], per_ep[(kb, tname)], rng)
    report["seconds"] = round(time.time() - t0)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3], *sys.argv[3:4])

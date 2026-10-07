"""A2 step 1: does the frozen LeVJEPA encoder show the cube and the grasp point? (held-out episodes)

    sh scripts/uvr python hjepa/levjepa_probe.py VAL_224.h5 OUT.json

LeVJEPA-VideoMix-Large (ViT-L/16, galilai-group on Hugging Face, revision pinned below; its model
code was read before use) encodes each view (scene, wrist) of each frame at 224x224 as a one-frame
clip: with block-causal attention these patch tokens equal the first slot of any clip. Episodes,
splits and targets are those of hjepa/probe.py (fit on the first 80% of the episodes, test on the
last 20%), so the numbers compare with the probes of our models and with A15's 2 cm mark.

Probes: ridge (linear) on the CLS tokens of both views, and on CLS plus mean patch token; an
attentive probe (one learned query attends over the 2 x 196 patch tokens; V-JEPA's standard
readout) on every 3rd frame. RMS error (cm) of the grasp point, the cube, and the cube resting on
the table.
"""

import json
import sys
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from transformers import AutoModel

from probe import CUBE_HALF, LAMBDAS, TOKEN_EVERY, Ridge, attentive, score, within_share

REPO, REVISION = "galilai-group/LeVJEPA-VideoMix-Large", "e831a0347737fcaa660b39c57d41c109de399845"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def encode(f, model, batch=32):
    cls, mean, tokens = [], [], []
    n = len(f["pixels"])
    for i in range(0, n, batch):
        px = torch.from_numpy(f["pixels"][i:i + batch]).cuda()  # (B, 224, 448, 3) scene | wrist
        x = px.permute(0, 3, 1, 2).float().div(255).unflatten(3, (2, 224)).permute(0, 3, 1, 2, 4)
        x = ((x.flatten(0, 1) - MEAN.cuda()) / STD.cuda()).unsqueeze(2)  # (2B, 3, T=1, 224, 224)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(pixel_values=x).last_hidden_state.float().unflatten(0, (-1, 2))  # (B, 2, 197, 1024)
        cls.append(h[:, :, 0].flatten(1).half().cpu())
        mean.append(h[:, :, 1:].mean(2).flatten(1).half().cpu())
        keep = [j for j in range(len(h)) if (i + j) % TOKEN_EVERY == 0]
        tokens.append(h[keep, :, 1:].flatten(1, 2).half().cpu())
        if i % (batch * 100) == 0:
            print(f"FLEET_PROGRESS {i}/{n}", flush=True)
    return torch.cat(cls), torch.cat(mean), torch.cat(tokens)


def main(data, out):
    t0 = time.time()
    model = AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True).cuda().eval()
    f = h5py.File(data, "r")
    ep, cube, gp = f["ep_idx"][:], f["cube_pos"][:].astype(np.float64), f["ee"][:].astype(np.float64)
    cls, mean, tokens = encode(f, model)
    print(f"encoded {len(cls)} frames in {time.time() - t0:.0f} s", flush=True)
    del model
    eps = np.unique(ep)
    fit, sel = np.isin(ep, eps[: int(0.8 * len(eps))]), np.isin(ep, eps[: int(0.6 * len(eps))])
    test, val = ~fit, fit & ~sel
    table = test & (cube[:, 2] < CUBE_HALF + 0.003)
    targets = {"cube": cube, "grasp_point": gp}
    report = {"encoder": f"{REPO}@{REVISION}", "data": data, "frames_test": int(test.sum()),
              "frames_test_cube_on_table": int(table.sum())}
    try:  # the same episodes as the 64x128 val set?
        with h5py.File(data.replace("_224", ""), "r") as g:
            report["same_episodes_as_64px"] = bool(np.allclose(g["cube_pos"][:], f["cube_pos"][:], atol=1e-6))
    except OSError:
        pass
    feats = {"cls": cls.double().numpy(), "cls_mean": torch.cat([cls, mean], 1).double().numpy(),
             "none": np.zeros((len(ep), 1))}
    for fname, x in feats.items():
        if fname != "none":
            report[f"{fname}_within_episode_var_share"] = within_share(x, ep)
        r_sel, r_fit = Ridge(x[sel]), Ridge(x[fit])
        for tname, y in targets.items():
            errs = [((y[val] - r_sel.predict(y[sel], x[val], lam)) ** 2).sum(1).mean() for lam in LAMBDAS]
            lam = float(LAMBDAS[int(np.argmin(errs))])
            p = r_fit.predict(y[fit], x, lam)
            report[f"ridge_{fname}->{tname}"] = dict(score(y[test], p[test]), lam=lam)
            if tname == "cube":
                report[f"ridge_{fname}->cube_on_table"] = dict(score(y[table], p[table]), lam=lam)
            print(fname, tname, report[f"ridge_{fname}->{tname}"], flush=True)
    rows = np.flatnonzero(np.arange(len(ep)) % TOKEN_EVERY == 0)
    p = attentive(tokens, np.concatenate([cube, gp], 1)[rows], fit[rows], sel[rows], val[rows])
    for tname, cols, mask in (("cube", slice(0, 3), test), ("grasp_point", slice(3, 6), test),
                              ("cube_on_table", slice(0, 3), table)):
        y = cube if tname != "grasp_point" else gp
        m = mask[rows]
        report[f"attentive_tokens->{tname}"] = score(y[rows][m], p[m, cols])
        print("attentive", tname, report[f"attentive_tokens->{tname}"], flush=True)
    report["seconds"] = round(time.time() - t0)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])

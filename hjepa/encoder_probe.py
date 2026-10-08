"""A2 step 1, all frozen encoders on one footing: does a frozen pretrained video encoder show the cube
and the grasp point? (held-out episodes)

    sh scripts/uvr python hjepa/encoder_probe.py VAL_224.h5 OUT.json ENCODER [BATCH]

ENCODER: levjepa (LeVJEPA-VideoMix-Large, ViT-L, Hugging Face, revision pinned below), or V-JEPA 2.1
vjepa21_L (ViT-L/16, 300M, distilled from G), vjepa21_g (ViT-g/16, 1B) or vjepa21_G (ViT-G/16, 2B):
weights in data/weights/ (dl.fbaipublicfiles.com/vjepa2/), model code from facebookresearch/vjepa2 at
the commit fetched by hjepa/a2_encoders.sh (VJEPA_SRC, default data/vjepa2_src).

Each view (scene, wrist) of each frame is encoded at 224x224 as one image (V-JEPA 2.1's image path;
a one-frame clip for LeVJEPA). Episodes, splits and targets are those of hjepa/probe.py (fit on the
first 80% of the episodes, test on the last 20%), so the numbers compare with the probes of our models
and with A15's 2 cm mark. Readouts, the same for every encoder:
  ridge     on the mean patch token of each view (and on CLS for LeVJEPA, which has one)
  attentive one learned query over the patch tokens pooled 2x2 (7x7 per view: about the 8x8 token
            grid of our 64x64 views, and it keeps the token store under 4 GB for G), every 3rd frame
RMS error (cm) of the grasp point, the cube, the cube relative to the grasp point, and the cube
resting on the table (A15's mark).
"""

import json
import os
import sys
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F

from probe import CUBE_HALF, LAMBDAS, TOKEN_EVERY, Ridge, attentive, score, within_share

LEVJEPA = ("galilai-group/LeVJEPA-VideoMix-Large", "e831a0347737fcaa660b39c57d41c109de399845")
VJEPA21 = {  # name: (architecture, local file, checkpoint key), as in vjepa2's src/hub/backbones.py.
    # Local names differ in more than case: Meta's vitg and vitG files collide on Windows (NTFS).
    "vjepa21_L": ("vit_large", "vjepa21_vitl_300m.pt", "ema_encoder"),
    "vjepa21_g": ("vit_giant_xformers", "vjepa21_vitg_1b.pt", "target_encoder"),
    "vjepa21_G": ("vit_gigantic_xformers", "vjepa21_vitG_2b.pt", "target_encoder"),
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_vjepa21(name):
    sys.path.insert(0, os.environ.get("VJEPA_SRC", "data/vjepa2_src"))
    from app.vjepa_2_1.models import vision_transformer as vit

    arch, file, key = VJEPA21[name]
    model = vit.__dict__[arch](patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True,
                               use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
                               img_temporal_dim_size=1, interpolate_rope=True)
    path = f"data/weights/{file}"
    try:  # mmap: the files hold several state dicts (up to 30 GB); only the encoder's tensors are read
        ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception:  # noqa: BLE001  (the checkpoints may hold non-tensor objects; Meta's own files)
        ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt[key].items()}
    model.load_state_dict(state, strict=True)
    del ckpt, state
    return model.to(torch.bfloat16)


def load(name):
    if name == "levjepa":
        from transformers import AutoModel

        return AutoModel.from_pretrained(LEVJEPA[0], revision=LEVJEPA[1], trust_remote_code=True)
    return load_vjepa21(name)


@torch.no_grad()
def encode(f, name, model, batch):
    """Per frame: the mean patch token per view (2 x D), the CLS per view (LeVJEPA), and for every
    TOKEN_EVERY-th frame the patch tokens pooled 2x2 (2 x 49 x D)."""
    mean, cls, tokens = [], [], []
    n = len(f["pixels"])
    for i in range(0, n, batch):
        px = torch.from_numpy(f["pixels"][i:i + batch]).cuda()  # (B, 224, 448, 3) scene | wrist
        x = px.permute(0, 3, 1, 2).float().div(255).unflatten(3, (2, 224)).permute(0, 3, 1, 2, 4)
        x = ((x.flatten(0, 1) - MEAN.cuda()) / STD.cuda()).unsqueeze(2)  # (2B, 3, T=1, 224, 224)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if name == "levjepa":
                h = model(pixel_values=x).last_hidden_state
                cls.append(h[:, 0].float().unflatten(0, (-1, 2)).flatten(1).half().cpu())
                h = h[:, 1:]
            else:
                h = model(x.to(torch.bfloat16))
        h = h.float().unflatten(0, (-1, 2))  # (B, 2, 196, D)
        mean.append(h.mean(2).flatten(1).half().cpu())
        keep = [j for j in range(len(h)) if (i + j) % TOKEN_EVERY == 0]
        if keep:
            g = h[keep].unflatten(2, (14, 14)).permute(0, 1, 4, 2, 3).flatten(0, 1)  # (2K, D, 14, 14)
            g = F.avg_pool2d(g, 2).flatten(2).transpose(1, 2).unflatten(0, (-1, 2))  # (K, 2, 49, D)
            tokens.append(g.flatten(1, 2).half().cpu())
        if i % (batch * 50) == 0:
            print(f"FLEET_PROGRESS {i}/{n}", flush=True)
    return torch.cat(mean), (torch.cat(cls) if cls else None), torch.cat(tokens)


def main(data, out, name, batch=None):
    t0 = time.time()
    model = load(name).cuda().eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params / 1e6:.1f}M parameters, loaded in {time.time() - t0:.0f} s", flush=True)
    batch = int(batch or {"vjepa21_G": 8, "vjepa21_g": 16}.get(name, 32))
    f = h5py.File(data, "r")
    ep, cube, gp = f["ep_idx"][:], f["cube_pos"][:].astype(np.float64), f["ee"][:].astype(np.float64)
    t1 = time.time()
    mean, cls, tokens = encode(f, name, model, batch)
    enc_s = time.time() - t1
    print(f"encoded {len(mean)} frames in {enc_s:.0f} s", flush=True)
    del model
    torch.cuda.empty_cache()
    eps = np.unique(ep)
    fit, sel = np.isin(ep, eps[: int(0.8 * len(eps))]), np.isin(ep, eps[: int(0.6 * len(eps))])
    test, val = ~fit, fit & ~sel
    table = test & (cube[:, 2] < CUBE_HALF + 0.003)
    targets = {"cube": cube, "grasp_point": gp, "cube_rel_grasp": cube - gp}
    report = {"encoder": name, "weights": LEVJEPA if name == "levjepa" else VJEPA21[name][1], "params": params,
              "data": data, "frames_test": int(test.sum()), "frames_test_cube_on_table": int(table.sum()),
              "encode_seconds": round(enc_s), "frames_per_second": round(len(mean) / enc_s, 1),
              "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    try:  # the same episodes as the 64x128 val set?
        with h5py.File(data.replace("_224", ""), "r") as g:
            report["same_episodes_as_64px"] = bool(np.allclose(g["cube_pos"][:], f["cube_pos"][:], atol=1e-6))
    except OSError:
        pass
    feats = {"mean": mean.double().numpy(), "none": np.zeros((len(ep), 1))}
    if cls is not None:
        feats["cls"] = cls.double().numpy()
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
    p = attentive(tokens, np.concatenate([cube, gp, cube - gp], 1)[rows], fit[rows], sel[rows], val[rows])
    for tname, cols, y, mask in (("cube", slice(0, 3), cube, test), ("grasp_point", slice(3, 6), gp, test),
                                 ("cube_rel_grasp", slice(6, 9), cube - gp, test),
                                 ("cube_on_table", slice(0, 3), cube, table)):
        m = mask[rows]
        report[f"attentive_pooled->{tname}"] = score(y[rows][m], p[m, cols])
        print("attentive", tname, report[f"attentive_pooled->{tname}"], flush=True)
    report["seconds"] = round(time.time() - t0)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])

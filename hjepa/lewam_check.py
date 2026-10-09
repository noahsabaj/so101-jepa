"""Offline check of a LeWAM run's action head through our adapter (hjepa/lewam_planner.py), on expert episodes:
from real history frames and a real goal frame h chunks ahead, does it propose the expert's next actions?

    sh scripts/uvr python hjepa/lewam_check.py CKPT DATA.h5 [SAMPLES]

Prints the normalized mean squared error of the proposed actions against the expert's, beside two baselines:
the mean action (0 in normalized units: MSE ~1) and the expert's actions of another, random sample. An adapter
that feeds the model wrongly (views, layout, history order, goal, horizon, normalization) scores near the
baselines; a model that has learned the task scores well below them.
"""

import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from lewam_planner import LeWAMAdapter


def main(ckpt, data, samples=200):
    rng = np.random.default_rng(0)
    ad = LeWAMAdapter(ckpt, "so101_lewam_policy", seed=0)
    fs, hs, hl = ad.fs, ad.hist_stride, ad.hist_len
    with h5py.File(data, "r") as f:
        px, act = f["pixels"], f["action"][:]
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        errs, base_other, hs_used = [], [], []
        for _ in range(samples):
            e = rng.integers(len(off))
            t = int(rng.integers(0, ln[e] - fs * 2))
            h = int(rng.integers(1, min(ad.H_max, (ln[e] - 1 - t) // fs) + 1))
            g = off[e] + t + h * fs
            rows = [t - k * hs for k in range(hl - 1, -1, -1) if t - k * hs >= 0]
            z = ad.encode_views(np.stack([px[off[e] + r] for r in rows]))  # (len(rows), 2, z)
            history = torch.zeros(1, 2, hl, ad.model.z_dim, device="cuda")
            pad = torch.ones(1, hl, dtype=torch.bool, device="cuda")
            history[0, :, hl - len(rows):] = z.permute(1, 0, 2)
            pad[0, hl - len(rows):] = False
            z_goal = ad.encode_views(px[g][None])[:, ad.planner.goal_view_indices]
            ad.planner._steps_left = np.array([float(h)])
            with torch.no_grad():
                a = ad.planner._propose({"z_goal": z_goal}, [0], history, pad)[0].float().cpu().numpy()
            n = len(a)
            truth = (act[off[e] + t:off[e] + t + n] - ad.action_mean) / ad.action_std
            if len(truth) < n:
                continue
            errs.append(((a - truth) ** 2).mean())
            hs_used.append(h)
            o = rng.integers(len(act) - n)
            base_other.append((((act[o:o + n] - ad.action_mean) / ad.action_std - truth) ** 2).mean())
    print(f"{ckpt}: {len(errs)} samples, proposed actions vs expert (normalized MSE): {np.mean(errs):.3f} "
          f"(median {np.median(errs):.3f}); mean-action baseline ~1; another sample's actions "
          f"{np.mean(base_other):.3f}", flush=True)
    errs, hs_used = np.array(errs), np.array(hs_used)
    for lo, hi in ((1, 1), (2, 4), (5, 10), (11, 20)):
        m = (hs_used >= lo) & (hs_used <= hi)
        if m.any():
            print(f"  goal {lo}-{hi} chunks ahead: {errs[m].mean():.3f} (n {m.sum()})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:4]))

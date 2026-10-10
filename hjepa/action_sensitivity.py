"""Action sensitivity of a world model (Strohm et al. 2026, arXiv 2610.03137, PLAN.md section 10): from held-out
frames, predict the next latent under K random actions. spread/motion = the mean per-dimension variance of the K
predictions over their mean squared displacement from the current latent (scale-free, 0 to 1; 0: the actions
change nothing). In the paper all 28 configurations below 0.05 failed to plan: a necessary condition, no env needed.

    sh scripts/uvr python hjepa/action_sensitivity.py CKPT DATA.h5 OUT.json [STRIDES] [CASES] [K]

STRIDES as hjepa/jump_test.py ("1"; "1,2,3,5,10" for v31, v36): the step is one 1-step action. The K actions are
random rows of the data's actions (its own scale); the 3 history actions are the true ones. Reported overall and
per phase of the expert (sim/collect.py PHASES: grasp, lift ... are where the cube must respond), with the share
of the frame that changes between t and t + 1 in the data (pixels moving more than 10/255: the paper's response
fraction, 0.35% on SLIM where plain LeWM failed, 0.6% on PushT where it planned).
"""

import json
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from planner import MEAN, STD, Planner

PHASES = ["start", "approach", "descend", "grasp", "lift", "carry", "lower", "release", "retreat", "reach", "free"]


@torch.no_grad()
def main(ckpt, data, out, strides="1", cases=600, k=32):
    strides, cases, k = [int(s) for s in str(strides).split(",")], int(cases), int(k)
    kmax = max(strides)
    planner, f = Planner(ckpt, "so101_flat"), h5py.File(data, "r")
    level1 = planner.model.get_level(1) if hasattr(planner.model, "get_level") else planner.model
    lens, offs, phase = f["ep_len"][:], f["ep_offset"][:], f["phase"][:]
    actions = f["action"][:]
    rng = np.random.default_rng(0)
    pool = [(e, t) for e, n in enumerate(lens) for t in range(3, int(n) - 1)]
    ids = [pool[i] for i in sorted(rng.choice(len(pool), min(cases, len(pool)), replace=False))]

    def pack(a):  # (n, 6) raw actions of 1 step each -> the model's action input, as jump_test.act with k = 1
        a = planner.normalize_action(a)
        if kmax == 1:
            return a
        pad = np.zeros((len(a), kmax * a.shape[1] + 1))
        pad[:, :a.shape[1]] = a
        pad[:, -1] = 1 / kmax
        return pad

    def encode(rows):
        px = torch.from_numpy(f["pixels"][rows]).permute(0, 3, 1, 2).float().cuda() / 255.0
        pr = (torch.as_tensor(f["proprio"][rows]).float() - planner.proprio_mean) / planner.proprio_std
        o = level1.encode({"pixels": ((px - MEAN.cuda()) / STD.cuda())[:, None], "proprio": pr[:, None].cuda()})
        return o["embed_0"][:, 0].float()

    ratio, motion_true, case_phase, frac = [], [], [], []
    for e, t in ids:
        r = offs[e] + t
        z = encode(list(range(r - 3, r + 2)))  # frames t-3 .. t+1
        hist = z[None, :4].expand(k, *z[:4].shape)
        cand = actions[rng.choice(len(actions), k, replace=False)]
        a = np.stack([np.concatenate([pack(actions[r - 3:r]), pack(cand[i:i + 1])]) for i in range(k)])
        pred = level1.predict(hist, level1.action_embed(torch.as_tensor(a, dtype=torch.float32, device="cuda")))
        zk = pred[:, -1].flatten(1)  # (K, D)
        zt = z[3].flatten()
        spread = zk.var(0, unbiased=False).mean()
        motion = ((zk - zt) ** 2).mean()
        ratio.append(float(spread / motion))
        motion_true.append(float(((z[4].flatten() - zt) ** 2).mean() / motion))
        case_phase.append(int(phase[r]))
        d = np.abs(f["pixels"][r + 1].astype(np.int16) - f["pixels"][r].astype(np.int16)).max(-1)
        frac.append(float((d > 10).mean()))
    ratio, case_phase = np.asarray(ratio), np.asarray(case_phase)
    report = dict(ckpt=ckpt, data=data, strides=strides, cases=len(ids), K=k,
                  spread_over_motion=round(float(ratio.mean()), 4), median=round(float(np.median(ratio)), 4),
                  below_0_05=round(float((ratio < 0.05).mean()), 3),
                  true_step_over_predicted_motion=round(float(np.mean(motion_true)), 3),
                  frame_fraction_changed=round(float(np.mean(frac)), 4),
                  per_phase={PHASES[p]: dict(n=int((case_phase == p).sum()),
                                             spread_over_motion=round(float(ratio[case_phase == p].mean()), 4))
                             for p in np.unique(case_phase)})
    print(json.dumps(report, indent=1), flush=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:7])

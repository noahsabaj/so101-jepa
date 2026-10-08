"""Long-horizon prediction with fixed steps against time-step jumps (PLAN.md A21).

    sh scripts/uvr python hjepa/jump_test.py CKPT DATA.h5 OUT.json [STRIDES] [cases]

From 4 history frames at frame t, predict the latent of frame t + n (n = 1, 5, 10, 20 steps: 0.2 to
4 s) with the true actions. A fixed-step model (STRIDES "1") rolls out n one-step predictions; a
time-step model (STRIDES "1,2,3,5,10", v31) takes n / k jumps of the largest stride k that divides
n, with history frames k apart. The score does not depend on the latent space, so models compare:
the predicted latent is matched against the encoded latents of every frame of the episode, and the
time error is |matched frame - (t + n)| in seconds (0 is perfect), plus the share matched exactly.
The cases are the same for every model, stride and horizon (rule 6, paired): frames with 3 history
steps of CASE_STRIDE before them and the longest horizon after, drawn with seed 0. OUT keeps their
(episode, frame) ids, the data file's sha256 and each case's error, for hjepa/compare.py.
"""

import json
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from planner import MEAN, STD, Planner
from value import sha256_file

HORIZONS = (1, 5, 10, 20)
CASE_STRIDE = 10  # the largest stride of any model compared (v31: 1,2,3,5,10): the cases do not depend on the model


def common_cases(lens, cases, seed=0):
    """(episode, frame) cases admissible for every stride up to CASE_STRIDE and every horizon, in episode order."""
    pool = [(e, t) for e, n in enumerate(lens) for t in range(3 * CASE_STRIDE, int(n) - max(HORIZONS))]
    pick = np.random.default_rng(seed).choice(len(pool), min(cases, len(pool)), replace=False)
    return [pool[i] for i in sorted(pick)]


@torch.no_grad()
def main(ckpt, data, out, strides="1", cases=300):
    strides, cases = [int(k) for k in str(strides).split(",")], int(cases)
    kmax = max(strides)
    if kmax > CASE_STRIDE:
        raise ValueError(f"stride {kmax} > CASE_STRIDE {CASE_STRIDE}: the common cases have too little history")
    planner, f = Planner(ckpt, "so101_flat"), h5py.File(data, "r")
    level1 = planner.model.get_level(1) if hasattr(planner.model, "get_level") else planner.model
    lens, offs = f["ep_len"][:], f["ep_offset"][:]
    zs = None  # encoded latents of every frame (on the CPU, filled in place), batched as in hjepa/probe.py
    for i in range(0, len(f["pixels"]), 512):
        px = torch.from_numpy(f["pixels"][i:i + 512]).permute(0, 3, 1, 2).float().cuda() / 255.0
        pr = (torch.as_tensor(f["proprio"][i:i + 512]).float() - planner.proprio_mean) / planner.proprio_std
        o = level1.encode({"pixels": ((px - MEAN.cuda()) / STD.cuda())[:, None], "proprio": pr[:, None].cuda()})
        z = o["embed_0"][:, 0].float().cpu()
        if zs is None:
            zs = torch.empty((len(f["pixels"]), *z.shape[1:]), dtype=torch.float32)
        zs[i:i + len(z)] = z

    gpu = {}

    def episode(e):  # only the current episode's latents on the GPU
        if e not in gpu:
            gpu.clear()
            gpu[e] = zs[offs[e]:offs[e] + lens[e]].cuda()
        return gpu[e]

    def act(e, t, k):  # the step from frame t: k raw actions (normalized), as the model was trained
        a = planner.normalize_action(f["action"][offs[e] + t:offs[e] + t + k])
        if kmax == 1:
            return a.reshape(-1)
        pad = np.zeros((kmax, a.shape[1]))
        pad[:k] = a
        return np.concatenate([pad.reshape(-1), [k / kmax]])

    ids = common_cases(lens, cases)
    report = {"ckpt": ckpt, "strides": strides, "cases": len(ids), "data": data,
              "data_fingerprint": sha256_file(data), "case_stride": CASE_STRIDE, "case_ids": [list(c) for c in ids]}
    for n in HORIZONS:
        k = max(s for s in strides if n % s == 0)
        errs = []
        for e, t in ids:
            z = episode(e)
            hist = z[[t - 3 * k, t - 2 * k, t - k, t]][None]
            acts = [act(e, t + j * k - 3 * k, k) for j in range(3)]  # the actions between history frames
            for j in range(n // k):
                acts.append(act(e, t + j * k, k))
                a = torch.as_tensor(np.stack(acts[-4:]), dtype=torch.float32, device="cuda")[None]
                nxt = level1.predict(hist[:, -4:], level1.action_embed(a))[:, -1:]
                hist = torch.cat([hist, nxt], 1)
            d = ((z - hist[0, -1]) ** 2).flatten(1).mean(1)
            errs.append(abs(int(d.argmin()) - (t + n)))
        errs = np.asarray(errs) * 0.2
        label = f"{n * 0.2:.1f}s"
        report[label] = dict(stride=k, time_error_s=round(float(errs.mean()), 3), exact=round(float((errs == 0).mean()), 3),
                             time_error_ci95_s=round(float(1.96 * errs.std(ddof=1) / np.sqrt(len(errs))), 3))
        report[f"time_error_{label}"] = [report[label]["time_error_s"], report[label]["time_error_ci95_s"]]
        report[f"time_error_{label}_cases"] = [round(float(x), 1) for x in errs]
        print(n, report[label], flush=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:6])

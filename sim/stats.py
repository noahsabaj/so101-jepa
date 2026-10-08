"""Quality numbers for a dataset made by collect.py (run where the data is).

    stats.py DATA.h5 [DATA.h5 ...]

Per file: episodes, steps, phase shares, the expert's pick-and-place results, cubes that left
the table, action and joint ranges, NaNs, and dark or flat frames in a sample.
"""

import json
import sys

import h5py
import hdf5plugin  # noqa: F401  (registers the Blosc filter)
import numpy as np

from collect import PHASES
from env import MAX_ACTION
from scene import CUBE_HALF

PICK = [PHASES.index(p) for p in ("approach", "descend", "grasp", "lift", "carry", "lower", "release", "retreat")]
LIFT, CARRY, LOWER, RETREAT = (PHASES.index(p) for p in ("lift", "carry", "lower", "retreat"))
FRAMES = 600  # sampled frames per file


def attempts(phase):
    """(start, end) of each complete pick-and-place in one episode: an 'approach' run through the
    end of its 'retreat' run. An attempt that the episode cuts off is left out."""
    out, start = [], None
    for t in range(len(phase)):
        if phase[t] == PICK[0] and (t == 0 or phase[t - 1] != PICK[0]):
            start = t
        if start is not None and phase[t] == RETREAT and (t + 1 == len(phase) or phase[t + 1] != RETREAT):
            if t + 1 < len(phase):
                out.append((start, t + 1))
            start = None
    return out


def pick_outcomes(phase, cube, ee):
    """One dict per complete pick-and-place of one episode. Lifted: the cube rose 5 cm. Placed:
    lifted, then at rest within 2 cm (xy) of the grasp point's place at the end of 'lower'."""
    out = []
    for a, b in attempts(phase):
        p, c, g = phase[a:b], cube[a:b], ee[a:b]
        up = np.isin(p, [LIFT, CARRY])
        lifted = bool(up.any() and c[up, 2].max() >= CUBE_HALF + 0.05)
        carry, lower = np.flatnonzero(p == CARRY), np.flatnonzero(p == LOWER)
        held = lifted and carry.size and c[carry[-1], 2] >= CUBE_HALF + 0.05
        err = float(np.linalg.norm(cube[b, :2] - g[lower[-1], :2])) if lower.size else np.inf
        resting = bool(cube[b, 2] < CUBE_HALF + 0.004)
        placed = lifted and resting and err < 0.02
        kind = ("not_lifted" if not lifted else "dropped_in_carry" if not held else
                "not_resting" if not resting else "placed" if placed else "placed_off_target")
        out.append(dict(kind=kind, lifted=lifted, placed=placed, place_err=err,
                        start_r=float(np.linalg.norm(c[0, :2]))))
    return out


def summary(outcomes):
    """Counts over many pick-and-place outcomes, with the lift rate by the cube's start radius."""
    kinds, by_r = {}, {}
    for o in outcomes:
        kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
        r = f"r{int(o['start_r'] * 100) // 3 * 3}cm"
        t, u = by_r.get(r, (0, 0))
        by_r[r] = (t + 1, u + o["lifted"])
    errs = [o["place_err"] for o in outcomes if np.isfinite(o["place_err"])]
    return {
        "pick_attempts": len(outcomes), "lifted": sum(o["lifted"] for o in outcomes),
        "placed_within_2cm": sum(o["placed"] for o in outcomes),
        "place_err_cm_median_p90": [round(100 * float(np.median(errs)), 2), round(100 * float(np.percentile(errs, 90)), 2)],
        "outcomes": dict(sorted(kinds.items())),
        "lift_rate_by_start_radius": {k: f"{u}/{t}" for k, (t, u) in sorted(by_r.items(), key=lambda kv: int(kv[0][1:-2]))},
    }


def stats(path):
    f = h5py.File(path, "r")
    offsets, lengths = f["ep_offset"][:], f["ep_len"][:]
    phase, cube, ee = f["phase"][:], f["cube_pos"][:], f["ee"][:]
    action, proprio = f["action"][:], f["proprio"][:]
    outcomes, off_table = [], 0
    for start, n in zip(offsets, lengths):
        s = slice(start, start + n)
        outcomes += pick_outcomes(phase[s], cube[s], ee[s])
        off_table += bool((cube[s, 2] < 0).any())
    share = np.bincount(phase, minlength=len(PHASES)) / len(phase)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(phase), size=min(FRAMES, len(phase)), replace=False))
    frames = np.stack([f["pixels"][i] for i in idx]).astype(np.float32)
    half = frames.shape[2] // 2
    scene, wrist = frames[:, :, :half], frames[:, :, half:]
    report = {
        "file": path, "episodes": len(lengths), "steps": int(lengths.sum()),
        "episode_len": [int(lengths.min()), int(lengths.max())],
        "phase_share": {p: round(float(s), 3) for p, s in zip(PHASES, share)},
        **summary(outcomes),
        "episodes_cube_off_table": off_table,
        "action_abs_p99_over_max": np.round(np.percentile(np.abs(action), 99, axis=0) / MAX_ACTION, 2).tolist(),
        "proprio_min": np.round(proprio.min(0), 2).tolist(), "proprio_max": np.round(proprio.max(0), 2).tolist(),
        "nan": int(np.isnan(action).sum() + np.isnan(proprio).sum()),
        "frames_sampled": len(idx),
        "dark_frames": {"scene": int((scene.mean((1, 2, 3)) < 15).sum()), "wrist": int((wrist.mean((1, 2, 3)) < 15).sum())},
        "flat_frames": {"scene": int((scene.std((1, 2, 3)) < 3).sum()), "wrist": int((wrist.std((1, 2, 3)) < 3).sum())},
        "mean_brightness": {"scene": round(float(scene.mean()), 1), "wrist": round(float(wrist.mean()), 1)},
    }
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        stats(p)

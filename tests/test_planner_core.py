"""Tests of the planner and the offline tests (hjepa/planner.py, hjepa/offline.py): normalization as in
training, the env's action bound in the solver, the value head's checkpoint binding, non-finite costs,
offline history and case identity (codex audit findings 2, 11, 12, 13, 27). CPU only, stand-in model.

    .venv/Scripts/python.exe -m pytest tests/test_planner_core.py     (or: python tests/test_planner_core.py)
"""

import sys
import tempfile
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for d in ("sim", "hjepa"):
    sys.path.insert(0, str(ROOT / d))

import offline  # noqa: E402
import planner  # noqa: E402
from data import ZScore, build_normalizer_artifact, save_normalizer_artifact  # noqa: E402
from env import MAX_ACTION  # noqa: E402
from stable_worldmodel.solver import CEMSolver  # noqa: E402


class CostModel(torch.nn.Module):
    """Cost: minus the sum of the first planned step (prefers ever larger actions)."""

    level = 1

    def get_cost(self, info, actions):
        return -actions[:, :, 0].sum(-1)

    def rollout(self, info, actions):
        return {}


class Columns:
    """A dataset of given columns, for build_normalizer_artifact."""

    def __init__(self, **cols):
        self.cols = cols

    def get_col_data(self, col):
        return self.cols[col]


def make_run(tmp, action):
    """A run dir as training leaves it: config.yaml, normalizer.pt (from the given actions), a checkpoint."""
    tmp = Path(tmp)
    cfg = OmegaConf.create({"data": {"dataset": {"name": "x", "keys_to_load": ["pixels", "action", "proprio"]}},
                            "level1": {"wm": {"history_size": 4}}})
    OmegaConf.save(cfg, tmp / "config.yaml")
    proprio = np.random.default_rng(0).normal(size=(50, 6))
    save_normalizer_artifact(build_normalizer_artifact(cfg, Columns(action=action, proprio=proprio)), tmp)
    (tmp / "model.ckpt").write_bytes(b"weights")
    return tmp / "model.ckpt"


def make_planner(ckpt, eval_config="so101_flat_cem", **patch):
    saved = planner.load_model, planner.build_solver
    planner.load_model = lambda cfg: torch.nn.Linear(1, 1)
    planner.build_solver = lambda cfg, model: CEMSolver(CostModel(), n_steps=3, num_samples=64, topk=8,
                                                        action_clip_sigma=2.0, device="cpu")
    try:
        return planner.Planner(str(ckpt), eval_config)
    finally:
        planner.load_model, planner.build_solver = saved


def constant_wrist_actions():
    a = np.random.default_rng(1).normal(scale=0.05, size=(50, 6))
    a[:, 4] = 0.0  # wrist roll never moves: std 0
    return a


def test_constant_channel_normalizes_as_in_training():
    """Finding 2: training mapped a constant channel to 0 (nan_to_num); planning divided 0 by 0."""
    a = constant_wrist_actions()
    with tempfile.TemporaryDirectory() as tmp:
        p = make_planner(make_run(tmp, a))
        art = torch.load(Path(tmp) / "normalizer.pt", weights_only=False)["stats"]["action"]
    train = ZScore(torch.tensor(art["mean"]), torch.tensor(art["std"]))(torch.tensor(a)).numpy()
    plan = p.normalize_action(a)
    assert np.isfinite(plan).all()
    np.testing.assert_allclose(plan, train, atol=1e-5)


def test_solver_scores_only_executable_actions():
    """Finding 27: the normalized solver bounds are the env's MAX_ACTION, so the plan needs no clipping."""
    a = np.random.default_rng(2).normal(scale=0.5, size=(50, 6))  # wide data: the 2-sigma prior exceeds MAX_ACTION
    with tempfile.TemporaryDirectory() as tmp:
        p = make_planner(make_run(tmp, a))
    lo, hi = p.solver.action_bounds
    np.testing.assert_allclose(lo * p.action_std + p.action_mean, -MAX_ACTION, atol=1e-6)
    np.testing.assert_allclose(hi * p.action_std + p.action_mean, MAX_ACTION, atol=1e-6)
    plan = p.solver.solve({"x": torch.zeros(1, 1)}, planning_horizon=5)["actions"][0].numpy()
    raw = plan * p.action_std + p.action_mean
    assert np.all(np.abs(raw) <= MAX_ACTION + 1e-5)
    assert np.all(raw[0] > 0.5 * MAX_ACTION)  # the first step: pushed toward the bound it prefers


def test_value_head_bound_to_its_checkpoint():
    """Finding 13: value.pt records its encoder checkpoint's sha256; the planner refuses another one."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = make_run(tmp, constant_wrist_actions())
        vpath = Path(tmp) / "value.pt"
        torch.save({"cfg": {}, "state_dict": {}, "encoder_ckpt_sha256": planner.sha256_file(ckpt)}, vpath)
        planner.check_value_binding(vpath, ckpt)
        ckpt.write_bytes(b"other weights")
        try:
            planner.check_value_binding(vpath, ckpt)
            raise AssertionError("a value head of another checkpoint was accepted")
        except ValueError:
            pass
        torch.save({"cfg": {}, "state_dict": {}}, vpath)  # made before the binding: a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            planner.check_value_binding(vpath, ckpt)
        assert w


def test_nan_costs_do_not_rank_perfect():
    """Finding 2: with NaN costs every comparison was false and the expert's rank 0 (perfect)."""
    assert offline.rank_of_expert(np.array([np.nan, 1.0, 2.0])) == 1.0
    assert offline.rank_of_expert(np.array([1.0, np.nan, 2.0, 0.5])) == 2 / 3
    assert offline.rank_of_expert(np.array([1.0, 2.0, 3.0])) == 0.0


def test_cem_rejects_all_nan_costs():
    class NaNCost(CostModel):
        def get_cost(self, info, actions):
            return torch.full(actions.shape[:2], float("nan"))
    s = CEMSolver(NaNCost(), n_steps=2, num_samples=8, topk=2, device="cpu")
    import gymnasium
    from stable_worldmodel.policy import PlanConfig
    s.configure(action_space=gymnasium.spaces.Box(-1, 1, shape=(1, 6)), n_envs=1,
                config=PlanConfig(horizon=3, receding_horizon=1))
    try:
        s.solve({"x": torch.zeros(1, 1)}, planning_horizon=3)
        raise AssertionError("NaN costs gave a plan")
    except FloatingPointError:
        pass


def test_offline_plan_records_case_history():
    """Finding 11: plan direction starts each case from a reset with its preceding real steps."""
    class Rec:
        def __init__(self):
            self.past = deque(maxlen=3)

        def reset(self):
            self.past.clear()

        def record(self, obs, action):
            self.past.append((obs, action))

    f = {"pixels": np.arange(20), "proprio": np.arange(20) * 10, "action": np.arange(20) * 100}
    p = Rec()
    p.record({}, -1)  # left over from another case
    offline.record_history(p, f, 10, 4)  # frame 14 of the episode at offset 10
    assert [int(o["pixels"]) for o, _ in p.past] == [11, 12, 13]
    assert [int(a) for _, a in p.past] == [1100, 1200, 1300]


def test_offline_case_ids():
    """Finding 12: the report names each case ([episode, frame]) in the order of the per-case arrays."""
    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.h5"
        with h5py.File(path, "w") as h:
            h["ep_len"], h["ep_offset"] = np.array([30, 40]), np.array([0, 30])
        f, picks, ids = offline.load(path, np.random.default_rng(0), 20, 15)
        f.close()
        fp = offline.file_fingerprint(path)
    assert len(ids) == len(picks) == 20
    assert all([0, 30][e] == off and t == t2 for (e, t), (off, t2) in zip(ids, picks))
    assert fp["size"] > 0 and len(fp["head_tail_sha256"]) == 64


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

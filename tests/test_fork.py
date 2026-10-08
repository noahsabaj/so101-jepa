"""Tests of our H-JEPA fork (third_party/H-JEPA): hierarchical planning with history, CEM, SIGReg,
normalization, artifact saves, dataset pickling. CPU only, tiny stand-in modules, seconds each.

    .venv/Scripts/python.exe -m pytest tests/test_fork.py     (or: python tests/test_fork.py)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/H-JEPA/h_jepa"))

from eval_config_utils import _build_policy_plan_config  # noqa: E402
from hierarchical_solver import HierarchicalSolver, LevelCostModel  # noqa: E402
from models.jepa import JEPA, ProjectedEncoder  # noqa: E402
from stable_worldmodel.solver import CEMSolver, GradientSolver  # noqa: E402

D, A, MACRO = 16, 6, 8


class PixelStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3 * 4 * 4, D)

    def forward(self, x, interpolate_pos_encoding=True):
        return self.lin(x.flatten(1))


class LatentStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(D, D)

    def forward(self, x):
        return self.lin(x)


class Predictor(torch.nn.Module):
    """Next latents from latents and actions of the same length (as the causal transformer)."""

    def __init__(self, action_dim):
        super().__init__()
        self.lin = torch.nn.Linear(action_dim, D)

    def forward(self, emb, act):
        assert emb.shape[1] == act.shape[1], f"{emb.shape[1]} latents for {act.shape[1]} actions"
        return emb + torch.tanh(self.lin(act))


class MacroEncoder(torch.nn.Module):
    """Masked mean of a window's raw actions, mapped to an 8-d macro-action."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(A, MACRO)

    def forward(self, x, mask=None):
        w = (~mask).float().unsqueeze(-1) if mask is not None else torch.ones_like(x[..., :1])
        return self.lin((x * w).sum(-2) / w.sum(-2).clamp_min(1))


def two_level_solver(solver_cls=GradientSolver):
    torch.manual_seed(0)
    l1 = JEPA(ProjectedEncoder(PixelStub()), Predictor(A), torch.nn.Identity(), level=1)
    l2 = JEPA(ProjectedEncoder(LatentStub()), Predictor(MACRO), torch.nn.Identity(), MacroEncoder(), level=2,
              temporal_stride=5, action_queue_size=16)
    l2._buffers["latent_action_queue"] = torch.randn(16, MACRO)
    l1.plan_history = 4
    models = [l1.eval(), l2.eval()]
    kw = dict(n_steps=2, num_samples=4, device="cpu") if solver_cls is GradientSolver else \
        dict(n_steps=2, num_samples=8, topk=2, device="cpu")
    solvers = {lv: solver_cls(LevelCostModel(m, models[lv:]), **kw) for lv, m in enumerate(models, 1)}
    hs = HierarchicalSolver(solvers, models)
    cfg = OmegaConf.load(ROOT / "hjepa/config/eval/so101_l2.yaml")
    cfg.policy = "none"
    import gymnasium
    hs.configure(action_space=gymnasium.spaces.Box(-1, 1, shape=(1, A)), n_envs=1, config=_build_policy_plan_config(cfg))
    return hs


def info(n_hist):
    """As hjepa/planner.py's _info: n_hist observed frames before the current one."""
    out = {"pixels": torch.randn(1, n_hist + 1, 3, 4, 4), "goal": torch.randn(1, 1, 3, 4, 4),
           "action": torch.zeros(1, n_hist + 1, A)}
    if n_hist:
        out["history_action"] = torch.randn(1, n_hist, A)
    return out


def test_hierarchy_replans_with_history():
    """Finding 1: the second upper-level plan (raw 6-d history beside 8-d macro-actions) crashed."""
    for solver_cls in (GradientSolver, CEMSolver):
        hs = two_level_solver(solver_cls)
        for steps, n_hist in ((0, 0), (2, 2), (4, 3), (6, 3)):  # first plan, then replans with history
            actions = hs.solve(info(n_hist), steps_taken=steps, eval_budget=50)["actions"]
            assert actions.shape[-1] == A and torch.isfinite(actions).all()


def test_level1_context_matches_history():
    """Level 1 rolls out from every observed frame, and its cost starts at the current frame."""
    hs = two_level_solver()
    emb = hs._encode_all_levels(info(3), key="pixels")
    assert emb["embed_1"].shape[1] == 4 and emb["embed_2"].shape[1] == 1
    l1 = hs.level_models[0]
    i = {"pixels": torch.zeros(1, 2, 4, 3, 4, 4), "embed_0": emb["embed_1"][:, None].expand(1, 2, 4, D),
         "history_action": torch.zeros(1, 2, 3, A)}
    out = l1.rollout(i, torch.zeros(1, 2, 5, A))
    assert out["predicted_embed_0"].shape[2] == 4 + 5 and out["context_frames"] == 4


def test_cem_one_elite_stays_finite():
    """Finding 26: with topk=1 the sample std of one elite was NaN, and so the plan."""
    from stable_worldmodel.policy import PlanConfig
    import gymnasium

    class Cost(torch.nn.Module):
        level = 1

        def get_cost(self, info, a):
            return (a ** 2).sum((-1, -2))

        def rollout(self, info, a):
            return {}

    s = CEMSolver(Cost(), n_steps=3, num_samples=8, topk=1, device="cpu")
    s.configure(action_space=gymnasium.spaces.Box(-1, 1, shape=(1, A)), n_envs=1,
                config=PlanConfig(horizon=3, receding_horizon=1))
    assert torch.isfinite(s.solve({"x": torch.zeros(1, 1)}, planning_horizon=3)["actions"]).all()


def test_seed_before_model_build():
    """Finding 3: main_hjepa seeded only at manager() time, after the model was built."""
    src = (ROOT / "third_party/H-JEPA/h_jepa/main_hjepa.py").read_text()
    run = src[src.index("def run(cfg)"):]
    seed = run.index("seed_everything(")
    assert seed < run.index("build_hdf5_dataset(") and seed < run.index("create_world_model(")


def _artifact(scale):
    return {"format": "lejepa_training_normalizer_v1",
            "stats": {"action": {"mean": np.zeros((1, A), np.float32), "std": np.full((1, A), scale, np.float32),
                                 "count": 10}}}


def test_resume_refuses_changed_data_or_normalizer():
    """Finding 4: a resume accepted a rebuilt dataset and overwrote the run's normalizer first."""
    import json
    import tempfile
    from data import RUN_IDENTITY, check_resume_identity, run_identity, save_normalizer_artifact

    class DS:
        def __init__(self, p):
            self.h5_path = p

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        h5 = tmp / "so101_train.h5"
        h5.write_bytes(b"a" * 1000)
        ident = run_identity([DS(h5)], _artifact(0.01))
        save_normalizer_artifact(_artifact(0.01), tmp)
        (tmp / RUN_IDENTITY).write_text(json.dumps(ident))
        check_resume_identity(tmp, run_identity([DS(h5)], _artifact(0.01)))  # the same run: accepted
        for changed in (run_identity([DS(h5)], _artifact(0.1)), {**ident, "data": {"so101_train.h5": {}}}):
            try:
                check_resume_identity(tmp, changed)
                raise AssertionError("a resume with other data was accepted")
            except RuntimeError:
                pass
        (tmp / RUN_IDENTITY).unlink()  # a run from before identity records: the saved normalizer still decides
        try:
            check_resume_identity(tmp, run_identity([DS(h5)], _artifact(0.1)))
            raise AssertionError("a resume with another normalizer was accepted")
        except RuntimeError:
            pass


def test_atomic_save_keeps_the_old_file():
    """Finding 19: a failed save left a partial file at the final name."""
    import tempfile
    from data import save_atomic

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m_object.ckpt"
        save_atomic({"w": 1}, path)
        try:
            save_atomic({"w": lambda: 0}, path)  # unpicklable: torch.save fails midway
        except Exception:
            pass
        assert torch.load(path, weights_only=False) == {"w": 1}


def test_normalizer_and_dataset_pickle():
    """Finding 31: the open h5py handle (and closure transforms) did not pickle into spawned workers."""
    import tempfile
    import h5py
    from data import ColumnTransform, ZScore
    from stable_worldmodel.data.dataset import HDF5Dataset

    with tempfile.TemporaryDirectory() as tmp:
        with h5py.File(Path(tmp) / "d.h5", "w") as h:
            h["ep_len"], h["ep_offset"] = np.array([5]), np.array([0])
            h["action"], h["proprio"] = np.zeros((5, A), np.float32), np.ones((5, A), np.float32)
        ds = HDF5Dataset("d", keys_to_load=["action", "proprio"], cache_dir=tmp, level1={"frameskip": 1, "num_steps": 2})
        ds.get_col_data("proprio")  # opens the file in this process
        ds.transform = ColumnTransform(ZScore(torch.zeros(A), torch.zeros(A)), "action", "action")
        clone = pickle.loads(pickle.dumps(ds))
        assert clone[0]["proprio_level1"].shape == (2, A) and torch.isfinite(clone[0]["action_level1"]).all()
        clone.close()
        ds.close()


def test_sigreg_statistic_in_fp32():
    """Finding 32: the final integral of SIGReg ran in bf16 under autocast."""
    from loss import SIGReg

    torch.manual_seed(0)
    proj = torch.randn(2, 64, 32)
    reg = SIGReg(num_proj=64)
    torch.manual_seed(1)
    ref = reg(proj)
    torch.manual_seed(1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = reg(proj)
    assert mixed.dtype == torch.float32 and torch.allclose(mixed, ref, rtol=1e-5)


def _launch():
    sys.path.insert(0, str(ROOT / "third_party/H-JEPA/h_jepa/scripts/slurm"))
    import launch
    return launch


def test_launcher_resume_and_submit_checks():
    """Findings 24, 33: a failed sbatch exited 0; resume advertised changes that training refuses."""
    import tempfile
    launch = _launch()
    try:
        launch.check_submitted({"a": "123", "b": None})
        raise AssertionError("a failed submission passed")
    except SystemExit as e:
        assert "1 of 2" in str(e)
    launch.check_submitted({"a": "123"})
    with tempfile.TemporaryDirectory() as tmp:
        saved = OmegaConf.create({"num_workers": 10, "loader": {"num_workers": "${num_workers}"}, "seed": 42,
                                  "x": "${seed}", "trainer": {"devices": 1, "max_epochs": 3}})
        OmegaConf.save(saved, Path(tmp) / "config.yaml")
        same = launch.compose_cfg(Path(tmp), "config", ["num_workers=4", "trainer.devices=1"])
        assert launch.comparable(same) == launch.comparable(OmegaConf.load(Path(tmp) / "config.yaml"))
        longer = launch.compose_cfg(Path(tmp), "config", ["trainer.max_epochs=100", "trainer.devices=1"])
        assert launch.comparable(longer) != launch.comparable(saved)


def test_launcher_refuses_relabelled_eval():
    """Finding 23: a done eval dir was skipped (or relabelled) whatever the new eval's settings."""
    import tempfile
    from planning_eval import claim_results_dir
    launch = _launch()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        args = launch.eval_args("so101_l2", 42, "/r/m_epoch_3_object.ckpt", ["solver.n_steps=5"])
        assert launch.same_eval(out, args) is None
        (out / "launch_eval.json").write_text(__import__("json").dumps(args))
        assert launch.same_eval(out, launch.eval_args("so101_l2", "42", "/r/m_epoch_3_object.ckpt", ["solver.n_steps=5"]))
        assert launch.same_eval(out, launch.eval_args("so101_l2", 43, "/r/m_epoch_3_object.ckpt", [])) is False
        cfg = OmegaConf.create({"policy": "none", "seed": 1, "start_index": 0})
        claim_results_dir(out, cfg, ignore=("start_index",))
        claim_results_dir(out, OmegaConf.merge(cfg, {"start_index": 5}), ignore=("start_index",))  # another clip job
        try:
            claim_results_dir(out, OmegaConf.merge(cfg, {"seed": 2}))
            raise AssertionError("an eval with another seed reused the dir")
        except RuntimeError:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

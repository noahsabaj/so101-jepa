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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

"""H-JEPA planning for the SO-101: load a trained model and plan from an observation to a goal.

The solver code is H-JEPA's: multi-start gradient descent per level (hierarchical_solver.py), or the
cross-entropy method (stable_worldmodel/solver/cem.py, added in our fork).
This module only converts our observations (uint8 64x128 pixels, 6 joint angles) the way the
training pipeline did, and the planned actions back to raw joint-target changes (rad).
The cost is the latent distance to the goal, or, with `cost: value` in the eval config, the learned
goal-reaching value of hjepa/value.py (value.pt beside the checkpoint).
A time-step model (PLAN.md A21, trained with level1 strides) plans with `strides: [...]` and
`max_stride: K` in the eval config: one plan per stride k (each step k raw actions, padded to K plus
k / K, as in training), the plan with the lowest cost wins, and the robot replans after the receding
horizon in raw steps (plan far, act short).
"""

import sys
from pathlib import Path

import gymnasium
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/H-JEPA/h_jepa"))

from data import IMAGENET_STATS, load_normalizer_artifact  # noqa: E402
from eval_config_utils import _build_policy_plan_config  # noqa: E402
from planning_eval import build_solver, load_model  # noqa: E402

MEAN = torch.tensor(IMAGENET_STATS["mean"]).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STATS["std"]).view(3, 1, 1)


class StrideEmbed(torch.nn.Module):
    """Planner actions of stride k (k raw actions per step) to the model's action: zeros up to
    max_stride raw actions, then k / max_stride."""

    def __init__(self, k, kmax, dim=6):
        super().__init__()
        self.k, self.kmax, self.dim = k, kmax, dim

    def forward(self, a):
        pad = a.new_zeros(*a.shape[:-1], self.dim * (self.kmax - self.k))
        return torch.cat([a, pad, a.new_full((*a.shape[:-1], 1), self.k / self.kmax)], -1)


class Planner:
    def __init__(self, ckpt, eval_config, seed=1234, overrides=()):
        cfg = OmegaConf.load(ROOT / "hjepa/config/eval" / f"{eval_config}.yaml")
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"policy={ckpt}", f"seed={seed}", *overrides]))
        self.cfg = cfg
        self.model = load_model(cfg)
        if cfg.get("cost", "latent") == "value":
            from value import load_value
            level1 = self.model.get_level(1) if hasattr(self.model, "get_level") else self.model
            level1.value_fn = load_value(Path(ckpt).parent / "value.pt")
        stats = load_normalizer_artifact(Path(ckpt).parent / "normalizer.pt")["stats"]
        self.proprio_mean, self.proprio_std = (torch.tensor(stats["proprio"][k]).float().view(-1) for k in ("mean", "std"))
        self.action_mean, self.action_std = (np.asarray(stats["action"][k]).reshape(-1) for k in ("mean", "std"))
        self.receding = int(cfg.get("plan_config", cfg.get("hierarchical_plan_config")).receding_horizon)
        self.strides, self.kmax = list(cfg.get("strides", [1])), int(cfg.get("max_stride", 1))
        self.solvers = {}
        for k in self.strides:
            solver = build_solver(cfg, self.model)
            solver.configure(action_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(1, 6 * k)), n_envs=1,
                             config=_build_policy_plan_config(cfg))
            self.solvers[k] = solver
        self.solver = self.solvers[self.strides[0]]
        self.last_stride = self.strides[0]

    def _pixels(self, pixels):
        x = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
        return ((x - MEAN) / STD)[None, None].cuda()

    def _proprio(self, proprio):
        return ((torch.as_tensor(proprio).float() - self.proprio_mean) / self.proprio_std)[None, None].cuda()

    def seed(self, seed):
        for solver in self.solvers.values():
            for s in getattr(solver, "level_solvers", {1: solver}).values():
                s.torch_gen.manual_seed(seed)

    @torch.no_grad()
    def encode(self, obs):
        """Level-1 latent of an observation (for the offline tests)."""
        level1 = self.model.get_level(1) if hasattr(self.model, "get_level") else self.model
        out = level1.encode({"pixels": self._pixels(obs["pixels"]), "proprio": self._proprio(obs["proprio"])})
        return out["embed_0"][0, 0]

    def plan(self, obs, goal, steps_taken=0, eval_budget=100):
        """Planned actions (raw joint-target changes, rad), shape (receding horizon, 6)."""
        if self.kmax == 1:
            actions = self.solver(self._info(obs, goal, 1), steps_taken=steps_taken, eval_budget=eval_budget)["actions"][0]
            actions = actions[: self.receding].cpu().numpy().reshape(-1, 6)
            return actions * self.action_std + self.action_mean
        level1 = self.model.get_level(1) if hasattr(self.model, "get_level") else self.model
        best = None
        for k, solver in self.solvers.items():  # one plan per stride; the lowest model cost wins
            level1.action_embed = StrideEmbed(k, self.kmax)
            plan = solver(self._info(obs, goal, k), steps_taken=steps_taken, eval_budget=eval_budget)["actions"]
            info = solver._expand_info_dict_for_samples(self._info(obs, goal, k), start_idx=0, end_idx=1, num_samples=1)
            with torch.no_grad():
                cost = float(self.model.get_cost(info, plan.to("cuda")[:, None]).min())
            if best is None or cost < best[0]:
                best = (cost, k, plan[0])
        _, self.last_stride, plan = best
        actions = plan.cpu().numpy().reshape(-1, 6)[: self.receding]  # replan after the receding horizon (raw steps)
        return actions * self.action_std + self.action_mean

    def _info(self, obs, goal, k):
        return {
            "pixels": self._pixels(obs["pixels"]), "proprio": self._proprio(obs["proprio"]),
            "goal": self._pixels(goal["pixels"]), "goal_proprio": self._proprio(goal["proprio"]),
            "action": torch.zeros(1, 1, 6 * k, device="cuda"),
        }

    def normalize_action(self, action):
        return (np.asarray(action) - self.action_mean) / self.action_std

"""H-JEPA planning for the SO-101: load a trained model and plan from an observation to a goal.

The solver code is H-JEPA's: multi-start gradient descent per level (hierarchical_solver.py), or the
cross-entropy method (stable_worldmodel/solver/cem.py, added in our fork).
This module only converts our observations (uint8 64x128 pixels, 6 joint angles) the way the
training pipeline did, and the planned actions back to raw joint-target changes (rad).
The cost is the latent distance to the goal, or, with `cost: value` in the eval config, the learned
goal-reaching value of hjepa/value.py (value.pt beside the checkpoint).
History: the predictor sees up to the training history (level1.wm.history_size; `plan_history: N`
in the eval config overrides it) of observed frames and the actions run between them. Call reset()
at the start of an episode and record(obs, action) for every action the robot runs.
A time-step model (PLAN.md A21, trained with level1 strides) plans with `strides: [...]`,
`max_stride: K` and `plan_steps: P` in the eval config: one plan per stride k, each P / k steps of k
raw actions (padded to K plus k / K, as in training), so every plan ends P raw steps ahead and their
costs compare; the plan with the lowest cost wins, and the robot replans after the receding horizon
in raw steps (plan far, act short).
Speed (eval config or overrides): `precision: bf16` (as in training) or `fp16` runs the model under
autocast; `compile: true` (or a torch.compile mode, e.g. reduce-overhead for CUDA graphs) compiles
the level-1 predictor.
Level 1 plans only actions within the env's MAX_ACTION (sim/env.py), so the model scores what the
robot runs. A value head must record the sha256 of the checkpoint it was trained on (hjepa/value.py).
"""

import contextlib
import hashlib
import sys
import warnings
from collections import deque
from pathlib import Path

import gymnasium
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/H-JEPA/h_jepa"))
sys.path.append(str(ROOT / "sim"))  # last: sim's modules shadow nothing

from data import IMAGENET_STATS, load_normalizer_artifact, safe_std  # noqa: E402
from env import MAX_ACTION  # noqa: E402
from eval_config_utils import _build_policy_plan_config  # noqa: E402
from planning_eval import build_solver, load_model  # noqa: E402

MEAN = torch.tensor(IMAGENET_STATS["mean"]).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STATS["std"]).view(3, 1, 1)


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def check_value_binding(value_path, ckpt):
    """A value head is trained on one encoder checkpoint's latents: refuse it with any other one."""
    blob = torch.load(value_path, map_location="cpu", weights_only=False)
    bound = blob.get("encoder_ckpt_sha256")
    if bound is None:
        warnings.warn(f"{value_path} does not record its encoder checkpoint (made before the binding); "
                      f"cannot check that it was trained on {ckpt}")
    elif bound != sha256_file(ckpt):
        raise ValueError(f"{value_path} was trained on another encoder checkpoint (sha256 {bound[:12]}...) "
                         f"than {ckpt}: train a value head for this checkpoint (hjepa/value.py)")


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
        self.model = load_model(cfg).requires_grad_(False)  # planning differentiates the actions only
        if cfg.get("cost", "latent") == "value":
            from value import load_value
            level1 = self.model.get_level(1) if hasattr(self.model, "get_level") else self.model
            check_value_binding(Path(ckpt).parent / "value.pt", ckpt)
            level1.value_fn = load_value(Path(ckpt).parent / "value.pt")
        # as training normalizes (h_jepa/data.py): a constant column (std 0) is centred, not divided by 0
        stats = load_normalizer_artifact(Path(ckpt).parent / "normalizer.pt")["stats"]
        self.proprio_mean, self.proprio_std = (torch.tensor(stats["proprio"][k]).float().view(-1) for k in ("mean", "std"))
        self.proprio_std = safe_std(self.proprio_std)
        self.action_mean, self.action_std = (np.asarray(stats["action"][k]).reshape(-1) for k in ("mean", "std"))
        self.action_std = safe_std(self.action_std)
        self.receding = int(cfg.get("plan_config", cfg.get("hierarchical_plan_config")).receding_horizon)
        self.strides, self.kmax = list(cfg.get("strides", [1])), int(cfg.get("max_stride", 1))
        if self.kmax > 1:
            self.plan_steps = int(cfg.plan_steps)  # every stride plans to the same time
            bad = [k for k in self.strides if self.plan_steps % k]
            if bad:
                raise ValueError(f"plan_steps={self.plan_steps} is not a multiple of strides {bad}")
        train_cfg = OmegaConf.load(Path(ckpt).parent / "config.yaml")
        self.history = int(cfg.get("plan_history", train_cfg.level1.wm.history_size))
        level1 = self.model.get_level(1) if hasattr(self.model, "get_level") else self.model
        level1.plan_history = self.history
        for lv in range(2, getattr(self.model, "num_levels", 1) + 1):  # upper levels: their training history
            self.model.get_level(lv).plan_history = int(train_cfg[f"level{lv}"].wm.history_size)
        self.past = deque(maxlen=(self.history - 1) * self.kmax)  # (observation, raw action run from it)
        # the env runs at most MAX_ACTION (sim/env.py): level 1 scores only actions it will execute
        lo, hi = self.normalize_action(-MAX_ACTION), self.normalize_action(MAX_ACTION)
        self.solvers = {}
        for k in self.strides:
            solver = build_solver(cfg, self.model)
            solver.configure(action_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(1, 6 * k)), n_envs=1,
                             config=_build_policy_plan_config(cfg))
            s1 = getattr(solver, "level_solvers", {1: solver})[1]
            s1.action_bounds = (np.tile(lo, s1.action_dim // 6), np.tile(hi, s1.action_dim // 6))
            self.solvers[k] = solver
        self.solver = self.solvers[self.strides[0]]
        self.last_stride = self.strides[0]
        dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.get("precision", "fp32")]
        self.autocast = (lambda: torch.autocast("cuda", dtype=dtype)) if dtype else contextlib.nullcontext
        mode = cfg.get("compile", False)
        if mode:
            level1 = self.model.get_level(1) if hasattr(self.model, "get_level") else self.model
            level1.predictor = torch.compile(level1.predictor, mode=None if mode is True else mode)

    def _pixels(self, pixels):
        x = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
        return ((x - MEAN) / STD)[None, None].cuda()

    def _proprio(self, proprio):
        return ((torch.as_tensor(proprio).float() - self.proprio_mean) / self.proprio_std)[None, None].cuda()

    def reset(self):
        """Start of an episode: no observed history."""
        self.past.clear()

    def record(self, obs, action):
        """The robot ran raw `action` (rad) from `obs`."""
        self.past.append((obs, np.asarray(action, np.float32)))

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
        with self.autocast():
            actions = self._plan(obs, goal, steps_taken, eval_budget)
        if not np.isfinite(actions).all():
            raise FloatingPointError(f"the planner returned non-finite actions: {actions.tolist()}")
        return actions

    def _plan(self, obs, goal, steps_taken, eval_budget):
        if self.kmax == 1:
            actions = self.solver(self._info(obs, goal, 1), steps_taken=steps_taken, eval_budget=eval_budget)["actions"][0]
            actions = actions[: self.receding].float().cpu().numpy().reshape(-1, 6)
            return actions * self.action_std + self.action_mean
        level1 = self.model.get_level(1) if hasattr(self.model, "get_level") else self.model
        best = None
        for k, solver in self.solvers.items():  # one plan per stride; the lowest model cost wins
            level1.action_embed = StrideEmbed(k, self.kmax)
            plan = solver(self._info(obs, goal, k), planning_horizon=self.plan_steps // k)["actions"]
            info = solver._expand_info_dict_for_samples(self._info(obs, goal, k), start_idx=0, end_idx=1, num_samples=1)
            with torch.no_grad():
                cost = float(self.model.get_cost(info, plan.to("cuda")[:, None]).min())
            if not np.isfinite(cost):
                raise FloatingPointError(f"non-finite model cost {cost} for the stride-{k} plan")
            if best is None or cost < best[0]:
                best = (cost, k, plan[0])
        _, self.last_stride, plan = best
        actions = plan.float().cpu().numpy().reshape(-1, 6)[: self.receding]  # replan after the receding horizon (raw steps)
        return actions * self.action_std + self.action_mean

    def _info(self, obs, goal, k):
        """Observed frames k raw steps apart (as in training at stride k), oldest first, the actions run
        between them (k normalized raw actions each), the current observation and the goal."""
        n, past = min(self.history - 1, len(self.past) // k), list(self.past)
        frames = [past[len(past) - j * k][0] for j in range(n, 0, -1)] + [obs]
        info = {
            "pixels": torch.cat([self._pixels(o["pixels"]) for o in frames], 1),
            "proprio": torch.cat([self._proprio(o["proprio"]) for o in frames], 1),
            "goal": self._pixels(goal["pixels"]), "goal_proprio": self._proprio(goal["proprio"]),
            "action": torch.zeros(1, n + 1, 6 * k, device="cuda"),
        }
        if n:
            acts = [np.concatenate([self.normalize_action(past[len(past) - j * k + i][1]) for i in range(k)])
                    for j in range(n, 0, -1)]
            info["history_action"] = torch.as_tensor(np.stack(acts), dtype=torch.float32, device="cuda")[None]
        return info

    def normalize_action(self, action):
        return (np.asarray(action) - self.action_mean) / self.action_std  # action_std: safe_std, as in training


def make_planner(ckpt, eval_config, seed=1234, overrides=()):
    """The planner an eval config names: `type: lewam` (hjepa/lewam_planner.py, PLAN.md A25), else Planner."""
    if OmegaConf.load(ROOT / "hjepa/config/eval" / f"{eval_config}.yaml").get("type") == "lewam":
        from lewam_planner import LeWAMAdapter
        return LeWAMAdapter(ckpt, eval_config, seed=seed, overrides=overrides)
    return Planner(ckpt, eval_config, seed=seed, overrides=overrides)

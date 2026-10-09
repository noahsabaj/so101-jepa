"""Planning with a LeWAM model (PLAN.md A25) behind the interface of hjepa/planner.py's Planner, so
sim/closed_loop.py, sim/self_play.py and hjepa/probe.py run it unchanged (through make_planner).

The planning is LeWAM's own (third_party/lewam/lewam/eval/planners.py): the goal-conditioned action head
proposes num_proposals action chunks, each is imagined through the latent dynamics to the goal step, and
plan_mode grad refines them with Adam through the dynamics (cost: the squared distance of the imagined
latent to the goal latent, both views), then the cheapest is run for exec_actions steps (default: one
chunk of frameskip steps) and the robot replans. This module only keeps the latent history of the observed
frames (history_len frames, history_stride steps apart) and converts observations and actions.
The goal horizon the head is told (in chunks of frameskip steps): horizon_frac of the steps left in the
budget, at most H_max; LeWAM's eval gives the true goal offset with a budget twice as long, so 0.5.

Eval config (hjepa/config/eval/, `type: lewam`): plan_mode (best_of_k | grad | cem | steer),
num_proposals, grad_steps, grad_lr, grad_tr, grad_all_k, exec_actions, horizon_frac.
"""

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from lewam_common import ROOT, split_views
from lewam.eval.planners import LeWAMPlanner  # noqa: E402
from lewam.models.lewam import build_model  # noqa: E402
from lewam.train.utils import _IMG_MEAN, _IMG_STD  # noqa: E402

PLAN_KEYS = ("plan_mode", "num_proposals", "rollout_steps", "plan_goal_time", "grad_steps", "grad_lr", "grad_clip",
             "grad_tr", "grad_action_clip", "grad_all_k", "cem_iters", "cem_elites", "cem_std", "cem_init",
             "steer_steps", "steer_lr", "steer_max_bias", "steer_k")


class _Planner(LeWAMPlanner):
    def _goal_latents(self, info_dict, replan):
        return info_dict["z_goal"]


class LeWAMAdapter:
    def __init__(self, ckpt, eval_config, seed=1234, overrides=()):
        cfg = OmegaConf.load(ROOT / "hjepa/config/eval" / f"{eval_config}.yaml")
        cfg = OmegaConf.to_container(OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides))), resolve=True)
        self.mcfg = json.loads((Path(ckpt).parent / "lewam_config.json").read_text())
        self.model = build_model(self.mcfg).cuda()
        self.model.load_state_dict(torch.load(ckpt, map_location="cuda"))
        self.model.eval().requires_grad_(False)  # planning differentiates the actions only
        self.fs, self.H_max = int(self.mcfg["fs"]), int(self.mcfg["H_max"])
        self.hist_len, self.hist_stride = int(self.mcfg["history_len"]), int(self.mcfg["history_stride"])
        self.horizon_frac = float(cfg.get("horizon_frac", 0.5))
        self.action_mean = np.asarray(self.mcfg["action_mean"], np.float32)
        self.action_std = np.maximum(np.asarray(self.mcfg["action_std"], np.float32), 1e-6)
        plan = {k: cfg[k] for k in PLAN_KEYS if k in cfg}
        plan.setdefault("rollout_steps", self.H_max)
        self.planner = _Planner(self.model, self.mcfg, self.fs, 6, h0=float(self.H_max), flow_seed=seed,
                                exec_actions_per_plan=int(cfg.get("exec_actions", 0)), **plan)
        self.planner._steps_left = np.ones(1)
        self.mean, self.std = _IMG_MEAN.cuda(), _IMG_STD.cuda()
        self.reset()

    # -- the Planner interface --
    def reset(self):
        """Start of an episode: no observed history."""
        self.frames, self.latents = [], {}

    def record(self, obs, action):
        """The robot ran raw `action` (rad) from `obs`."""
        self.frames.append(obs["pixels"])

    def seed(self, seed):
        self.planner._flow_seed, self.planner._flow_rng = int(seed), None

    @torch.no_grad()
    def encode_views(self, pixels):
        """(B, 64, 128, 3) uint8 frames -> (B, views, z) latents."""
        x = torch.as_tensor(np.stack(split_views(np.asarray(pixels)), 1)).cuda()  # (B, 2, 64, 64, 3)
        x = (x.permute(0, 1, 4, 2, 3).float().flatten(0, 1) / 255.0 - self.mean) / self.std
        return self.model.encode(x).view(len(pixels), 2, -1)

    def encode(self, obs):
        return self.encode_views(obs["pixels"][None])[0].flatten()

    def plan(self, obs, goal, steps_taken=0, eval_budget=100):
        """Planned actions (raw joint-target changes, rad), shape (exec_actions, 6)."""
        t = len(self.frames)  # the current frame's step
        rows = [s for s in (t - k * self.hist_stride for k in range(self.hist_len - 1, -1, -1)) if s >= 0]
        for s in rows:
            if s not in self.latents:
                self.latents[s] = self.encode_views([obs["pixels"] if s == t else self.frames[s]])[0]
        self.latents = {s: z for s, z in self.latents.items() if s >= t - (self.hist_len - 1) * self.hist_stride}
        history = torch.zeros(1, 2, self.hist_len, self.model.z_dim, device="cuda")
        pad = torch.ones(1, self.hist_len, dtype=torch.bool, device="cuda")
        history[0, :, self.hist_len - len(rows):] = torch.stack([self.latents[s] for s in rows], 1)
        pad[0, self.hist_len - len(rows):] = False
        z_goal = self.encode_views(goal["pixels"][None])[:, self.planner.goal_view_indices]
        left = (eval_budget - steps_taken) * self.horizon_frac / self.fs
        self.planner._steps_left = np.array([float(np.clip(left, 1.0, self.H_max))])
        with torch.no_grad():
            actions = self.planner._propose({"z_goal": z_goal}, [0], history, pad)[0]
        actions = actions.float().cpu().numpy() * self.action_std + self.action_mean
        if not np.isfinite(actions).all():
            raise FloatingPointError(f"the planner returned non-finite actions: {actions.tolist()}")
        return actions

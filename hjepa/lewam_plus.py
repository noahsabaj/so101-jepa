"""LeWAM plus (PLAN.md A27 step 3): the test-bed changes to LeWAM, each off by default, trained with LeWAM's own
trainer (third_party/lewam/scripts/train_lewam.py, through lewam_common.trainer_namespace) and its episode split.

    torchrun --nproc_per_node W hjepa/lewam_plus.py CONFIG RUN DATA.h5 [key=value ...]

CONFIG is a config in hjepa/config/lewam/ with a `plus:` section (plus-pusht, plus-cube); RUN names the run
(data/ckpts/lewam/RUN/seed42). The changes, each tested alone against the same config with all of them off:
  A17 plus.proprio_keys: a proprio token (z-scored h5 columns at the start frame, plus.proprio_dims of them),
      seen by the state slots, clean actions and action slots; dropped per sample with plus.p_drop_proprio
  A26 plus.hier: a flow head for the latent hier_sub chunks ahead toward a goal up to hier_far chunks away
      (eval: LEWAM_HIER=1 conditions the policy or planner on its subgoal while the goal is farther than hier_sub)
  A28 plus.distill: the best of distill_k imagined policy rollouts (the eval best_of_k planner) as action targets,
      on the samples where it beats the data's own actions rolled out the same way
  A29 plus.roll_k: the state loss also on chunks 2..roll_k, rolled out on the model's own latents
  A30 train.p_drop_goal (LeWAM's own) at training, LEWAM_CFG=w (classifier-free guidance) at eval
  A31 model.state_head=flow (LeWAM's own)
  A32 plus.shortcut: shortcut models (Frans et al. 2024) for 1..shortcut_m/2 flow steps; LEWAM_FLOW_STEPS at eval
  A33 model.encoder_backbone=vit (LeWAM's own)
Evals: hjepa/lewam_eval_plus.py (LeWAM's eval_lewam.py with these models).
"""

import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lewam_common import ROOT, lewam_model, trainer_namespace  # noqa: E402

import lewam.train.datasets as lewam_data  # noqa: E402
import lewam.train.utils as lewam_utils  # noqa: E402
from lewam.models.module import sinusoid  # noqa: E402

DEFAULTS = dict(
    proprio_keys=[], proprio_dims=None, p_drop_proprio=0.3,
    roll_k=1, roll_frac=0.25, w_roll=1.0,
    shortcut=False, shortcut_m=32, shortcut_frac=0.25, w_shortcut=1.0,
    distill=False, distill_k=16, distill_steps=5, distill_every=4, distill_frac=0.25, distill_from=10, w_distill=1.0,
    hier=False, hier_far=30, hier_sub=5, hier_frac=0.5, w_hier=1.0,
)
PLUS = {"epoch": 0, "step": 0, "proprio": None}  # the trainer's epoch and train step; the proprio table (A17)
TIME_SCALE = 1000.0  # new sinusoid inputs (step size, subgoal flow time and horizon) span [0, 1000], as in DiT


def options(cfg):
    """The plus options of a run: from a Hydra config (cfg.plus) or a checkpoint config (cfg["plus"])."""
    if hasattr(cfg, "train"):
        from omegaconf import OmegaConf

        given = OmegaConf.to_container(cfg.plus, resolve=True) if "plus" in cfg else {}
    else:
        given = cfg.get("plus") or {}
    unknown = set(given) - set(DEFAULTS) - {"proprio_dim", "proprio_mean", "proprio_std"}
    assert not unknown, f"unknown plus options {sorted(unknown)}"
    return {**DEFAULTS, **given}


# ------------------------------------------------------------------ data


def proprio_table(dataset_path, keys, dims):
    """(N, P) float32, z-scored: the h5 columns `keys` side by side, then the columns `dims` of them; with the
    mean and std (constant columns get std 1, as LeWAM's actions)."""
    if PLUS["proprio"] is None:
        with h5py.File(dataset_path, "r") as f:
            cols = [np.asarray(f[k][:], dtype=np.float32).reshape(f[k].shape[0], -1) for k in keys]
        raw = np.concatenate(cols, 1)
        if dims is not None:
            raw = raw[:, list(dims)]
        ok = raw[~np.isnan(raw).any(1)]
        mean, std = ok.mean(0), ok.std(0)
        std[std == 0] = 1.0
        PLUS["proprio"] = (torch.from_numpy((raw - mean) / std), mean.tolist(), std.tolist())
    return PLUS["proprio"]


class PlusDataset(lewam_data.TrajectoryDataset):
    """LeWAM's sample plus a dict: the proprio at the start frame (A17); the data's actions for the next
    max(roll_k, distill_steps) chunks and which of those chunks lie in the episode (A28, A29); the frame rows 2..roll_k
    chunks ahead (A29); a far goal up to hier_far chunks ahead, the row hier_sub chunks ahead (or the far goal if
    nearer) and the far horizon (A26). PlusCollate reads the frames of those rows for a fixed share of each batch."""

    def __init__(self, cfg, *args):
        super().__init__(*args)
        self.p = options(cfg)
        self.proprio = None
        if self.p["proprio_keys"]:
            self.proprio = proprio_table(cfg.dataset_path, self.p["proprio_keys"], self.p["proprio_dims"])[0]
        self.n_chunks = max(self.p["roll_k"], self.p["distill_steps"] if self.p["distill"] else 1)

    def __getitem__(self, idx):
        base = super().__getitem__(idx)
        start_idx = int(self.possible_starts[idx])
        t0, left = int(self.frame_starts[start_idx]), int(self.frames_to_terminal[start_idx])
        fs, p = self.frameskip, self.p
        n_act = self.n_chunks * fs
        n_valid = min(n_act, left)
        actions = torch.zeros(n_act, self.actions.shape[1])
        actions[:n_valid] = self.actions[t0:t0 + n_valid].float()
        extra = {"roll_actions": actions,
                 "roll_valid": torch.tensor([float((q + 1) * fs <= left) for q in range(self.n_chunks)])}
        if self.proprio is not None:
            extra["proprio"] = self.proprio[t0]
        if p["roll_k"] > 1:
            extra["roll_rows"] = torch.tensor([t0 + min((q + 1) * fs, left) for q in range(1, p["roll_k"])])
        if p["hier"]:
            h_far = min(int(torch.randint(1, p["hier_far"] + 1, ()).item()), max(1, left // fs))
            h_sub = min(p["hier_sub"], h_far)
            extra["hier_rows"] = torch.tensor([t0 + min(h_far * fs, left), t0 + min(h_sub * fs, left)])
            extra["hier_h"] = torch.tensor(h_far / p["hier_far"])
        return (*base, extra)


class PlusCollate:
    """default_collate, then the frames of roll_rows (A29) and hier_rows (A26) for the first roll_frac and hier_frac
    of the batch: a fixed share, so no shape depends on the data (no GPU sync), and the loader reads only those."""

    def __init__(self, frames, cfg):
        self.frames, self.p = frames, options(cfg)

    def __call__(self, samples):
        batch = torch.utils.data.default_collate(samples)
        extra, B = batch[-1], len(samples)
        for key, frac in (("roll", self.p["roll_frac"]), ("hier", self.p["hier_frac"])):
            rows = extra.pop(f"{key}_rows", None)
            if rows is not None:
                b = max(1, round(frac * B))
                extra[f"{key}_frames"] = self.frames[rows[:b].reshape(-1)].reshape(b, rows.shape[1], *self.frames.shape[1:])
        return batch


# ------------------------------------------------------------------ model


class SubgoalHead(nn.Module):
    """A26: flow matching of the latent hier_sub chunks ahead, given the newest latent, a goal latent up to hier_far
    chunks away and that horizon (hier_far units). A residual MLP."""

    def __init__(self, z_dim, width=1024, depth=3, t_dim=64):
        super().__init__()
        self.t_dim = t_dim
        self.inp = nn.Linear(3 * z_dim + 2 * t_dim, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, z_dim))

    def velocity(self, z_t, tau, z_now, z_goal, h):
        x = self.inp(torch.cat([z_t, z_now, z_goal, sinusoid(tau * TIME_SCALE, self.t_dim),
                                sinusoid(h * TIME_SCALE, self.t_dim)], -1))
        for block in self.blocks:
            x = x + block(x)
        return self.out(x)

    def loss(self, z_now, z_goal, h, target):
        tau = torch.rand(target.shape[0], device=target.device)
        noise = torch.randn_like(target)
        v = self.velocity(torch.lerp(noise, target, tau[:, None].to(target.dtype)), tau, z_now, z_goal, h)
        return ((v.float() - (target - noise).float()) ** 2).mean()

    @torch.no_grad()
    def sample(self, z_now, z_goal, h, steps=8, generator=None):
        z = torch.randn(z_now.shape, device=z_now.device, generator=generator)
        for i in range(steps):
            tau = torch.full((z.shape[0],), i / steps, device=z.device)
            z = z + self.velocity(z, tau, z_now, z_goal, h).float() / steps
        return z


class LeWAMPlus(lewam_model.LeWAM):
    """LeWAM with the plus options (module docstring). With every option off it computes what LeWAM does: the same
    tokens, attention map, losses and samplers, so its own baseline runs through the same code."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.plus = p = options(cfg)
        self.H_max = int(cfg.get("H_max", 10))  # in the config, not kept by LeWAM
        self.ctx = {}  # eval: proprio of the replanning envs, guidance weight (PlusMixin)
        d = self.d_model
        self.proprio_dim = int(p.get("proprio_dim", 0)) if p["proprio_keys"] else 0
        if self.proprio_dim:
            self.proprio_proj = nn.Linear(self.proprio_dim, d)
            self.proprio_type = nn.Parameter(torch.zeros(1, d))
            self._add_proprio_token()
        if p["shortcut"]:
            assert self.action_head == "flow", "shortcut models need the flow action head"
            self.step_proj = nn.Linear(d, d)
            nn.init.zeros_(self.step_proj.weight)
            nn.init.zeros_(self.step_proj.bias)
        if p["roll_k"] > 1 or p["distill"]:
            assert self.state_head == "mse" and self.num_states_pred == 1 and self.num_views == 1, \
                "rollouts at training: one camera, one mse state slot"
        assert not p["distill"] or self.use_horizon, "distill plans to sampled goals (goal_type sampled)"
        if p["hier"]:
            assert self.num_views == 1 and self.goal_cond
            self.subgoal = SubgoalHead(self.z_dim)

    def _add_proprio_token(self):
        """One proprio token at the end of the state stream (before the clean actions). The state slots (dynamics),
        the clean actions and the action slots (policy) attend it; it attends the history and itself. The rest of
        the attention map is LeWAM's."""
        idx = self.token_indices
        at = (idx["clean_actions"] + idx["action_slots"])[0]
        shift = torch.tensor([i + (i >= at) for i in range(self.n_tokens)])
        n = self.n_tokens + 1
        attends = torch.zeros(n, n, dtype=torch.bool)
        attends[shift[:, None], shift[None, :]] = self.attends
        new = {k: [i + (i >= at) for i in rows] for k, rows in idx.items()}
        new["proprio"] = [at]
        attends[at, new["hist"] + [at]] = True
        for rows in (new["state_slots"], new["clean_actions"], new["action_slots"]):
            if rows:
                attends[rows, at] = True
        self.attends, self.n_tokens, self.token_indices = attends, n, new
        state_rows, action_rows = new["state_slots"], new["action_slots"]
        self.layout = ((new["clean_actions"] + action_rows)[0],
                       slice(state_rows[0], state_rows[-1] + 1) if state_rows else slice(0, 0),
                       slice(action_rows[0], action_rows[-1] + 1))

    # -- forward

    def forward_mot(self, z_history, history_pad, clean_actions, noisy_actions, noisy_state, tau_a, tau_s,
                    z_goal=None, h_norm=None, goal_keep=None, proprio=None, proprio_keep=None, step_d=None):
        """LeWAM.forward_mot (the same tokens) plus the proprio token (proprio None: the eval context's, else
        dropped) and the shortcut step size (step_d None: the eval context's, else 0)."""
        if z_history.dim() == 4:
            z_history = z_history.flatten(1, 2)
        z_history = z_history.float()  # lewam_common: an fp32 token buffer under bf16 autocast
        B, d, dev, ti = z_history.shape[0], self.d_model, z_history.device, self.token_indices
        tokens = z_history.new_zeros(B, self.n_tokens, d)
        state_type = self.state_type.weight
        tokens[:, ti["hist"]] = self.hist_proj(z_history) + self._history_pos_embed() + state_type[0]
        slot_pos = self._state_slot_pos_embed()
        if self.state_head == "mse":
            tokens[:, ti["state_slots"]] = self.next_state_query.repeat(1, self.num_views, 1).expand(B, -1, -1) \
                + slot_pos + state_type[1]
        else:
            tokens[:, ti["state_slots"]] = self.next_state_proj(noisy_state) + slot_pos + state_type[1]
        if goal_keep is None:
            goal_keep = torch.full((B,), z_goal is not None, dtype=torch.bool, device=dev)
        if self.goal_cond and z_goal is not None:
            goal = self.goal_proj(z_goal[:, None] if z_goal.dim() == 2 else z_goal)
            if self.num_views > 1:
                goal = goal + self.view_pos[:, self.goal_view_indices]
            tokens[:, ti["goal"]] = goal + state_type[2]
        action_type = self.action_type.weight
        tokens[:, ti["clean_actions"]] = self.action_proj(clean_actions) \
            + self.action_pos_embed[:, :self.n_clean_actions] + action_type[0]
        if self.action_head == "mse":
            tokens[:, ti["action_slots"]] = self.action_query.expand(B, -1, -1) + self.action_pos_embed + action_type[1]
        else:
            tokens[:, ti["action_slots"]] = self.action_proj(noisy_actions) + self.action_pos_embed + action_type[1]
        if self.proprio_dim:
            if proprio is None:
                proprio, proprio_keep = self._context_proprio(B, dev)
            tokens[:, ti["proprio"]] = (self.proprio_proj(proprio.float()) + self.proprio_type)[:, None]

        cond_state, cond_action = self._compute_slot_conditioning(tau_a, tau_s, h_norm, goal_keep, B, dev)
        if self.plus["shortcut"] and cond_action is not None:
            if step_d is None:
                step_d = torch.full((B,), float(self.ctx.get("step_d", 0.0)), device=dev)
            cond_action = cond_action + self.step_proj(sinusoid(step_d * TIME_SCALE, d))
        mask = self._plus_mask(history_pad, goal_keep if self.goal_cond else None,
                               proprio_keep if self.proprio_dim else None)
        for block in self.blocks:
            tokens = block(tokens, mask, self.layout, cond_state, cond_action)
        return self.action_out(tokens[:, ti["action_slots"]]), self.state_out(tokens[:, ti["state_slots"]])

    def _plus_mask(self, history_pad, goal_keep, proprio_keep):
        """lewam_common's mask (LeWAM's, no GPU sync) with a dropped proprio token masked as a key. A dropped
        token's own row still attends the history, so no row is empty."""
        B = history_pad.shape[0]
        key_pad = torch.zeros(B, self.n_tokens, dtype=torch.bool, device=history_pad.device)
        key_pad[:, self.token_indices["hist"]] = history_pad.repeat(1, self.num_views)
        if goal_keep is not None:
            key_pad[:, self.token_indices["goal"]] = ~goal_keep[:, None]
        if proprio_keep is not None:
            key_pad[:, self.token_indices["proprio"]] = ~proprio_keep[:, None]
        attends = self.attends[None] & ~key_pad[:, None, :]
        mask = torch.zeros(B, self.n_tokens, self.n_tokens, device=history_pad.device)
        return mask.masked_fill(~attends, float("-inf"))[:, None]

    def _context_proprio(self, B, dev):
        """The eval context's proprio (one row per replanning env, set by PlusMixin; a planner's rows repeat each
        env's), or a dropped token: no context, an imagined step (the planners do not predict proprio), or rows
        that are not whole repeats of the envs."""
        c = self.ctx.get("proprio")
        if c is None or not self.ctx.get("keep") or B % c.shape[0]:
            return torch.zeros(B, self.proprio_dim, device=dev), torch.zeros(B, dtype=torch.bool, device=dev)
        return c.repeat_interleave(B // c.shape[0], 0), torch.ones(B, dtype=torch.bool, device=dev)

    def step_size(self, n):
        """The shortcut step condition for n Euler steps: 1/n where the self-consistency loss trained it
        (n < shortcut_m), else 0 (the flow velocity)."""
        return 1.0 / n if n < self.plus["shortcut_m"] else 0.0

    # -- inference

    def predict_actions(self, z_history, history_pad, z_goal=None, h_norm=None, action_noise=None, generator=None):
        """LeWAM.predict_actions (Euler from noise at tau 0 to actions at tau 1) with classifier-free guidance
        (ctx cfg_w = w != 1: v = v_free + w (v_goal - v_free), the goal-free half of a doubled batch; A30), the
        shortcut step size (A32) and the context's proprio."""
        if self.action_head != "flow":
            return super().predict_actions(z_history, history_pad, z_goal, h_norm, action_noise, generator)
        B, dev, n = z_history.shape[0], z_history.device, self.n_flow_steps
        w = float(self.ctx.get("cfg_w", 1.0))
        guided = w != 1.0 and self.goal_cond and z_goal is not None
        zero_clean = torch.zeros(B, self.n_clean_actions, self.action_raw_dim, device=dev)
        zero_state = torch.zeros(B, self.n_state_tokens, self.z_dim, device=dev)
        one = torch.ones(B, device=dev)
        goal_keep = torch.full((B,), z_goal is not None, dtype=torch.bool, device=dev)
        proprio, proprio_keep = self._context_proprio(B, dev) if self.proprio_dim else (None, None)
        step_d = torch.full((B,), self.step_size(n), device=dev) if self.plus["shortcut"] else None
        fixed = [z_history, history_pad, zero_clean, zero_state, one, z_goal, h_norm, goal_keep, proprio, proprio_keep,
                 step_d]
        if guided:
            fixed = [None if x is None else torch.cat([x, x]) for x in fixed]
            fixed[7] = torch.cat([goal_keep, torch.zeros_like(goal_keep)])
        hist, pad, clean, state, one_, goal, h, keep, prop, prop_keep, step = fixed
        actions = action_noise if action_noise is not None else self.sample_action_noise(B, dev, generator)
        for i in range(n):
            tau = torch.full((B,), i / n, device=dev)
            x, t = (torch.cat([actions, actions]), torch.cat([tau, tau])) if guided else (actions, tau)
            v, _ = self.forward_mot(hist, pad, clean, x, state, t, one_, goal, h, keep, prop, prop_keep, step)
            if guided:
                v_goal, v_free = v.chunk(2)
                v = v_free + w * (v_goal - v_free)
            actions = actions + v / n
        return actions

    # -- training

    def _action_inputs(self, B, dev):
        zero_clean = torch.zeros(B, self.n_clean_actions, self.action_raw_dim, device=dev)
        zero_state = torch.zeros(B, self.n_state_tokens, self.z_dim, device=dev) if self.state_head == "flow" else None
        return zero_clean, zero_state, torch.ones(B, device=dev)

    def plus_loss(self, z_history, history_pad, action_target, action_valid, state_target, state_valid, z_goal,
                  h_norm, goal_keep, proprio, proprio_keep):
        """LeWAM.loss with the proprio token and, for shortcut models, step d = 0 (plain flow matching). Also
        returns the predicted state slots (the first step of the A29 rollout)."""
        B, dev = z_history.shape[0], z_history.device
        if self.action_head == "flow":
            tau_a = self._sample_flow_time(B, self.flow_alpha_act, dev)
            action_noise = torch.randn_like(action_target)
            noisy_actions = torch.lerp(action_noise, action_target, tau_a[:, None, None].to(action_target.dtype))
        else:
            tau_a = action_noise = noisy_actions = None
        if self.state_head == "flow":
            tau_s = self._sample_flow_time(B, self.flow_alpha_state, dev)
            state_noise = torch.randn_like(state_target)
            noisy_states = torch.lerp(state_noise, state_target, tau_s[:, None, None].to(state_target.dtype))
        else:
            tau_s = state_noise = noisy_states = None
        step_d = torch.zeros(B, device=dev) if self.plus["shortcut"] else None
        action_pred, state_pred = self.forward_mot(
            z_history, history_pad, action_target[:, :self.n_clean_actions], noisy_actions, noisy_states, tau_a, tau_s,
            z_goal, h_norm, goal_keep, proprio, proprio_keep, step_d)
        target = action_target if self.action_head == "mse" else action_target - action_noise
        loss_action = self._masked_mse(action_pred, target, action_valid)
        if self.num_states_pred == 0:
            return loss_action, loss_action.new_zeros(()), state_pred
        target = state_target if self.state_head == "mse" else state_target - state_noise
        return loss_action, self._masked_mse(state_pred, target, state_valid), state_pred

    def action_loss(self, z_history, history_pad, action_target, action_valid, z_goal, h_norm, proprio=None,
                    proprio_keep=None):
        """The flow-matching loss of the action head alone (A28: toward the planner's actions)."""
        B, dev = z_history.shape[0], z_history.device
        tau = self._sample_flow_time(B, self.flow_alpha_act, dev)
        noise = torch.randn_like(action_target)
        clean, state, one = self._action_inputs(B, dev)
        if self.proprio_dim and proprio is None:
            proprio, proprio_keep = torch.zeros(B, self.proprio_dim, device=dev), torch.zeros(B, dtype=torch.bool, device=dev)
        step_d = torch.zeros(B, device=dev) if self.plus["shortcut"] else None
        pred, _ = self.forward_mot(z_history, history_pad, clean, torch.lerp(noise, action_target, tau[:, None, None]),
                                   state, tau, one, z_goal, h_norm, None, proprio, proprio_keep, step_d)
        return self._masked_mse(pred, action_target - noise, action_valid)

    def shortcut_loss(self, z_history, history_pad, action_target, action_valid, z_goal, h_norm, goal_keep, proprio,
                      proprio_keep):
        """A32, the self-consistency of shortcut models (Frans et al. 2024): one step of size D from tau (on the
        grid of D) equals the mean of two steps of size D/2, for D in {2/M, ..., 1}; the smallest half step (1/M)
        is the flow velocity (d = 0), which plus_loss trains."""
        M = self.plus["shortcut_m"]
        B, dev = z_history.shape[0], z_history.device
        D = 2.0 ** torch.randint(1, int(math.log2(M)) + 1, (B,), device=dev) / M
        half = D / 2
        tau = torch.floor(torch.rand(B, device=dev) / D) * D
        noise = torch.randn_like(action_target)
        x = torch.lerp(noise, action_target, tau[:, None, None])
        clean, state, one = self._action_inputs(B, dev)

        def velocity(x, tau, d):
            return self.forward_mot(z_history, history_pad, clean, x, state, tau, one, z_goal, h_norm, goal_keep,
                                    proprio, proprio_keep, d)[0].float()

        with torch.no_grad():
            d_half = torch.where(half <= 1.0 / M, torch.zeros_like(half), half)
            v1 = velocity(x, tau, d_half)
            v2 = velocity(x + half[:, None, None] * v1, tau + half, d_half)
        return self._masked_mse(velocity(x, tau, D), (v1 + v2) / 2, action_valid)

    def _state_step(self, hist, pad, chunk):
        """One imagined chunk: the predicted next latent (b, 1, z) and the slid history, as the planners do."""
        B, dev = hist.shape[0], hist.device
        clean = chunk[:, :self.n_clean_actions]
        zero_slots = torch.zeros(B, self.num_actions_pred, self.action_raw_dim, device=dev)
        no_prop = (torch.zeros(B, self.proprio_dim, device=dev), torch.zeros(B, dtype=torch.bool, device=dev)) \
            if self.proprio_dim else (None, None)
        _, pred = self.forward_mot(hist, pad, clean, zero_slots, None, torch.ones(B, device=dev), None, None, None,
                                   None, *no_prop, None)
        nxt = pred[:, :1]
        return nxt, torch.cat([hist[:, 1:], nxt.to(hist.dtype)], 1), \
            torch.cat([pad[:, 1:], torch.zeros_like(pad[:, :1])], 1)

    def rollout_loss(self, z_history, history_pad, first_pred, actions, targets, valid):
        """A29: the state loss on chunks 2..K, rolled out from the first predicted latent (plus_loss) on the model's
        own latents with the data's actions; imagined steps see no proprio, as at planning. targets (b, K - 1, z),
        valid (b, >= K): 1 where chunk q's target lies in the episode."""
        fs = self.fs
        hist = torch.cat([z_history[:, 1:].float(), first_pred.float()], 1)
        pad = torch.cat([history_pad[:, 1:], torch.zeros_like(history_pad[:, :1])], 1)
        losses = []
        for q in range(1, targets.shape[1] + 1):
            nxt, hist, pad = self._state_step(hist, pad, actions[:, q * fs:(q + 1) * fs])
            losses.append(self._masked_mse(nxt, targets[:, q - 1:q], valid[:, q:q + 1]))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def imagine_best(self, z_history, history_pad, z_goal, h_steps, data_actions, K, n_steps):
        """A28: the eval planner's best_of_k at training. K policy rollouts of n_steps chunks from each history
        toward its goal, the cost at the goal step (h_steps, or the last imagined step); and the data's own actions
        rolled out the same way. Returns the cheapest plans (b, n_steps * fs, A), their cost and the data's cost."""
        b, fs, dev = z_history.shape[0], self.fs, z_history.device
        rows = torch.arange(b * K, device=dev)
        hist, pad = z_history.repeat_interleave(K, 0).float(), history_pad.repeat_interleave(K, 0)
        goal, h = z_goal.repeat_interleave(K, 0), h_steps.repeat_interleave(K, 0)
        plans, latents = [], []
        for step in range(n_steps):
            h_norm = (h - step).clamp(min=1.0, max=float(self.H_max)) / self.H_max
            chunk = self.predict_actions(hist, pad, z_goal=goal, h_norm=h_norm)[:, :fs]
            nxt, hist, pad = self._state_step(hist, pad, chunk)
            plans.append(chunk)
            latents.append(nxt[:, 0].float())
        at_goal = (h.clamp(max=n_steps) - 1).long()
        cost = ((torch.stack(latents, 1)[rows, at_goal] - goal.float()) ** 2).mean(-1).view(b, K)
        best_cost, best = cost.min(1)
        plans = torch.stack(plans, 1).reshape(b, K, n_steps * fs, -1)[torch.arange(b, device=dev), best]
        hist, pad, latents = z_history.float(), history_pad, []
        for step in range(n_steps):
            nxt, hist, pad = self._state_step(hist, pad, data_actions[:, step * fs:(step + 1) * fs])
            latents.append(nxt[:, 0].float())
        at_goal = (h_steps.clamp(max=n_steps) - 1).long()
        data_cost = ((torch.stack(latents, 1)[torch.arange(b, device=dev), at_goal] - z_goal.float()) ** 2).mean(-1)
        return plans, best_cost, data_cost


def build_model(cfg):
    """lewam.models.lewam.build_model, a LeWAMPlus for a config with a `plus` section."""
    if "plus" not in cfg:
        return _build_model(cfg)
    defaults = dict(encoder_backbone="resnet18dp", encoder_ckpt=None, img_size=224, z_dim=192, proj_hidden=768,
                    d_model=192, n_heads=4, depth=4, dropout=0.1, n_flow_steps=8, fs=5, num_actions_pred=10,
                    num_states_pred=1, history_len=2, goal_type="none", state_head="mse", flow_alpha_act=1.0,
                    num_views=1)
    return LeWAMPlus({**defaults, **cfg})


_build_model = lewam_model.build_model


# ------------------------------------------------------------------ trainer


@contextmanager
def eval_mode(model):
    was = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was)


def make_run_batch(ns):
    """LeWAM's run_batch (scripts/train_lewam.py; one camera) with the plus terms. `ns` is the trainer's namespace
    (its sigreg_loss, lewam_common's fast). The logged losses stay on the GPU (lewam_common nosync)."""

    def run_batch(batch, model, idm_head, sigreg, cfg, device, train):
        assert idm_head is None, "plus: no IDM head"
        *base, extra = batch
        tc, p = cfg.train, model.plus
        (history_frames, history_pad, action_target, action_valid, state_frames, state_valid, goal_frames,
         h_norm) = (x.to(device, non_blocking=True) for x in base)
        extra = {k: v.to(device, non_blocking=True) for k, v in extra.items()}
        action_target = action_target.float()
        assert history_frames.dim() == 5, "plus: one camera"
        B, n_history = history_frames.shape[:2]
        n_states = cfg.model.num_states_pred
        image_shape = history_frames.shape[-3:]
        goal_cond = cfg.model.goal_type != "none"
        if cfg.model.goal_type != "sampled":
            h_norm = None
        mean, std = lewam_utils._IMG_MEAN.to(device), lewam_utils._IMG_STD.to(device)

        def normalize(frames):
            return (frames.reshape(-1, *image_shape).float() / 255.0 - mean) / std

        pixels = [history_frames, state_frames] + ([goal_frames] if goal_cond else [])
        pixels = torch.cat([normalize(x) for x in pixels])
        if train and tc.encoder_checkpoint_chunks > 1:
            from torch.utils.checkpoint import checkpoint

            z = torch.cat([checkpoint(model.encode, chunk, use_reentrant=False)
                           for chunk in pixels.chunk(tc.encoder_checkpoint_chunks)])
        else:
            z = model.encode(pixels)
        z_history = z[:B * n_history].reshape(B, n_history, -1)
        state_target = z[B * n_history:B * (n_history + n_states)].reshape(B, n_states, -1)
        z_goal = goal_keep = None
        if goal_cond:
            z_goal = z[B * (n_history + n_states):]
            if train and tc.p_drop_goal > 0:
                goal_keep = torch.rand(B, device=device) >= tc.p_drop_goal
        proprio = proprio_keep = None
        if model.proprio_dim:
            proprio = extra["proprio"].float()
            proprio_keep = torch.rand(B, device=device) >= (p["p_drop_proprio"] if train else 0.0)

        loss_action, loss_state, state_pred = model.plus_loss(
            z_history, history_pad, action_target, action_valid, state_target, state_valid, z_goal, h_norm, goal_keep,
            proprio, proprio_keep)
        loss = tc.w_act * loss_action + tc.w_dyn * loss_state
        terms = {"act": loss_action.detach().float(), "state": loss_state.detach().float()}
        if tc.w_reg > 0:
            loss_reg = ns["_fast"](ns["sigreg_loss"])(sigreg, model, z_history, state_target, z_goal,
                                                     cfg.model.sigreg_grouped)
            loss = loss + tc.w_reg * loss_reg
            terms["reg"] = loss_reg.detach().float()

        if p["roll_k"] > 1:  # A29
            frames = extra["roll_frames"]
            b = frames.shape[0]
            with torch.no_grad():
                targets = model.encode(normalize(frames)).reshape(b, frames.shape[1], -1)
            loss_roll = model.rollout_loss(z_history[:b], history_pad[:b], state_pred[:b, :1], extra["roll_actions"][:b],
                                           targets, extra["roll_valid"][:b])
            loss = loss + p["w_roll"] * loss_roll
            terms["roll"] = loss_roll.detach().float()
        if p["shortcut"]:  # A32
            b = max(1, round(p["shortcut_frac"] * B))
            cut = lambda x: None if x is None else x[:b]  # noqa: E731
            loss_cut = model.shortcut_loss(
                z_history[:b].detach(), history_pad[:b], action_target[:b], action_valid[:b],
                None if z_goal is None else z_goal[:b].detach(), cut(h_norm), cut(goal_keep), cut(proprio),
                cut(proprio_keep))
            loss = loss + p["w_shortcut"] * loss_cut
            terms["shortcut"] = loss_cut.detach().float()
        if p["hier"]:  # A26
            frames = extra["hier_frames"]
            b = frames.shape[0]
            with torch.no_grad():
                far, sub = model.encode(normalize(frames)).reshape(b, 2, -1).float().unbind(1)
            loss_hier = model.subgoal.loss(z_history[:b, -1].detach().float(), far, extra["hier_h"][:b].float(), sub)
            loss = loss + p["w_hier"] * loss_hier
            terms["hier"] = loss_hier.detach().float()
        if p["distill"] and train:  # A28, every distill_every train steps from epoch distill_from
            PLUS["step"] += 1
            if PLUS["epoch"] >= p["distill_from"] and PLUS["step"] % p["distill_every"] == 0:
                b = max(1, round(p["distill_frac"] * B))
                hist, goal = z_history[:b].detach(), z_goal[:b].detach()
                with eval_mode(model):
                    plans, best_cost, data_cost = model.imagine_best(
                        hist, history_pad[:b], goal, (h_norm[:b] * cfg.model.H_max).round(), extra["roll_actions"][:b],
                        p["distill_k"], p["distill_steps"])
                use = (best_cost < data_cost).float()
                n_act = model.num_actions_pred
                target = torch.zeros(b, n_act, plans.shape[-1], device=device)
                target[:, :min(n_act, plans.shape[1])] = plans[:, :n_act]
                valid = use[:, None] * (torch.arange(n_act, device=device) < plans.shape[1]).float()
                loss_distill = model.action_loss(hist, history_pad[:b], target, valid, goal, h_norm[:b])
                loss = loss + p["w_distill"] * loss_distill
                # logged x distill_every: the epoch means then estimate the value on a distill step
                terms["distill"] = loss_distill.detach().float() * p["distill_every"]
                terms["distill_use"] = use.mean().detach() * p["distill_every"]
                terms["plan_gain"] = (data_cost - best_cost).mean().detach() * p["distill_every"]
        if n_states:
            terms["zstd"] = state_target.reshape(-1, state_target.shape[-1]).std(0).mean().detach().float()
        return loss, terms, B

    return run_batch


def model_config(lewam_cfg, cfg):
    """LeWAM's model config plus the plus options and the proprio size and z-scoring (kept in lewam_config.json)."""
    plus = options(cfg)
    if plus["proprio_keys"]:
        table, mean, std = proprio_table(cfg.dataset_path, plus["proprio_keys"], plus["proprio_dims"])
        plus.update(proprio_dim=table.shape[1], proprio_mean=mean, proprio_std=std)
    return {**lewam_cfg, "plus": plus}


def stat_keys(store):
    first = [k for k in ("act", "state", "reg", "zstd") if k in store]
    return first + sorted(k for k in store if k not in first)


def trainer_patches():
    """(LeWAM's text, ours, count): our run_batch, dataset, collate, model config, epoch counter and log."""
    return [
        ("loss, loss_terms, n = run_batch(", "loss, loss_terms, n = _plus_run_batch(", 2),
        ("            TrajectoryDataset(\n", "            _PlusDataset(cfg,\n", 1),
        ("    loader_kwargs = dict(num_workers=data_cfg.num_workers, pin_memory=True)\n",
         "    loader_kwargs = dict(num_workers=data_cfg.num_workers, pin_memory=True, collate_fn=_PlusCollate(frames, cfg))\n", 1),
        ("    lewam_cfg = model_config(cfg, actions.shape[1])\n",
         "    lewam_cfg = _plus_model_config(model_config(cfg, actions.shape[1]), cfg)\n", 1),
        ("    for epoch in range(start_epoch, train_cfg.epochs):\n",
         "    for epoch in range(start_epoch, train_cfg.epochs):\n        _PLUS['epoch'] = epoch\n", 1),
        ('for key in ("act", "state", "reg", "zstd") if key in store)', "for key in _stat_keys(store))", 1),
    ]


def install():
    """Models with a `plus` section build as LeWAMPlus (training, warm starts; the eval loader: lewam_eval_plus)."""
    lewam_model.build_model = build_model


if __name__ == "__main__":
    import lewam_train

    config, run, data, *overrides = sys.argv[1:]
    install()
    config_dir = ROOT / "hjepa" / "config" / "lewam"
    init = lewam_train.config_value(config_dir, config, "init_from")
    if init:
        lewam_train.warm_start(ROOT / init)
    lewam_train.speedups()
    os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data"))
    run_dir = ROOT / "data" / "ckpts" / "lewam" / run / "seed42"
    rank_suffix = "" if os.environ.get("RANK", "0") == "0" else os.environ["RANK"]
    sys.argv = [str(ROOT / "third_party" / "lewam" / "scripts" / "train_lewam.py"), "--config-dir", str(config_dir),
                "--config-name", config, f"dataset_path={data}", f"run_name={run}", f"run_dir={run_dir}",
                f"hydra.run.dir={run_dir}/hydra{rank_suffix}", *overrides]
    names = {"_PlusDataset": PlusDataset, "_PlusCollate": PlusCollate, "_plus_model_config": model_config,
             "_PLUS": PLUS, "_stat_keys": stat_keys}
    ns = {}
    names["_plus_run_batch"] = lambda *args, **kwargs: ns["run_batch_plus"](*args, **kwargs)
    ns["run_batch_plus"] = make_run_batch(ns)
    trainer_namespace("__main__", nosync=os.environ.get("LEWAM_NOSYNC", "1") == "1", patches=trainer_patches(),
                      names=names, namespace=ns)

"""LeWAM's eval (third_party/lewam/scripts/eval_lewam.py) for lewam_plus models (PLAN.md A27 step 3); plain LeWAM
checkpoints evaluate as before.

    python hjepa/lewam_eval_plus.py [eval_lewam.py arguments ...]

Environment (printed first in the log; LeWAM's results file does not record it):
  LEWAM_CFG=w          classifier-free guidance weight of the policy's goal (A30; 1 = off)
  LEWAM_FLOW_STEPS=n   Euler steps of the flow samplers (A32; 0 = the checkpoint's)
  LEWAM_HIER=1         the subgoal head's latent as the goal while the goal is farther than hier_sub chunks (A26)
  LEWAM_PROPRIO=0      no proprio token (A17: the same model, vision only)
lewam_eval_groups.py uses this with LEWAM_PLUS=1.
"""

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lewam_plus  # noqa: E402  (through lewam_common: LeWAM's code on the path)

import lewam.eval.build as build  # noqa: E402
import lewam.eval.loaders as loaders  # noqa: E402
from lewam.eval.planners import LeWAMPlanner  # noqa: E402
from lewam.eval.policy import LeWAMPolicy  # noqa: E402

SCRIPT = lewam_plus.ROOT / "third_party" / "lewam" / "scripts" / "eval_lewam.py"
SETTINGS = dict(cfg_w=float(os.environ.get("LEWAM_CFG", "1")), flow_steps=int(os.environ.get("LEWAM_FLOW_STEPS", "0")),
                hier=os.environ.get("LEWAM_HIER", "0") == "1", proprio=os.environ.get("LEWAM_PROPRIO", "1") == "1")


class PlusMixin:
    """At each replan of a LeWAMPlus model: the guidance weight; the proprio of the replanning envs, z-scored as at
    training (imagined steps drop it: _slide_history; steer, which plans envs in batches, never gets it); with
    LEWAM_HIER, the subgoal head's latent as the goal while the goal is farther than hier_sub chunks, the horizon then
    hier_sub chunks (a new subgoal at every replan)."""

    _goal_override = None

    def _plus(self):
        return isinstance(self.model, lewam_plus.LeWAMPlus)

    def _propose(self, info_dict, replan, history, history_pad):
        if not self._plus():
            return super()._propose(info_dict, replan, history, history_pad)
        model = self.model
        model.ctx = {"cfg_w": SETTINGS["cfg_w"]}
        if model.proprio_dim and SETTINGS["proprio"] and getattr(self, "plan_mode", None) != "steer":
            model.ctx.update(proprio=self._proprio(info_dict, replan), keep=True)
        if not (SETTINGS["hier"] and model.plus["hier"]):
            return super()._propose(info_dict, replan, history, history_pad)

        p = model.plus
        z_goal = super()._goal_latents(info_dict, replan)
        steps = np.maximum(self._steps_left[replan], 1.0)
        far = steps > p["hier_sub"]
        if far.any():
            z_now = history[:, -1] if history.dim() == 3 else history[:, 0, -1]
            h = torch.tensor(np.minimum(steps, p["hier_far"]) / p["hier_far"], device=self.device, dtype=torch.float32)
            z_sub = model.subgoal.sample(z_now.float(), z_goal.float(), h, generator=self._flow_generator(self.device))
            z_goal = torch.where(torch.as_tensor(far, device=self.device)[:, None], z_sub, z_goal.float())
        saved = self._steps_left.copy()
        self._steps_left[np.asarray(replan)[far]] = float(p["hier_sub"])
        self._goal_override = z_goal
        try:
            return super()._propose(info_dict, replan, history, history_pad)
        finally:
            self._goal_override = None
            self._steps_left[:] = saved

    def _goal_latents(self, info_dict, replan):
        if self._goal_override is not None:
            return self._goal_override
        return super()._goal_latents(info_dict, replan)

    def _proprio(self, info_dict, replan):
        p = self.model.plus
        cols = []
        for key in p["proprio_keys"]:
            assert key in info_dict, f"proprio: no info_dict[{key!r}] (keys {sorted(info_dict)})"
            x = info_dict[key][replan]
            x = (x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))).float().cpu()
            if x.dim() == 3:  # (envs, time, dims): the newest
                x = x[:, -1]
            cols.append(x.reshape(len(replan), -1))
        x = torch.cat(cols, 1)
        if p["proprio_dims"] is not None:
            x = x[:, list(p["proprio_dims"])]
        x = (x - torch.tensor(p["proprio_mean"])) / torch.tensor(p["proprio_std"])
        return x.to(self.device)

    # planners: the true proprio at the first imagined step of each rollout, none after
    def _proprio_on(self):
        if self._plus() and "proprio" in self.model.ctx:
            self.model.ctx["keep"] = True

    def _slide_history(self, history, history_pad, next_latent):
        if self._plus():
            self.model.ctx["keep"] = False
        return LeWAMPlanner._slide_history(history, history_pad, next_latent)

    def _imagine_rollout(self, *args, **kwargs):
        self._proprio_on()
        return super()._imagine_rollout(*args, **kwargs)

    def _rollout_cost_differentiable(self, *args, **kwargs):
        self._proprio_on()
        return super()._rollout_cost_differentiable(*args, **kwargs)


class PlusPolicy(PlusMixin, LeWAMPolicy):
    pass


class PlusPlanner(PlusMixin, LeWAMPlanner):
    pass


_load_lewam = loaders.load_lewam


def load_lewam(run_name, which="best"):
    model, cfg = _load_lewam(run_name, which)
    if SETTINGS["flow_steps"]:
        model.n_flow_steps = SETTINGS["flow_steps"]
    return model, cfg


def install():
    """LeWAM's loader builds plus models (lewam_plus.build_model) and its eval builds PlusPolicy / PlusPlanner."""
    lewam_plus.install()
    loaders.build_model = lewam_plus.build_model
    loaders.load_lewam = load_lewam
    build.LeWAMPolicy, build.LeWAMPlanner = PlusPolicy, PlusPlanner
    print(f"[plus eval] {SETTINGS}", flush=True)


if __name__ == "__main__":
    install()
    sys.argv = [str(SCRIPT), *sys.argv[1:]]
    runpy.run_path(str(SCRIPT), run_name="__main__")

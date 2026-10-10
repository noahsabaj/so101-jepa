"""LeWAM's eval (third_party/lewam/scripts/eval_lewam.py) on a small machine, and several evals in one process.

    python hjepa/lewam_eval_groups.py GROUP [eval_lewam.py arguments ...]
    LEWAM_EVAL_RUNS="policy:42 best_of_k:42 grad:42 ..." python hjepa/lewam_eval_groups.py GROUP --config-name cube ...

Memory: one Cube env as built takes ~0.35 GB more, mostly copies of what every env has the same of, so the paper's 50
at once need ~20 GB. lean_envs() (on by default; LEWAM_EVAL_LEAN=0 turns it off) keeps one of each: one renderer
context (the meshes and textures uploaded once; each env keeps its own scene from its own model: the same pixels,
checked), one inverse-kinematics model, one copy of each mesh file, and it gives the compiler's freed memory back to
the system. A Cube env then takes ~0.12 GB: 50 fit in ~9 GB.

Time: fast_setup() (on by default; LEWAM_EVAL_FAST=0 turns it off) removes work repeated at each reset and each run
(see its doc); the envs and the draw of episodes are the same. Render on the GPU where there is one (MUJOCO_GL=egl,
and in WSL GALLIUM_DRIVER=d3d12): with the shared renderer, one GL context serves every env.

Groups: with GROUP below eval.num_eval (50; GROUP must divide it), a World of GROUP envs runs the (episode, start)
pairs group by group and the per-episode successes are merged. This is not quite the paper's protocol: the policy's
random draws go to other episodes, and the gradient planner clips the gradient norm over all the envs it plans for at
once. GROUP=50 is the paper's eval exactly.

Several runs: LEWAM_EVAL_RUNS lists MODE:SEED pairs (modes in MODES); the other arguments are the overrides they share.
The envs, the dataset and the model are made once and serve every run: each run resets every env to its episode's
start state, as a fresh process would.

LEWAM_EVAL_VIDEO=1 keeps the per-episode videos (off by default: they do not change the result and cost time).
LEWAM_EVAL_VRAM_GB caps what PyTorch may take of the GPU (a GPU that drives displays): past it, CUDA out of memory.
LEWAM_EVAL_GRAD_ROWS (default 320; 0 = all at once): the gradient planner's forward and backward that many plans at a
time (chunked_grad_plan: the same gradient, less memory). On noah-pc (8 GB, displays): VRAM_GB 2.5, GRAD_ROWS 160.
"""

import ctypes
import hashlib
import os
import runpy
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import stable_worldmodel as swm

SCRIPT = Path(__file__).resolve().parents[1] / "third_party" / "lewam" / "scripts" / "eval_lewam.py"
GROUP = int(sys.argv[1]) if __name__ == "__main__" else 0
VIDEO = os.environ.get("LEWAM_EVAL_VIDEO", "0") == "1"
MODES = {  # the paper's three Cube evals: the reactive policy, best of 32 imagined rollouts, gradient planning
    "policy": ["eval.mode=lewam_policy"],
    "best_of_k": ["eval.mode=lewam_plan", "eval.plan_mode=best_of_k"],
    "grad": ["eval.mode=lewam_plan", "eval.plan_mode=grad", "eval.grad_all_k=true", "eval.grad_tr=0.01"],
}


class SharedRenderer(mujoco.Renderer):
    """A mujoco.Renderer that borrows the GL context and MjrContext (the uploaded meshes and textures, the offscreen
    and shadow buffers) of the first renderer made for the same meshes, textures and render settings. Its scene is
    its own, from its own model, so per-env colours, lights and cameras still apply."""

    BASES = {}

    def __init__(self, model, height, width):
        arrays = [getattr(model, name) for name in ("mesh_vert", "mesh_face", "mesh_normal", "mesh_texcoord",
                                                    "mesh_facenormal", "mesh_facetexcoord", "tex_data", "tex_height",
                                                    "tex_width", "tex_nchannel", "hfield_data", "skin_vert")
                  if hasattr(model, name)]
        q, g = model.vis.quality, model.vis.global_
        key = (height, width, q.shadowsize, q.offsamples, g.offwidth, g.offheight, model.nflex,
               hashlib.blake2b(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest())
        if key not in self.BASES:
            self.BASES[key] = mujoco.Renderer(model, height=height, width=width)
        self.__dict__.update(self.BASES[key].__dict__)
        self._model = model
        self._scene = mujoco.MjvScene(model=model, maxgeom=10000)
        self._scene_option = mujoco.MjvOption()
        self._depth_rendering = self._segmentation_rendering = False

    def close(self):
        self._gl_context = self._mjr_context = None  # the base renderer owns them


def lean_envs():
    """Share what every OGBench manipulation env has the same of (see the module doc)."""
    from dm_control.utils import io as dm_io
    from ogbench.manipspace import controllers
    from ogbench.manipspace.envs.env import CustomMuJoCoEnv

    read, files = dm_io.GetResource, {}

    def get_resource(name, mode="rb"):
        if (name, mode) not in files:
            files[name, mode] = read(name, mode)
        return files[name, mode]

    ik_init, ik_models = controllers.DiffIKController.__init__, {}

    def shared_ik(self, model, *args, **kwargs):  # the IK model is only read: one serves every env
        key = (model.nq, model.nbody, model.nsite, model.nmeshvert, model.body_pos.tobytes(), model.jnt_range.tobytes())
        ik_init(self, ik_models.setdefault(key, model), *args, **kwargs)

    compile_, trim = CustomMuJoCoEnv.compile_model_and_data, ctypes.CDLL("libc.so.6").malloc_trim

    def compile_and_trim(self):
        compile_(self)
        trim(0)

    def shared_renderer(self):
        if self._model is None:
            raise ValueError("Call `reset` before rendering.")
        self._renderer = SharedRenderer(self._model, self._render_height, self._render_width)
        mujoco.mjv_defaultFreeCamera(self._model, self._camera)

    dm_io.GetResource = get_resource
    controllers.DiffIKController.__init__ = shared_ik
    CustomMuJoCoEnv.compile_model_and_data = compile_and_trim
    CustomMuJoCoEnv._initialize_renderer = shared_renderer


def fast_setup():
    """Cut the time an eval spends before and between its steps, with the same envs and the same draw:
    - The Cube env's variation check marks its model dirty at every reset (np.allclose where it means "changed"), so
      every reset compiled the model again (~0.2 s an env). A model compiled from the same XML and assets is now
      copied (mj_copyModel), not compiled again.
    - dm_control hashed every mesh file (~60 MB) at every XML export: each file is hashed once.
    - sample_eval_episodes found each episode's length by a scan of every row per episode (~9 s a run on Cube): one
      pass now (np.maximum.at)."""
    import copy
    import inspect

    from dm_control.mjcf import attribute
    from lewam.eval import episodes
    from ogbench.manipspace import mjcf_utils
    from ogbench.manipspace.envs.env import CustomMuJoCoEnv

    class Hashlib:
        sums = {}

        def __getattr__(self, name):
            return getattr(hashlib, name)

        def sha1(self, data=b""):
            if not isinstance(data, bytes) or len(data) < 2**20:
                return hashlib.sha1(data)
            if id(data) not in self.sums:
                self.sums[id(data)] = (data, hashlib.sha1(data))  # the data kept: its id is not reused
            return self.sums[id(data)][1].copy()

    attribute.hashlib = Hashlib()

    models, trim = {}, ctypes.CDLL("libc.so.6").malloc_trim

    def compile_once(self):  # CustomMuJoCoEnv.compile_model_and_data, with the compile done once per XML and assets
        glob = getattr(self._mjcf_model.visual, "global")
        glob.offwidth, glob.offheight = self._render_width, self._render_height
        xml, assets = mjcf_utils.to_string(self._mjcf_model), mjcf_utils.get_assets(self._mjcf_model)
        key = (xml, tuple(sorted((name, id(data)) for name, data in assets.items())))
        if key not in models:
            models[key] = (mujoco.MjModel.from_xml_string(xml=xml, assets=assets), list(assets.values()))
        self._model = copy.deepcopy(models[key][0])
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = self._physics_timestep
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        if self._passive_viewer_handle is not None:
            self._passive_viewer_handle._sim().load(self._model, self._data, "")
        if self._renderer is not None:
            self._renderer.close()
            self._initialize_renderer()
        self.post_compilation()
        self._dirty = False
        trim(0)

    CustomMuJoCoEnv.compile_model_and_data = compile_once

    src = inspect.getsource(episodes.sample_eval_episodes)
    for old, new in [
        ("lengths = np.array([np.max(row_steps[row_episodes == episode]) + 1 for episode in episode_ids])",
         "row_ids = np.searchsorted(episode_ids, row_episodes); lengths = np.zeros(len(episode_ids), row_steps.dtype); "
         "np.maximum.at(lengths, row_ids, row_steps); lengths += 1"),
        ("row_max_start = np.array([max_start_by_episode[episode] for episode in row_episodes])",
         "row_max_start = max_start[row_ids]"),
    ]:
        assert old in src, f"sample_eval_episodes changed: {old!r} not found"
        src = src.replace(old, new)
    namespace = dict(vars(episodes))
    exec(compile(src, episodes.__file__, "exec"), namespace)
    episodes.sample_eval_episodes = namespace["sample_eval_episodes"]


class GroupedWorld(swm.World):
    def __init__(self, *args, num_envs, **kwargs):
        assert num_envs % GROUP == 0, f"group {GROUP} does not divide {num_envs} envs"
        super().__init__(*args, num_envs=GROUP, **kwargs)

    def evaluate(self, *, episodes_idx, start_steps, video=None, **kwargs):
        successes = []
        for a in range(0, len(episodes_idx), GROUP):
            self.set_policy(self.policy)  # fresh per-env state (histories, horizons); the random draws go on
            metrics = super().evaluate(episodes_idx=episodes_idx[a:a + GROUP], start_steps=start_steps[a:a + GROUP],
                                       video=video if VIDEO else None, **kwargs)
            successes.append(np.asarray(metrics["episode_successes"], dtype=bool))
            print(f"[groups] episodes {a}..{a + GROUP - 1}: {int(successes[-1].sum())}/{GROUP}", flush=True)
        successes = np.concatenate(successes)
        return {"success_rate": float(successes.mean() * 100.0), "episode_successes": successes}


GRAD_STEP = """                cost = self._rollout_cost_differentiable(opt_history, opt_pad, opt_goal, opt_steps, plan, state_noise)
                with torch.no_grad():
                    better = cost < best_cost
                    best_cost = torch.where(better, cost.detach(), best_cost)
                    best_plan[better] = plan.detach()[better]
                if iteration == self.grad_steps:
                    break
                loss = cost.sum()
                if self.grad_tr > 0:
                    loss = loss + self.grad_tr * ((plan - init_plan) ** 2).sum()
                optimizer.zero_grad()
                loss.backward()
"""
GRAD_STEP_CHUNKED = """                optimizer.zero_grad()
                costs = []
                for first in range(0, plan.shape[0], GRAD_ROWS):
                    rows = slice(first, first + GRAD_ROWS)
                    cost_rows = self._rollout_cost_differentiable(
                        opt_history[rows], opt_pad[rows], opt_goal[rows], None if opt_steps is None else opt_steps[rows],
                        plan[rows], None if state_noise is None else [None if n is None else n[rows] for n in state_noise])
                    if iteration < self.grad_steps:
                        loss = cost_rows.sum()
                        if self.grad_tr > 0:
                            loss = loss + self.grad_tr * ((plan[rows] - init_plan[rows]) ** 2).sum()
                        loss.backward()
                    costs.append(cost_rows.detach())
                cost = torch.cat(costs)
                with torch.no_grad():
                    better = cost < best_cost
                    best_cost = torch.where(better, cost.detach(), best_cost)
                    best_plan[better] = plan.detach()[better]
                if iteration == self.grad_steps:
                    break
"""


def chunked_grad_plan(rows):
    """LeWAMPlanner._gradient_plan with each Adam step's forward and backward done `rows` plans at a time, the
    gradients added up in plan.grad. A plan's cost depends on its own row only, so plan.grad is the one backward over
    all rows gives (to float rounding); the norm clip and the Adam step still see the whole plan. Peak memory falls
    from every row's activations to a chunk's: 50 Cube envs x 32 plans at once took over 3 GB."""
    import inspect
    import textwrap

    from lewam.eval import planners

    src = inspect.getsource(planners.LeWAMPlanner._gradient_plan)
    assert GRAD_STEP in src, "LeWAMPlanner._gradient_plan changed: its Adam step was not found"
    namespace = {**vars(planners), "GRAD_ROWS": rows}
    exec(compile(textwrap.dedent(src.replace(GRAD_STEP, GRAD_STEP_CHUNKED)), planners.__file__, "exec"), namespace)
    planners.LeWAMPlanner._gradient_plan = namespace["_gradient_plan"]


def made_once(make):
    """make, with each result kept and returned again for the same arguments (lists and configs as arguments too)."""
    made = {}

    def once(*args, **kwargs):
        key = repr((args, sorted(kwargs.items())))
        if key not in made:
            made[key] = make(*args, **kwargs)
        return made[key]

    return once


def run_many(runs, args):
    """Each MODE:SEED of runs as eval_lewam.py's main with args, in this process: the World (its envs), the dataset
    and the model are made by the first run and reused by the others."""
    import torch
    from hydra import compose, initialize_config_dir
    from lewam.eval import episodes, loaders

    config_name, overrides = "pusht", []
    it = iter(args)
    for arg in it:
        if arg == "--config-name":
            config_name = next(it)
        elif arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
        else:
            overrides.append(arg)
    episodes.get_dataset = made_once(episodes.get_dataset)
    loaders.load_lewam = made_once(loaders.load_lewam)
    swm.World = made_once(GroupedWorld)
    draw, draws = episodes.sample_eval_episodes, {}

    def draw_once(dataset, *args, **kwargs):  # the modes of one seed draw the same pairs: draw them once
        key = (id(dataset), repr((args, sorted(kwargs.items()))))
        if key not in draws:
            draws[key] = draw(dataset, *args, **kwargs)
        return draws[key]

    episodes.sample_eval_episodes = draw_once
    main = runpy.run_path(str(SCRIPT), run_name="lewam_eval")["main"]
    with initialize_config_dir(config_dir=str(SCRIPT.parents[1] / "configs" / "eval"), version_base="1.3"):
        for run in runs:
            mode, seed = run.split(":")
            cfg = compose(config_name=config_name, overrides=[*overrides, *MODES[mode], f"seed={seed}"])
            t0 = time.time()
            main(cfg)
            print(f"[run] {mode} {seed} done in {time.time() - t0:.0f} s", flush=True)
            torch.cuda.empty_cache()  # give back what this run's planner held: the next one starts with the GPU free


if __name__ == "__main__":
    if os.environ.get("LEWAM_EVAL_LEAN", "1") == "1":
        lean_envs()
    if os.environ.get("LEWAM_EVAL_FAST", "1") == "1":
        fast_setup()
    if os.environ.get("LEWAM_EVAL_GRAD_ROWS", "320") != "0":  # 0: the planner's own one backward over all rows
        chunked_grad_plan(int(os.environ.get("LEWAM_EVAL_GRAD_ROWS", "320")))
    if os.environ.get("LEWAM_EVAL_VRAM_GB"):
        import torch

        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(float(os.environ["LEWAM_EVAL_VRAM_GB"]) * 2**30 / total)
    sys.path.insert(0, str(SCRIPT.parent))
    if os.environ.get("LEWAM_EVAL_RUNS"):
        run_many(os.environ["LEWAM_EVAL_RUNS"].split(), sys.argv[2:])
    else:
        swm.World = GroupedWorld
        sys.argv = [str(SCRIPT), *sys.argv[2:]]
        runpy.run_path(str(SCRIPT), run_name="__main__")

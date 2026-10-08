"""Benchmark and pixel test of SO101Env.render() (RENDER_HILLCLIMB.md).

  python scripts/bench_render.py gl                    which GL renderer this process gets
  python scripts/bench_render.py save FILE.npz         baseline frames: 200 steps of random-action
                                                       rollouts, seeds 1000-1009, randomize=True
  python scripts/bench_render.py check FILE.npz        the same rollouts through env.step(): every
                                                       frame and every step result must be equal
  python scripts/bench_render.py profile [N]           physics and each part of render(), per step
  python scripts/bench_render.py ab FILE.npz ROUNDS V1 V2 ...
                                                       frames/s of each render variant (VARIANTS),
                                                       interleaved rounds, and its pixel test

Run with MUJOCO_GL=egl and Mesa's software device (llvmpipe), as in kat-pc's WSL jobs:
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<software device> python scripts/bench_render.py ...
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import mujoco  # noqa: E402

from env import MAX_ACTION, SO101Env  # noqa: E402

SEEDS = range(1000, 1010)
STEPS = 20  # per seed: 10 x 20 = 200 frames


def rollouts(env):
    """(seed, step, obs, qpos) over the random-action rollouts."""
    for seed in SEEDS:
        env.reset(seed)
        rng = np.random.default_rng(seed + 7)
        for t in range(STEPS):
            obs = env.step(rng.uniform(-MAX_ACTION, MAX_ACTION))
            yield seed, t, obs, env.data.qpos.copy()


def gl():
    import OpenGL.GL as GL
    env = SO101Env()
    env.renderer._gl_context.make_current()
    print("GL_VENDOR", GL.glGetString(GL.GL_VENDOR), "GL_RENDERER", GL.glGetString(GL.GL_RENDERER),
          "GL_VERSION", GL.glGetString(GL.GL_VERSION))


def save(path):
    env = SO101Env(randomize=True)
    frames, proprio, qpos = [], [], []
    for _, _, obs, q in rollouts(env):
        frames.append(obs["pixels"])
        proprio.append(obs["proprio"])
        qpos.append(q)
    np.savez_compressed(path, frames=np.stack(frames), proprio=np.stack(proprio), qpos=np.stack(qpos))
    print("saved", path, np.stack(frames).shape)


def check(path):
    ref = np.load(path)
    env = SO101Env(randomize=True)
    bad = 0
    for i, (_, _, obs, q) in enumerate(rollouts(env)):
        ok = (np.array_equal(obs["pixels"], ref["frames"][i]) and obs["pixels"].dtype == ref["frames"].dtype
              and np.array_equal(obs["proprio"], ref["proprio"][i]) and np.array_equal(q, ref["qpos"][i]))
        bad += not ok
    print(f"check: {200 - bad}/200 frames and step results equal -> {'PASS' if bad == 0 else 'FAIL'}")
    return bad == 0


def snapshots(env):
    """Per seed, the MjData after each step of the rollouts (render input), for replay."""
    out = []
    for seed in SEEDS:
        env.reset(seed)
        rng = np.random.default_rng(seed + 7)
        states = []
        for t in range(STEPS):
            env.step(rng.uniform(-MAX_ACTION, MAX_ACTION))
            states.append(copy.copy(env.data))
        out.append((seed, states))
    return out


def replay(env, snaps, render, frames=None):
    """Render every snapshot with `render(env)`; returns (seconds in render, frames)."""
    data, total, got = env.data, 0.0, []
    for seed, states in snaps:
        env.data = data
        env.reset(seed)  # the episode's domain (camera pose, light, colours) is in the model
        for d in states:
            env.data = d
            t = time.perf_counter()
            f = render(env)
            total += time.perf_counter() - t
            if frames is not None:
                got.append(f)
    env.data = data
    return total, got


# --- render variants -------------------------------------------------------------------------
def v_base(env):
    """The render() of commit b31a708."""
    views = []
    for cam in ("scene", "wrist"):
        env.renderer.update_scene(env.data, camera=cam)
        views.append(env.renderer.render())
    return np.concatenate(views, axis=1)


def v_current(env):
    """SO101Env.render as it is in this checkout."""
    return env.render()


def _bench(env):
    """Per-env state of the variants: fixed cameras, buffers, a second renderer, a worker thread."""
    b = getattr(env, "_bench", None)
    if b is None:
        from types import SimpleNamespace
        b = env._bench = SimpleNamespace()
        b.cams = []
        for name in ("scene", "wrist"):
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_FIXED
            c.fixedcamid = env.model.camera(name).id
            b.cams.append(c)
        h = env.renderer.height
        b.buf = [np.empty((h, h, 3), np.uint8), np.empty((h, h, 3), np.uint8)]
        b.wide = np.empty((h, 2 * h, 3), np.uint8)
        b.rect_l, b.rect_r = mujoco.MjrRect(0, 0, h, h), mujoco.MjrRect(h, 0, h, h)
        b.rect_w = mujoco.MjrRect(0, 0, 2 * h, h)
        b.r2 = b.r3 = None
        b.pool = None
    return b


def _draw(r, data, cam, rect=None):
    mujoco.mjv_updateScene(r.model, data, r._scene_option, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, r._scene)
    mujoco.mjr_render(rect or r._rect, r._scene, r._mjr_context)


def _assemble(b):
    h = b.buf[0].shape[0]
    out = np.empty((h, 2 * h, 3), np.uint8)
    out[:, :h] = b.buf[0][::-1]
    out[:, h:] = b.buf[1][::-1]
    return out


def v_direct(env):
    """One context; skip the Renderer wrapper (camera lookup, flipud copy, concatenate)."""
    b, r = _bench(env), env.renderer
    r._gl_context.make_current()
    for i in range(2):
        _draw(r, env.data, b.cams[i])
        mujoco.mjr_readPixels(b.buf[i], None, r._rect, r._mjr_context)
    return _assemble(b)


def v_side(env):
    """One context; the two views side by side in the offscreen buffer, one mjr_readPixels."""
    b, r = _bench(env), env.renderer
    r._gl_context.make_current()
    _draw(r, env.data, b.cams[0], b.rect_l)
    _draw(r, env.data, b.cams[1], b.rect_r)
    mujoco.mjr_readPixels(b.wide, None, b.rect_w, r._mjr_context)
    return b.wide[::-1].copy()


def v_pipe(env):
    """Two contexts, one thread: draw both views, then read both (the first view rasterizes while
    the second one's vertices are processed)."""
    b, r = _bench(env), env.renderer
    if b.r2 is None:
        b.r2 = mujoco.Renderer(env.model, r.height, r.width)
    r2 = b.r2
    r._gl_context.make_current()
    _draw(r, env.data, b.cams[0])
    r2._gl_context.make_current()
    _draw(r2, env.data, b.cams[1])
    r._gl_context.make_current()
    mujoco.mjr_readPixels(b.buf[0], None, r._rect, r._mjr_context)
    r2._gl_context.make_current()
    mujoco.mjr_readPixels(b.buf[1], None, r2._rect, r2._mjr_context)
    return _assemble(b)


def _wrist_job(env, b):
    if b.r3 is None:  # made in the worker thread, so its GL context is current there
        b.r3 = mujoco.Renderer(env.model, env.renderer.height, env.renderer.width)
    r2 = b.r3
    r2._gl_context.make_current()
    _draw(r2, env.data, b.cams[1])
    mujoco.mjr_readPixels(b.buf[1], None, r2._rect, r2._mjr_context)


def v_thread(env):
    """Two contexts, two threads: the wrist view renders in a worker thread beside the scene view."""
    b, r = _bench(env), env.renderer
    if b.pool is None:
        from concurrent.futures import ThreadPoolExecutor
        b.pool = ThreadPoolExecutor(max_workers=1)
    fut = b.pool.submit(_wrist_job, env, b)
    r._gl_context.make_current()
    _draw(r, env.data, b.cams[0])
    mujoco.mjr_readPixels(b.buf[0], None, r._rect, r._mjr_context)
    fut.result()
    return _assemble(b)


def _offset(x, y):
    def render(env):
        """One context; each view at viewport offset (x, y) of the offscreen buffer."""
        b, r = _bench(env), env.renderer
        h = r.height
        r._gl_context.make_current()
        for i in range(2):
            rect = mujoco.MjrRect(x + 2 * h * i, y, h, h)
            _draw(r, env.data, b.cams[i], rect)
            mujoco.mjr_readPixels(b.buf[i], None, rect, r._mjr_context)
        return _assemble(b)
    return render


def _without(flag):
    def render(env):
        """Diagnostic only (changes pixels): base render with a scene flag off, to split the cost."""
        s = env.renderer.scene
        s.flags[flag] = 0
        try:
            return v_base(env)
        finally:
            s.flags[flag] = 1
    return render


VARIANTS = {"base": v_base, "current": v_current, "direct": v_direct, "side": v_side, "pipe": v_pipe,
            "thread": v_thread, "off64": _offset(64, 0), "off32": _offset(32, 32), "off0_64": _offset(0, 64),
            "noshadow": _without(mujoco.mjtRndFlag.mjRND_SHADOW)}


def ab(path, rounds, names):
    ref = np.load(path)["frames"]
    env = SO101Env(randomize=True)
    snaps = snapshots(env)
    for name in names:  # pixel test first (and a warm-up)
        _, frames = replay(env, snaps, VARIANTS[name], frames=[])
        same = sum(np.array_equal(a, b) and a.dtype == b.dtype and a.shape == b.shape for a, b in zip(frames, ref))
        print(f"{name}: pixels {same}/200 equal -> {'IDENTICAL' if same == 200 else 'DIFFERENT'}", flush=True)
        if same < 200:
            f, g = np.stack(frames).astype(int), ref.astype(int)
            d = np.abs(f - g).max(axis=3)
            h = d.shape[2] // 2
            print(f"  differing pixels: scene view {(d[:, :, :h] > 0).sum()}, wrist view {(d[:, :, h:] > 0).sum()} "
                  f"of {d[:, :, :h].size} each; max abs difference {d.max()}; mean over differing "
                  f"{d[d > 0].mean():.2f}", flush=True)
    secs = {n: [] for n in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for name in order:
            secs[name].append(replay(env, snaps, VARIANTS[name])[0])
        print(f"round {r + 1}: " + "  ".join(f"{n} {200 / secs[n][-1]:.2f}" for n in names) + " frames/s", flush=True)
    base = names[0]
    for name in names:
        ratios = np.array(secs[base]) / np.array(secs[name])
        print(f"{name}: {200 * rounds / sum(secs[name]):.2f} frames/s; ratio to {base} per round "
              f"{np.round(ratios, 3).tolist()}, median {np.median(ratios):.3f}")


# --- profile ---------------------------------------------------------------------------------
def profile(n_per_seed):
    env = SO101Env(randomize=True)
    r = env.renderer
    keys = ["physics", "update_scene(scene)", "mjr_render(scene)", "mjr_readPixels(scene)", "flipud(scene)",
            "update_scene(wrist)", "mjr_render(wrist)", "mjr_readPixels(wrist)", "flipud(wrist)",
            "wrapper rest", "concatenate", "render() total"]
    acc = dict.fromkeys(keys, 0.0)
    count = 0
    for seed in SEEDS:
        env.reset(seed)
        rng = np.random.default_rng(seed + 7)
        for _ in range(n_per_seed):
            action = np.clip(rng.uniform(-MAX_ACTION, MAX_ACTION), -MAX_ACTION, MAX_ACTION)
            t0 = time.perf_counter()
            start = env.target
            env.target = np.clip(env.joints() + action, env.ctrl_lo, env.ctrl_hi)
            for k in range(env.substeps):
                env.data.ctrl[:] = env._command(start + (env.target - start) * (k + 1) / env.substeps)
                mujoco.mj_step(env.model, env.data)
            t1 = time.perf_counter()
            acc["physics"] += t1 - t0
            views = []
            for cam in ("scene", "wrist"):
                a = time.perf_counter()
                r.update_scene(env.data, camera=cam)
                b = time.perf_counter()
                # Renderer.render, step by step (mujoco 3.14 rendering/classic/renderer.py)
                original_flags = r._scene.flags.copy()
                r._gl_context.make_current()
                out = np.empty((r._height, r._width, 3), dtype=np.uint8)
                c = time.perf_counter()
                mujoco.mjr_render(r._rect, r._scene, r._mjr_context)
                d = time.perf_counter()
                mujoco.mjr_readPixels(out, None, r._rect, r._mjr_context)
                e = time.perf_counter()
                out[:] = np.flipud(out)
                f = time.perf_counter()
                del original_flags
                views.append(out)
                acc[f"update_scene({cam})"] += b - a
                acc["wrapper rest"] += c - b
                acc[f"mjr_render({cam})"] += d - c
                acc[f"mjr_readPixels({cam})"] += e - d
                acc[f"flipud({cam})"] += f - e
            g = time.perf_counter()
            np.concatenate(views, axis=1)
            h = time.perf_counter()
            acc["concatenate"] += h - g
            acc["render() total"] += h - t1
            count += 1
    step_total = acc["physics"] + acc["render() total"]
    print(f"{count} steps; ms per step (share of physics + render):")
    for k in keys:
        print(f"  {k:24s} {1000 * acc[k] / count:9.3f} ms  {100 * acc[k] / step_total:6.2f}%")
    print(f"  steps/s {count / step_total:.2f}; render() alone {count / acc['render() total']:.2f} frames/s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gl":
        gl()
    elif cmd == "save":
        save(sys.argv[2])
    elif cmd == "check":
        sys.exit(0 if check(sys.argv[2]) else 1)
    elif cmd == "profile":
        profile(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif cmd == "ab":
        ab(sys.argv[2], int(sys.argv[3]), sys.argv[4:])

# Render hill-climb: SO101Env.render()

Branch `perf-render` from `b31a708`, 2026-10-08. Goal: a faster `SO101Env.render()` (sim/env.py) with
pixel-identical output. Only `render()` changed.

## Result

| | frames/s (scene + wrist, 64x128) | ratio |
|---|---|---|
| Baseline (`b31a708`) | 1.71 | 1.00 |
| Final (`f0daa75`) | 1.83 | **1.073** (median of 3 interleaved rounds: 1.236, 1.070, 1.073) |

Round 1 of the final run measured the baseline at 1.48 frames/s (hp-1's power cap). Rounds 2 and 3, and
the 4 baseline passes in the two screening runs, measured 1.69-1.71. So the honest speedup is 1.07x.

Pixel test: `check` runs the 200 rollout steps through `env.step()` with the new code. 200/200 frames,
proprio and qpos are equal (`np.array_equal`) to the frames saved from `b31a708`. PASS.

## Setup

- Node: hp-1 (i5-1135G7, 8 threads, Mesa 26.0.8, LLVM 21.1.8). Baseline and candidate ran in the same
  job, in interleaved rounds (the order rotates each round). Each pass renders the same 200 states.
- GL: `MUJOCO_GL=egl`, as kat-pc's WSL jobs use (`scripts/gpu.sh` sets `GL=egl` on NVIDIA). WSL has no GPU
  EGL, so EGL gets Mesa's software device (llvmpipe). On hp-1, EGL device 0 is the Iris Xe GPU and
  device 1 is llvmpipe, so all runs used `MUJOCO_EGL_DEVICE_ID=1`
  (`GL_RENDERER llvmpipe (LLVM 21.1.8, 256 bits)`). hp-1 has no OSMesa: Mesa 25.1 removed it.
- Python: the worktree has no `third_party/H-JEPA` checkout, so `uv run` cannot sync the project there.
  `sim/` does not need H-JEPA. The runs used the venv of the main `so101-jepa` copy on hp-1, which is
  built from the same lock (mujoco 3.14.0, numpy 2.4.6, scipy 1.17.1, PyOpenGL 3.1.10). Nothing was
  downloaded.
- Baseline frames: random-action rollouts (`rng = default_rng(seed + 7)`, uniform in +-MAX_ACTION),
  seeds 1000-1009, 20 steps each, `randomize=True`. That is 200 frames, saved by `save` at `b31a708`
  to `outputs/render/baseline_frames.npz` (on hp-1; a copy is in this worktree's `outputs/`, which is
  not in git). llvmpipe output can depend on the CPU and the Mesa version, so make new baseline frames
  on any other machine.

## Profile (baseline, hp-1, 200 steps, ms per step)

| part | ms | share of step |
|---|---|---|
| physics (100 `mj_step`) | 2.21 | 0.4% |
| `update_scene(scene)` | 0.04 | 0.0% |
| `mjr_render(scene)` | 181.7 | 30.5% |
| `mjr_readPixels(scene)` | 269.2 | 45.1% |
| `flipud(scene)` | 0.03 | 0.0% |
| `update_scene(wrist)` | 0.05 | 0.0% |
| `mjr_render(wrist)` | 87.1 | 14.6% |
| `mjr_readPixels(wrist)` | 55.9 | 9.4% |
| `flipud(wrist)` | 0.04 | 0.0% |
| rest of the Renderer wrapper | 0.07 | 0.0% |
| `np.concatenate` | 0.01 | 0.0% |
| **render() total** | **594.2** | **99.6%** |

`render()` is 99.6% of `step()` plus `render()`, far over the 50% bar. GL runs asynchronously, so the
time in `mjr_render` is vertex work and triangle setup on the calling thread. The time in
`mjr_readPixels` is mostly the wait for llvmpipe's rasterizer threads. The scene view costs 3x the
wrist view (451 ms against 143 ms) because the whole arm is in view.

Diagnostic (not a candidate, because it changes pixels): with shadows off, render runs at 5.47 frames/s,
3.22x faster. So about 69% of the render time goes to shadows. `mjr_render` draws a 4096x4096 shadow map
and a second lit pass for each view.

## Ideas tried

Each idea was measured as a ratio to the baseline in the same job, and was pixel-tested on all 200
frames before its speed counted.

| # | idea | ratio | pixels | decision |
|---|---|---|---|---|
| 1 | `direct`: skip the Renderer wrapper (fixed `MjvCamera`s, `mjv_updateScene`/`mjr_render`/`mjr_readPixels` called directly, reused buffers, one copy with the flip) | 1.000 (0.997, 1.002) | identical | rejected: no gain, as the wrapper is 0.2 ms of 594 ms |
| 2 | `side`: both views side by side in the offscreen buffer (wrist viewport at x=64), one `mjr_readPixels` | 1.041 | **different**: scene view 0 pixels, wrist view 3,543 of 819,200 (max diff 59, mean 1.1) | rejected (pixels) |
| 3 | `pipe`: a second GL context, one thread: draw both views, then read both, so one view rasterizes while the other's vertices are processed | 1.020 | identical | rejected: under 5%; idea 4 is the stronger form |
| 4 | `thread`: the wrist view on a second GL context in a worker thread, beside the scene view (`mjr_render` releases the GIL) | 1.074 screening, 1.073 final | identical | **accepted** (`f0daa75`) |
| 5 | `off32`: each view at viewport offset (32, 32), so the 64x64 view spans 4 llvmpipe tiles (and 4 rasterizer threads) instead of 1 | 1.285 | **different**: 467 + 4,469 pixels of 2 x 819,200 (max 64, mean 1.2) | rejected (pixels) |
| 5b | `off64`, `off0_64` (the same at other offsets; pixel test only) | not timed | **different** (off64: 407 + 4,093 pixels) | rejected |

Ideas 1-4 were screened together in one job (2 rounds), and idea 5 in a second job. Any viewport
offset that is not 0 changes some pixels by 1 level, with a few edge pixels changing more. The
viewport transform rounds differently, and so MSAA coverage changes at a few triangle edges.

Not worth a run, with the reasons:
- Does `update_scene` rebuild static geometry on each call? It takes 0.04-0.05 ms (under 0.02%), and
  the mesh data lives in the `MjrContext` on the GL side already. There is nothing to save.
- Output buffers and copies: `flipud` and `concatenate` together take 0.08 ms. The `direct` variant
  removed them and gained nothing.
- Python-wrapper overhead: 0.07 ms.
- Culling the geoms outside the wrist camera: after idea 4 the wrist view is off the critical path. A
  geom outside the camera can still cast a shadow into the view, so this would also put pixels at risk.

Stop reason: ideas 1, 2 and 3 each gained under 5%. Idea 4, screened with them, passed, and idea 5 was
rejected for pixels. The profile shows that no pixel-identical lever is left inside `render()`: almost
all of the time is inside `mjr_render`.

## The accepted change (`f0daa75`)

`render()` makes a second `mujoco.Renderer` the first time it runs. It creates it inside a
one-thread `ThreadPoolExecutor`, so that its EGL context is current in that thread. Both scenes are
updated on the calling thread, because `mjv_updateScene` may use the `MjData` stack, which must not be
shared between threads. The worker then renders and reads back the wrist view while the calling
thread does the scene view. The output is a new array on each call, as before.

Cost: one more GL context per env. That is a second 4096x4096 shadow map, the offscreen buffers and
the mesh buffers. My estimate is about 100 MB, which I did not measure.

Throughput: the gain is per process. The second thread uses CPU that one process alone leaves idle.
A job that already fills every core with processes (e.g. the 14-process dataset build) gets no more
throughput from this change.

## Bottleneck after the change

The scene view's `mjr_render` and readback on the calling thread take about 450 ms of the ~545 ms
frame, and the wrist view now overlaps with it. About 69% of the render time is shadows. Each view
renders its own 4096x4096 shadow map and a second lit pass over the arm meshes. The 64x64 main pass
fits in one llvmpipe tile, so one rasterizer thread does it.

The levers that are left all change pixels or configuration, so they are not in this branch. They
need a decision:
- smaller `visual/quality/shadowsize` (default 4096): changes pixels. Shadows off entirely gave 3.2x.
- offset viewports, which spread each view over 4 tiles: 1.29x, with about 0.3% of the pixels changed,
  mostly by 1 level.
- `LP_NUM_THREADS` for multi-process jobs (environment, not `render()`): not measured.

## Rerun

From the worktree root (Git Bash), on hp-1:

```sh
fleet run --on hp-1 -- sh -c 'export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1; PY=../so101-jepa/.venv/bin/python; $PY scripts/bench_render.py check outputs/render/baseline_frames.npz && $PY scripts/bench_render.py ab outputs/render/baseline_frames.npz 3 base current'
```

Other commands: `profile 20` (the split above), and `ab FILE ROUNDS base direct side pipe thread off32
noshadow` (the variants). To make new baseline frames, first restore the baseline `render()` with
`git show b31a708:sim/env.py > sim/env.py`, then run `bench_render.py save outputs/render/baseline_frames.npz`
on the node, then `git checkout sim/env.py`.

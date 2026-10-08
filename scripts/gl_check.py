"""Which OpenGL renderer does MuJoCo get in kat-pc's WSL, and how fast does SO101Env render with each?

Run in WSL with the so101-jepa venv:  python gl_check.py   (spawns one child per configuration)
Child mode: python gl_check.py child NAME   (env vars set by the parent)
"""

import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.expanduser("~/so101-jepa")
CONFIGS = {
    "egl_default": {"MUJOCO_GL": "egl"},
    "egl_d3d12": {"MUJOCO_GL": "egl", "GALLIUM_DRIVER": "d3d12"},
    "egl_d3d12_nvidia": {"MUJOCO_GL": "egl", "GALLIUM_DRIVER": "d3d12", "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA"},
    "egl_llvmpipe": {"MUJOCO_GL": "egl", "LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe"},
}
OUT = "/tmp/gl_check"


def child(name):
    sys.path.insert(0, os.path.join(REPO, "sim"))
    from env import SO101Env
    from OpenGL import GL

    env = SO101Env(randomize=True)
    env.reset(1000)
    print(name, "GL_VENDOR", GL.glGetString(GL.GL_VENDOR), "GL_RENDERER", GL.glGetString(GL.GL_RENDERER), flush=True)
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(5):  # warm-up
        env.step(rng.uniform(-0.1, 0.1, 6))
    t = time.perf_counter()
    for _ in range(40):
        frames.append(env.step(rng.uniform(-0.1, 0.1, 6))["pixels"])
    dt = time.perf_counter() - t
    print(f"{name} steps/s {40 / dt:.2f}", flush=True)
    np.save(f"{OUT}/{name}.npy", np.stack(frames))


def main():
    os.makedirs(OUT, exist_ok=True)
    print("dxg:", os.path.exists("/dev/dxg"), "wsl libs:", sorted(os.listdir("/usr/lib/wsl/lib"))[:8] if os.path.isdir("/usr/lib/wsl/lib") else None)
    for name, extra in CONFIGS.items():
        env = {**os.environ, **extra}
        r = subprocess.run([sys.executable, __file__, "child", name], env=env, capture_output=True, text=True, timeout=600)
        print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if name in l or "Error" in l or "error" in l)[-1500:], flush=True)
    ref = f"{OUT}/egl_llvmpipe.npy"
    if os.path.exists(ref):
        a = np.load(ref)
        for name in CONFIGS:
            p = f"{OUT}/{name}.npy"
            if os.path.exists(p) and name != "egl_llvmpipe":
                b = np.load(p)
                d = np.abs(a.astype(int) - b.astype(int))
                print(f"{name} vs llvmpipe: identical {np.array_equal(a, b)}, differing pixels {(d.max(-1) > 0).mean():.4f}, "
                      f"max diff {d.max()}, mean diff {d.mean():.3f}", flush=True)


if __name__ == "__main__":
    child(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "child" else main()

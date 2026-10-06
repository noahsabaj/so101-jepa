"""Insertion test in MuJoCo: put an STS3215 servo into the pocket of the SO-101 base.

The base is fixed, made of the convex pieces from bakeoff/geometry.py. The servo is a free body,
driven by the compliant hand of bakeoff/hand.py (the same force law in all three simulators).

Per clearance (0, 0.2, 0.4 mm per side: the servo is made smaller) and hand condition (aligned,
perturbed; see hand.py), 100 trials (seeds 0-99).
Measures: success, the deepest penetration, unstable trials (NaN or a servo speed above 1 m/s),
physics steps per second, and render frames per second (64x64 and 480x640).

Run: MUJOCO_GL=egl uv run python bakeoff/insert_mujoco.py [trials]
"""

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

import hand

PARTS = Path("bakeoff/parts")
OUT = Path("bakeoff/results")
CLEARANCES_MM = [0.0, 0.2, 0.4]
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TIMESTEP = 0.0005
CONTACT = dict(solref=[0.002, 1.0], solimp=[0.95, 0.99, 0.0005, 0.5, 2], friction=[0.4, 0.005, 0.0001])


def servo_meshes(clearance):
    """Servo convex pieces made smaller by `clearance` m per side across the insertion axis (x)."""
    servo = trimesh.load(PARTS / "servo.obj")
    ext = servo.extents
    scale = np.array([1.0, (ext[1] - 2 * clearance) / ext[1], (ext[2] - 2 * clearance) / ext[2]])
    center = servo.bounds.mean(axis=0)
    pieces = []
    for f in sorted(PARTS.glob("servo_*.obj")):
        p = trimesh.load(f)
        p.vertices = (p.vertices - center) * scale + center
        pieces.append(p)
    return pieces


def add_mesh_geoms(spec, body, name, meshes, rgba, mass=None):
    for i, m in enumerate(meshes):
        spec.add_mesh(name=f"{name}_{i}", uservert=m.vertices.flatten().tolist(),
                      userface=m.faces.flatten().tolist())
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_{i}", rgba=rgba, **CONTACT)
        if mass is not None:
            geom.mass = mass / len(meshes)


def build_model(clearance, info):
    spec = mujoco.MjSpec()
    spec.option.timestep = TIMESTEP
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    world = spec.worldbody
    base = world.add_body(name="base")
    add_mesh_geoms(spec, base, "base", [trimesh.load(f) for f in sorted(PARTS.glob("base_[0-9]*.obj"))],
                   rgba=[1, 0.82, 0.12, 1])
    servo = world.add_body(name="servo")
    servo.add_freejoint()
    add_mesh_geoms(spec, servo, "servo", servo_meshes(clearance), rgba=[0.1, 0.1, 0.1, 1],
                   mass=info["servo_mass_kg"])
    cam = world.add_camera(name="view", pos=[0.16, -0.20, 0.16])
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    cam.targetbody = "base"
    world.add_light(pos=[0.2, -0.2, 0.4], dir=[-0.4, 0.4, -0.8], diffuse=[0.9, 0.9, 0.9])
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    return spec.compile()


def snapshot(model, data, path):
    with mujoco.Renderer(model, height=480, width=640) as r:
        r.update_scene(data, camera="view")
        Image.fromarray(r.render()).save(path)


def run_trial(model, data, info, rng, condition, snap=None):
    seated = np.array(info["seated_servo_pos"])
    err_pos, err_rot = hand.trial_error(rng, condition)
    body = model.body("servo").id
    mass = float(model.body_subtreemass[body])
    mujoco.mj_resetData(model, data)
    pos0, rot0 = hand.target_pose(0.0, seated, err_pos, err_rot)
    data.qpos[:3] = pos0
    q = rot0.as_quat()
    data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
    mujoco.mj_forward(model, data)
    if snap:
        snapshot(model, data, f"{snap}_start.png")

    steps = int((hand.START_OUT / hand.SPEED + hand.SETTLE) / TIMESTEP)
    max_pen, unstable = 0.0, False
    for k in range(steps):
        target_pos, target_rot = hand.target_pose(k * TIMESTEP, seated, err_pos, err_rot)
        q = data.qpos[3:7]
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        # qvel of a free joint: linear velocity (world), then angular velocity (body frame)
        f, tau = hand.wrench(data.qpos[:3], rot, data.qvel[:3], rot.apply(data.qvel[3:6]),
                             target_pos, target_rot, mass)
        data.xfrc_applied[body, :3], data.xfrc_applied[body, 3:] = f, tau
        mujoco.mj_step(model, data)
        if data.ncon:
            max_pen = max(max_pen, -float(data.contact.dist[:data.ncon].min()))
        if not np.isfinite(data.qpos).all() or np.linalg.norm(data.qvel[:3]) > 1.0:
            unstable = True
            break
    if snap:
        snapshot(model, data, f"{snap}_end.png")
    q = data.qpos[3:7]
    result = hand.outcome(data.qpos[:3].copy(), Rotation.from_quat([q[1], q[2], q[3], q[0]]), seated, unstable)
    return dict(result, max_pen_mm=max_pen * 1000, steps=k + 1)


def render_fps(model, data, size, n=300):
    with mujoco.Renderer(model, height=size[0], width=size[1]) as r:
        r.update_scene(data, camera="view")
        r.render()
        from OpenGL import GL
        gl = GL.glGetString(GL.GL_RENDERER).decode()
        t0 = time.perf_counter()
        for _ in range(n):
            r.update_scene(data, camera="view")
            r.render()
        return n / (time.perf_counter() - t0), gl


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = json.loads((PARTS / "parts.json").read_text())
    report = {"simulator": f"mujoco {mujoco.__version__}", "contact": "convex pieces (CoACD)", "clearances": {}}
    cases = [(c, cond) for c in CLEARANCES_MM for cond in hand.CONDITIONS]
    for n, (c, cond) in enumerate(cases):
        model = build_model(c / 1000, info)
        data = mujoco.MjData(model)
        rng = np.random.default_rng(0)
        trials, t0 = [], time.perf_counter()
        for i in range(TRIALS):
            trials.append(run_trial(model, data, info, rng, cond, snap=OUT / f"mujoco_c{c}_{cond}" if i == 0 else None))
            print(f"FLEET_PROGRESS {n * TRIALS + i + 1}/{len(cases) * TRIALS} {c} mm {cond}: {trials[-1]}", flush=True)
        wall = time.perf_counter() - t0
        report["clearances"][f"{c} {cond}"] = {
            "success": sum(t["success"] for t in trials), "trials": TRIALS,
            "unstable": sum(t["unstable"] for t in trials),
            "max_penetration_mm": round(max(t["max_pen_mm"] for t in trials), 3),
            "median_pos_err_mm": round(float(np.median([t["pos_err_mm"] for t in trials])), 3),
            "physics_steps_per_s": round(sum(t["steps"] for t in trials) / wall),
            "geoms": int(model.ngeom),
        }
    fps64, gl = render_fps(model, data, (64, 64))
    report["render_fps_64x64"], report["gl_renderer"] = round(fps64), gl
    report["render_fps_480x640"] = round(render_fps(model, data, (480, 640))[0])
    (OUT / "mujoco.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

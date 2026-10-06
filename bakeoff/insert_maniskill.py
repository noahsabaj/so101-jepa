"""Insertion test in ManiSkill's engine (SAPIEN 3, PhysX): the same parts, hand and trials as
bakeoff/insert_mujoco.py. ManiSkill builds its scenes on SAPIEN; this test uses SAPIEN directly
(one CPU scene, trials in sequence), so the numbers compare the contact physics.

Run: uv run --with mani_skill python bakeoff/insert_maniskill.py [trials]
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sapien
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

import hand

PARTS = Path(os.environ.get("PARTS", "bakeoff/parts"))
OUT = Path("bakeoff/results")
CLEARANCES_MM = [0.0, 0.2, 0.4]
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TIMESTEP = 0.0005
# PhysX settings for tight insertion (TGS solver, small contact offset, slow push-out of overlaps).
PHYSX = dict(tgs=True, contact_offset=0.0005, depenetration=0.05, position_iterations=20)  # best of a 4-setting sweep
PHYSX.update(json.loads(os.environ.get("PHYSX", "{}")))
CAM_POS = np.array([0.16, -0.20, 0.16])


def servo_pieces(clearance, tmp):
    """Servo convex pieces made smaller by `clearance` m per side across the insertion axis (x),
    written to `tmp` (SAPIEN reads collision shapes from files)."""
    servo = trimesh.load(PARTS / "servo.obj")
    ext, centre = servo.extents, servo.bounds.mean(axis=0)
    scale = np.array([1.0, (ext[1] - 2 * clearance) / ext[1], (ext[2] - 2 * clearance) / ext[2]])
    paths, volume = [], 0.0
    for i, f in enumerate(sorted(PARTS.glob("servo_*.obj"))):
        p = trimesh.load(f)
        p.vertices = (p.vertices - centre) * scale + centre
        volume += p.volume
        paths.append(Path(tmp) / f"servo_{i}.obj")
        p.export(paths[-1])
    return paths, volume, scale


def look_at(pos, target):
    """SAPIEN camera pose: x forward, y left, z up."""
    fwd = np.asarray(target, float) - pos
    fwd /= np.linalg.norm(fwd)
    left = np.cross([0.0, 0.0, 1.0], fwd)
    left /= np.linalg.norm(left)
    q = Rotation.from_matrix(np.stack([fwd, left, np.cross(fwd, left)], axis=1)).as_quat()
    return sapien.Pose(pos, [q[3], q[0], q[1], q[2]])


def build_scene(clearance, info, tmp):
    sapien.physx.set_scene_config(enable_tgs=PHYSX["tgs"], enable_pcm=True)
    sapien.physx.set_shape_config(contact_offset=PHYSX["contact_offset"], rest_offset=0.0)
    sapien.physx.set_body_config(solver_position_iterations=PHYSX["position_iterations"], solver_velocity_iterations=2)
    sapien.physx.set_default_material(static_friction=0.4, dynamic_friction=0.4, restitution=0.0)
    scene = sapien.Scene()
    scene.set_timestep(TIMESTEP)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([-0.4, 0.4, -0.8], [0.9, 0.9, 0.9])

    b = scene.create_actor_builder()
    for f in sorted(PARTS.glob("base_[0-9]*.obj")):
        b.add_convex_collision_from_file(str(f))
    b.add_visual_from_file(str(PARTS / "base_crop.obj"), material=sapien.render.RenderMaterial(base_color=[1, 0.82, 0.12, 1]))
    b.build_static(name="base")

    paths, volume, scale = servo_pieces(clearance, tmp)
    b = scene.create_actor_builder()
    for p in paths:
        b.add_convex_collision_from_file(str(p), density=info["servo_mass_kg"] / volume)
    b.add_visual_from_file(str(PARTS / "servo.obj"), scale=scale.tolist(),
                           material=sapien.render.RenderMaterial(base_color=[0.1, 0.1, 0.1, 1]))
    servo = b.build(name="servo")
    servo.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).set_max_depenetration_velocity(PHYSX["depenetration"])

    cam = scene.add_camera("view", 640, 480, np.radians(45), 0.01, 10)
    cam.entity.set_pose(look_at(CAM_POS, info["seated_servo_pos"]))
    return scene, servo, cam


def snapshot(scene, cam, path):
    scene.update_render()
    cam.take_picture()
    rgb = (np.clip(cam.get_picture("Color")[..., :3], 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def run_trial(scene, servo, cam, info, rng, condition, snap=None):
    seated = np.array(info["seated_servo_pos"])
    err_pos, err_rot = hand.trial_error(rng, condition)
    body = servo.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    pos0, rot0 = hand.target_pose(0.0, seated, err_pos, err_rot)
    q = rot0.as_quat()
    servo.set_pose(sapien.Pose(pos0, [q[3], q[0], q[1], q[2]]))
    body.set_linear_velocity([0, 0, 0])
    body.set_angular_velocity([0, 0, 0])
    if snap:
        snapshot(scene, cam, f"{snap}_start.png")

    steps = int((hand.START_OUT / hand.SPEED + hand.SETTLE) / TIMESTEP)
    max_pen, unstable = 0.0, False
    for k in range(steps):
        target_pos, target_rot = hand.target_pose(k * TIMESTEP, seated, err_pos, err_rot)
        pose = servo.pose
        rot = Rotation.from_quat([*pose.q[1:], pose.q[0]])
        f, tau = hand.wrench(np.asarray(pose.p), rot, np.asarray(body.linear_velocity),
                             np.asarray(body.angular_velocity), target_pos, target_rot, body.mass)
        body.add_force_torque(f, tau)
        scene.step()
        for c in scene.get_contacts():
            for p in c.points:
                max_pen = max(max_pen, -float(p.separation))
        if not np.isfinite(servo.pose.p).all() or np.linalg.norm(body.linear_velocity) > 1.0:
            unstable = True
            break
    if snap:
        snapshot(scene, cam, f"{snap}_end.png")
    pose = servo.pose
    result = hand.outcome(np.asarray(pose.p), Rotation.from_quat([*pose.q[1:], pose.q[0]]), seated, unstable)
    return dict(result, max_pen_mm=max_pen * 1000, steps=k + 1)


def render_fps(scene, size, n=300):
    cam = scene.add_camera(f"fps{size[0]}", size[1], size[0], np.radians(45), 0.01, 10)
    cam.entity.set_pose(look_at(CAM_POS, [0.03, 0.0, 0.04]))
    t0 = time.perf_counter()
    for _ in range(n):
        scene.update_render()
        cam.take_picture()
        cam.get_picture("Color")
    return n / (time.perf_counter() - t0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = json.loads((PARTS / "parts.json").read_text())
    report = {"simulator": f"sapien {sapien.__version__} (ManiSkill's engine), PhysX CPU", "physx": PHYSX,
              "contact": "convex pieces (CoACD)", "clearances": {}}
    cases = [(c, cond) for c in CLEARANCES_MM for cond in hand.CONDITIONS]
    with tempfile.TemporaryDirectory() as tmp:
        for n, (c, cond) in enumerate(cases):
            scene, servo, cam = build_scene(c / 1000, info, tmp)
            rng = np.random.default_rng(0)
            trials, t0 = [], time.perf_counter()
            for i in range(TRIALS):
                trials.append(run_trial(scene, servo, cam, info, rng, cond,
                                        snap=OUT / f"maniskill_c{c}_{cond}" if i == 0 else None))
                print(f"FLEET_PROGRESS {n * TRIALS + i + 1}/{len(cases) * TRIALS} {c} mm {cond}: {trials[-1]}", flush=True)
            wall = time.perf_counter() - t0
            report["clearances"][f"{c} {cond}"] = {
                "success": sum(t["success"] for t in trials), "trials": TRIALS,
                "unstable": sum(t["unstable"] for t in trials),
                "max_penetration_mm": round(max(t["max_pen_mm"] for t in trials), 3),
                "median_pos_err_mm": round(float(np.median([t["pos_err_mm"] for t in trials])), 3),
                "physics_steps_per_s": round(sum(t["steps"] for t in trials) / wall),
            }
    report["render_fps_64x64"] = round(render_fps(scene, (64, 64)))
    report["render_fps_480x640"] = round(render_fps(scene, (480, 640)))
    (OUT / os.environ.get("REPORT", "maniskill.json")).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

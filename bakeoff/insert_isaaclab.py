"""Insertion test in Isaac Lab (Isaac Sim, PhysX 5 on the GPU): the same parts, hand law and trials
as bakeoff/insert_mujoco.py, but the base collides through an SDF of its exact mesh (no convex
decomposition), and the 100 trials of each case run in parallel as 100 environments.

Run (Linux, NVIDIA RTX GPU; accept the Isaac Sim EULA with OMNI_KIT_ACCEPT_EULA=YES):
    python bakeoff/insert_isaaclab.py [trials]
"""

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("trials", type=int, nargs="?", default=100)
args = parser.parse_args()
app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: E402

import hand  # noqa: E402

PARTS = Path("bakeoff/parts")
OUT = Path("bakeoff/results")
CLEARANCES_MM = [0.0, 0.2, 0.4]
TIMESTEP = 0.0005
SPACING = 0.3  # m between environments


def add_mesh(stage, path, mesh, sdf):
    """A collision mesh prim: exact SDF collision (sdf=True) or its convex hull."""
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.CreatePointsAttr([Gf.Vec3f(*v) for v in mesh.vertices.astype(float)])
    prim.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    prim.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    coll = PhysxSchema.PhysxCollisionAPI.Apply(prim.GetPrim())
    coll.CreateContactOffsetAttr(0.002)
    coll.CreateRestOffsetAttr(0.0)
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim())
    if sdf:
        mesh_api.CreateApproximationAttr("sdf")
        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim.GetPrim()).CreateSdfResolutionAttr(512)
    else:
        mesh_api.CreateApproximationAttr("convexHull")


def build_env(stage, i, clearance, info):
    """Environment i: a static base (exact mesh, SDF) and a servo (its convex pieces)."""
    origin = np.array([(i % 10) * SPACING, (i // 10) * SPACING, 0.0])
    root = f"/World/envs/env_{i}"
    UsdGeom.Xform.Define(stage, root).AddTranslateOp().Set(Gf.Vec3d(*origin))
    add_mesh(stage, f"{root}/Base/mesh", trimesh.load(PARTS / "base_crop.obj"), sdf=True)

    servo = trimesh.load(PARTS / "servo.obj")
    ext, centre = servo.extents, servo.bounds.mean(axis=0)
    scale = np.array([1.0, (ext[1] - 2 * clearance) / ext[1], (ext[2] - 2 * clearance) / ext[2]])
    body = UsdGeom.Xform.Define(stage, f"{root}/Servo")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(info["servo_mass_kg"])
    for k, f in enumerate(sorted(PARTS.glob("servo_*.obj"))):
        p = trimesh.load(f)
        p.vertices = (p.vertices - centre) * scale + centre
        add_mesh(stage, f"{root}/Servo/piece_{k}", p, sdf=False)
    return origin


def torch_rot(q_wxyz):
    """Rotation matrices (n, 3, 3) from quaternions (n, 4), w first."""
    w, x, y, z = q_wxyz.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


def rotvec(R):
    """Rotation vectors (n, 3) of rotation matrices (n, 3, 3)."""
    angle = torch.arccos(((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    axis = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    return axis * (angle / (2 * torch.sin(angle).clamp_min(1e-9))).unsqueeze(-1)


def run_case(clearance, condition, info, n):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=TIMESTEP, device="cuda:0"))
    stage = sim.stage
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
    origins = torch.tensor(np.stack([build_env(stage, i, clearance, info) for i in range(n)]),
                           dtype=torch.float32, device="cuda")
    servo = RigidObject(RigidObjectCfg(prim_path="/World/envs/env_.*/Servo", spawn=None))
    sim.reset()

    rng = np.random.default_rng(0)
    errs = [hand.trial_error(rng, condition) for _ in range(n)]
    seated = torch.tensor(info["seated_servo_pos"], device="cuda")
    err_pos = torch.tensor(np.stack([e[0] for e in errs]), dtype=torch.float32, device="cuda")
    err_rot = torch.tensor(np.stack([e[1].as_matrix() for e in errs]), dtype=torch.float32, device="cuda")
    q_err = np.stack([e[1].as_quat()[[3, 0, 1, 2]] for e in errs])

    state = servo.data.default_root_state.clone()
    state[:, :3] = origins + seated + torch.tensor([hand.START_OUT, 0, 0], device="cuda") + err_pos
    state[:, 3:7] = torch.tensor(q_err, dtype=torch.float32, device="cuda")
    state[:, 7:] = 0.0
    servo.write_root_state_to_sim(state)

    gains = torch.tensor([hand.K_AXIAL, hand.K_LATERAL, hand.K_LATERAL], device="cuda")
    mass = info["servo_mass_kg"]
    steps = int((hand.START_OUT / hand.SPEED + hand.SETTLE) / TIMESTEP)
    unstable = torch.zeros(n, dtype=torch.bool, device="cuda")
    t0 = time.perf_counter()
    for k in range(steps):
        x = hand.START_OUT * max(0.0, 1.0 - k * TIMESTEP * hand.SPEED / hand.START_OUT)
        target = origins + seated + torch.tensor([x, 0, 0], device="cuda") + err_pos
        pos, quat = servo.data.root_pos_w, servo.data.root_quat_w
        f = gains * (target - pos) - hand.D_LIN * servo.data.root_lin_vel_w
        f[:, 0] = f[:, 0].clamp(-hand.F_AXIAL_MAX, hand.F_AXIAL_MAX)
        f[:, 2] += mass * hand.GRAVITY
        R = torch_rot(quat)
        tau = hand.K_ANG * rotvec(err_rot @ R.transpose(1, 2)) - hand.D_ANG * servo.data.root_ang_vel_w
        # forces in the body frame (Isaac Lab applies external wrenches in the body frame)
        servo.set_external_force_and_torque((R.transpose(1, 2) @ f.unsqueeze(-1)).squeeze(-1).unsqueeze(1),
                                            (R.transpose(1, 2) @ tau.unsqueeze(-1)).squeeze(-1).unsqueeze(1))
        servo.write_data_to_sim()
        sim.step(render=False)
        servo.update(TIMESTEP)
        unstable |= ~torch.isfinite(servo.data.root_pos_w).all(-1) | (servo.data.root_lin_vel_w.norm(dim=-1) > 1.0)
    wall = time.perf_counter() - t0

    pos_err = (servo.data.root_pos_w - origins - seated).norm(dim=-1)
    ang = rotvec(torch_rot(servo.data.root_quat_w)).norm(dim=-1)
    success = (~unstable) & (pos_err < 0.001) & (ang < np.radians(2.0))
    result = {
        "success": int(success.sum()), "trials": n, "unstable": int(unstable.sum()),
        "max_penetration_mm": None,  # not measured: PhysX contact reports are not read here
        "median_pos_err_mm": round(float(pos_err.median()) * 1000, 3),
        "physics_steps_per_s_all_envs": round(steps / wall),
        "env_steps_per_s": round(steps * n / wall),
    }
    sim.clear_instance()
    return result


def render_fps(size, n=300):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=TIMESTEP, device="cuda:0"))
    add_mesh(sim.stage, "/World/Base/mesh", trimesh.load(PARTS / "base_crop.obj"), sdf=False)
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
    cam = TiledCamera(TiledCameraCfg(prim_path="/World/Camera", height=size[0], width=size[1], data_types=["rgb"],
                                     spawn=sim_utils.PinholeCameraCfg(), offset=TiledCameraCfg.OffsetCfg(
                                         pos=(0.16, -0.20, 0.16), rot=(1, 0, 0, 0), convention="world")))
    sim.reset()
    t0 = time.perf_counter()
    for _ in range(n):
        sim.step(render=True)
        cam.update(TIMESTEP)
        cam.data.output["rgb"].cpu()
    fps = n / (time.perf_counter() - t0)
    sim.clear_instance()
    return fps


def main():
    info = json.loads((PARTS / "parts.json").read_text())
    report = {"simulator": "isaac lab (isaac sim, physx 5 gpu)", "contact": "base: exact mesh SDF; servo: convex pieces",
              "clearances": {}}
    for c in CLEARANCES_MM:
        for cond in hand.CONDITIONS:
            report["clearances"][f"{c} {cond}"] = r = run_case(c / 1000, cond, info, args.trials)
            print(f"{c} mm {cond}: {r}", flush=True)
    report["render_fps_64x64"] = round(render_fps((64, 64)))
    report["render_fps_480x640"] = round(render_fps((480, 640)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "isaaclab.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
    app.close()

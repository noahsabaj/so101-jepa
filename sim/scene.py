"""The SO-101 scene in MuJoCo: the official arm model, a table, a cube, a scene camera and a wrist
camera, plus the domain randomization of each episode.

Changes to the official model (assets/so101/so101_new_calib_camera.xml):
- The jaw meshes do not collide: the convex hull of the fixed jaw fills the gap between the
  fingers. Box pads on the inner finger faces (measured from the meshes) take their place.
- A wrist camera looks from the wrist camera module at the space between the fingers.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

MODEL = str(Path(__file__).resolve().parents[1] / "assets/so101/so101_new_calib_camera.xml")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IMAGE = 64  # each view is IMAGE x IMAGE; the observation is the scene and wrist views side by side
CUBE_HALF = 0.0125

# Pads in the gripperframe site frame (x along the fingers, z across the jaws, at gripper = 0 rad):
# (centre, half sizes) in m. The finger faces are at z = -2..0 mm (fixed) and z = 16..20 mm (moving).
FIXED_PAD = ([-0.020, 0.0, -0.0040], [0.025, 0.007, 0.0035])
MOVING_PAD = ([-0.020, 0.0, 0.0210], [0.025, 0.007, 0.0035])
SCENE_CAM_POS, SCENE_CAM_TARGET = np.array([0.46, 0.0, 0.36]), np.array([0.17, 0.0, 0.02])
WRIST_CAM_FORWARD = 0.025  # m: the wrist camera sits at the front of its module, not inside it


def _frame(m, d, kind, name):
    """World position and rotation matrix of a body or site."""
    if kind == "site":
        i = m.site(name).id
        return d.site_xpos[i].copy(), d.site_xmat[i].reshape(3, 3).copy()
    i = m.body(name).id
    return d.xpos[i].copy(), d.xmat[i].reshape(3, 3).copy()


def _local(parent, child):
    """Pose of `child` (pos, R) in the frame of `parent` (pos, R), as MuJoCo pos and quat (wxyz)."""
    return (parent[1].T @ (child[0] - parent[0])).tolist(), _quat_wxyz(parent[1].T @ child[1])


def _look_at(cam_pos, target, up):
    """Rotation matrix of a MuJoCo camera (it looks along its -z axis) at `target`."""
    z = cam_pos - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def build_spec():
    spec = mujoco.MjSpec.from_file(MODEL)
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.global_.offwidth = spec.visual.global_.offheight = 640
    for geom in spec.geoms:
        if geom.meshname in ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1") and geom.group == 3:
            geom.contype = geom.conaffinity = 0

    world = spec.worldbody
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.45, 0.45, 0.01],
                   pos=[0.20, 0.0, -0.01], rgba=[0.6, 0.55, 0.5, 1], friction=[0.8, 0.005, 0.0001])
    cube = world.add_body(name="cube", pos=[0.22, 0.0, CUBE_HALF])
    cube.add_freejoint(name="cube")
    cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, mass=0.02,
                  rgba=[0.8, 0.1, 0.1, 1], friction=[1.0, 0.005, 0.0001], condim=4)
    world.add_light(name="key", pos=[0.3, -0.3, 0.9], dir=[-0.1, 0.3, -0.9], diffuse=[0.7] * 3, castshadow=True)
    world.add_camera(name="scene", fovy=55, pos=SCENE_CAM_POS.tolist(),
                     quat=_quat_wxyz(_look_at(SCENE_CAM_POS, SCENE_CAM_TARGET, np.array([0.0, 0.0, 1.0]))))

    # Place the pads and the wrist camera from the arm's frames at its reference pose.
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site = _frame(model, data, "site", "gripperframe")

    def site_box(centre):
        return site[0] + site[1] @ np.asarray(centre), site[1]

    for body_name, (centre, half), name in [("gripper", FIXED_PAD, "fixed_pad"),
                                            ("moving_jaw_so101_v1", MOVING_PAD, "moving_pad")]:
        pos, quat = _local(_frame(model, data, "body", body_name), site_box(centre))
        spec.body(body_name).add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos, quat=quat,
                                      rgba=[0.2, 0.2, 0.2, 1], friction=[1.2, 0.005, 0.0001], condim=4,
                                      solref=[0.005, 1.0], mass=0.001)
    cam_body = _frame(model, data, "body", "wrist_camera")
    grasp_point = site[0] + site[1] @ np.array([-0.020, 0.0, 0.010])
    cam_rot = _look_at(cam_body[0], grasp_point, up=site[1][:, 2])
    cam_pos = cam_body[0] - cam_rot[:, 2] * WRIST_CAM_FORWARD  # a camera looks along its -z axis
    pos, quat = _local(cam_body, (cam_pos, cam_rot))
    spec.body("wrist_camera").add_camera(name="wrist", pos=pos, quat=quat, fovy=75)
    return spec


def _quat_wxyz(rot):
    q = Rotation.from_matrix(rot).as_quat()
    return [q[3], q[0], q[1], q[2]]


@dataclass
class Domain:
    """The random parts of one episode (domain randomization)."""
    cam_pos: np.ndarray
    cam_target: np.ndarray
    light_dir: np.ndarray
    light_level: float
    table_rgb: np.ndarray
    cube_rgb: np.ndarray
    cube_friction: float
    joint_offset: np.ndarray  # rad, added to every command: calibration error and backlash
    kp_scale: float


def sample_domain(rng) -> Domain:
    return Domain(
        cam_pos=SCENE_CAM_POS + rng.uniform(-0.02, 0.02, 3),
        cam_target=SCENE_CAM_TARGET + rng.uniform(-0.02, 0.02, 3),
        light_dir=np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.1, 0.6), -1.0]),
        light_level=rng.uniform(0.45, 0.9),
        table_rgb=rng.uniform(0.35, 0.8, 3),
        cube_rgb=np.array([0.8, 0.1, 0.1]) + rng.uniform(-0.1, 0.1, 3),
        cube_friction=rng.uniform(0.7, 1.2),
        joint_offset=np.radians(rng.uniform(-0.45, 0.45, 6)),
        kp_scale=rng.uniform(0.8, 1.2),
    )


def apply_domain(model, domain: Domain, base_kp):
    cam = model.camera("scene")
    cam.pos[:] = domain.cam_pos
    cam.quat[:] = _quat_wxyz(_look_at(domain.cam_pos, domain.cam_target, up=np.array([0.0, 0.0, 1.0])))
    light = model.light("key")
    light.dir[:] = domain.light_dir / np.linalg.norm(domain.light_dir)
    light.diffuse[:] = domain.light_level
    model.geom("table").rgba[:3] = domain.table_rgb
    model.geom("cube").rgba[:3] = np.clip(domain.cube_rgb, 0, 1)
    model.geom("cube").friction[0] = domain.cube_friction
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = base_kp[i] * domain.kp_scale
        model.actuator_biasprm[i, 1] = -base_kp[i] * domain.kp_scale

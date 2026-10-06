"""The compliant hand of the insertion test, shared by the three simulators.

The hand is a force law, not a body: each physics step it applies a wrench to the servo's centre
of mass that pulls it toward a target pose. Across the insertion axis the spring is soft, so the
pocket walls can correct the hand's pose error. Along the axis the push is limited to 10 N.
"""

import numpy as np
from scipy.spatial.transform import Rotation

START_OUT = 0.020  # m: the servo starts this far out of the pocket, along +x of the base
SPEED = 0.020  # m/s: the target moves into the pocket at this speed
SETTLE = 1.0  # s: time after the motion
# Hand pose error, uniform per axis: position across the insertion axis (y, z), rotation (x, y, z).
# "aligned": a good simulator must insert; "perturbed": a stress test of contact stability.
CONDITIONS = {"aligned": (0.0001, 0.1), "perturbed": (0.0015, 2.0)}  # (m, deg)

K_AXIAL, K_LATERAL, D_LIN = 1000.0, 300.0, 4.0  # N/m, N/m, N s/m
K_ANG, D_ANG = 0.5, 0.005  # N m/rad, N m s/rad
F_AXIAL_MAX = 5.0  # N
GRAVITY = 9.81


def trial_error(rng, condition):
    """The hand's belief error for one trial: a position offset (y, z) and a rotation."""
    pos_err, ang_err = CONDITIONS[condition]
    pos = np.array([0.0, *rng.uniform(-pos_err, pos_err, 2)])
    rot = Rotation.from_euler("xyz", rng.uniform(-ang_err, ang_err, 3), degrees=True)
    return pos, rot


def target_pose(t, seated_pos, err_pos, err_rot):
    """Target position (m) and rotation of the servo at time t (s)."""
    move_t = START_OUT / SPEED
    x = START_OUT * max(0.0, 1.0 - t / move_t)
    return np.asarray(seated_pos) + [x, 0.0, 0.0] + err_pos, err_rot


def wrench(pos, rot, lin_vel, ang_vel, target_pos, target_rot, mass):
    """Force (N) and torque (N m), world frame, applied at the servo's centre of mass."""
    e = target_pos - pos
    f = np.array([K_AXIAL * e[0], K_LATERAL * e[1], K_LATERAL * e[2]]) - D_LIN * lin_vel
    f[0] = np.clip(f[0], -F_AXIAL_MAX, F_AXIAL_MAX)
    f[2] += mass * GRAVITY  # the hand carries the servo's weight
    e_rot = (target_rot * rot.inv()).as_rotvec()
    tau = K_ANG * e_rot - D_ANG * ang_vel
    return f, tau


def outcome(pos, rot, seated_pos, unstable):
    """Success: within 1 mm and 2 degrees of the seated pose (the seated rotation is identity)."""
    pos_err = float(np.linalg.norm(pos - np.asarray(seated_pos)))
    ang_err = float(np.degrees(rot.magnitude()))
    return dict(success=bool(not unstable and pos_err < 0.001 and ang_err < 2.0),
                unstable=bool(unstable), pos_err_mm=pos_err * 1000, ang_err_deg=ang_err)

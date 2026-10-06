"""Scripted expert: inverse kinematics and long play episodes (pick-and-place, reach, free motion).

The expert moves the joint targets toward waypoints with a speed limit and action noise. It sees
the true state (it is the data source, not the controller under test).
"""

import mujoco
import numpy as np

from env import MAX_ACTION, REST, SO101Env, sample_cube_xy
from scene import CUBE_HALF

# The grasp point in the gripperframe site frame (x along the fingers, z across the jaws): the
# centre of a 25 mm cube that rests on the fixed pad face, at the middle of the pads' length.
GRASP_POINT = np.array([-0.020, 0.0, 0.012])
OPEN, CLOSED = 0.55, -0.15  # gripper joint targets (rad); CLOSED squeezes the cube
DOWN = np.array([0.0, 0.0, -1.0])
# The pad tips are 25 mm past the grasp point, so the grasp point cannot go below 25 mm over the
# table. Grasp with the tips 3 mm over the table (grasp point 15.5 mm over the cube centre), and
# place with the held cube about 2.5 mm over the table.
GRASP_DZ, PLACE_DZ = 0.0155, 0.018


def ik(env: SO101Env, point, finger_dir, jaw_dir, q0, iters=80):
    """Arm joints (5) that put GRASP_POINT at `point` with the fingers along `finger_dir` and the
    jaw axis along `jaw_dir`; damped least squares on a scratch copy of the model data."""
    m, d = env.model, mujoco.MjData(env.model)
    d.qpos[:] = env.data.qpos
    q = np.array(q0, float)
    arm = env.qadr[:5]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    body = m.site_bodyid[env.site]
    for _ in range(iters):
        d.qpos[arm] = q[:5]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        pos, rot = d.site_xpos[env.site], d.site_xmat[env.site].reshape(3, 3)
        p = pos + rot @ GRASP_POINT
        e_pos = point - p
        e_rot = 0.5 * (np.cross(rot[:, 0], finger_dir) + np.cross(rot[:, 2], jaw_dir))
        mujoco.mj_jac(m, d, jacp, jacr, p, body)
        J = np.vstack([jacp[:, env.vadr[:5]], 0.3 * jacr[:, env.vadr[:5]]])
        e = np.concatenate([e_pos, 0.3 * e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q[:5] = np.clip(q[:5] + np.clip(dq, -0.2, 0.2), env.ctrl_lo[:5], env.ctrl_hi[:5])
        if np.linalg.norm(e_pos) < 5e-4 and np.linalg.norm(e_rot) < 0.02:
            break
    return q, float(np.linalg.norm(e_pos))


def jaw_dir_for(yaw):
    """A horizontal jaw axis that closes across the cube faces (cube yaw folded to +-45 deg)."""
    a = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return np.array([np.cos(a), np.sin(a), 0.0])


class Expert:
    """Plays one long episode: tasks follow each other until the episode ends."""

    def __init__(self, env: SO101Env, rng, noise=0.02, grasp_dz=GRASP_DZ, place_dz=PLACE_DZ):
        self.env, self.rng, self.noise = env, rng, noise
        self.grasp_dz, self.place_dz = grasp_dz, place_dz
        self.plan = []  # list of (joint target, steps allowed, phase name)
        self.phase = "start"

    def waypoint(self, point, yaw, gripper, steps, phase):
        q, err = ik(self.env, point, DOWN, jaw_dir_for(yaw), self.plan[-1][0] if self.plan else self.env.joints())
        q[5] = gripper
        self.plan.append((q, steps, phase))
        return err

    def pick_and_place(self):
        cube = self.env.cube_pose()
        xy, yaw = cube[:2], 2 * np.arctan2(cube[6], cube[3])
        goal = sample_cube_xy(self.rng)
        while np.linalg.norm(goal - xy) < 0.06:
            goal = sample_cube_xy(self.rng)
        z = CUBE_HALF
        self.waypoint([*xy, z + 0.08], yaw, OPEN, 12, "approach")
        self.waypoint([*xy, z + self.grasp_dz], yaw, OPEN, 8, "descend")
        self.waypoint([*xy, z + self.grasp_dz], yaw, CLOSED, 5, "grasp")
        self.waypoint([*xy, z + 0.10], yaw, CLOSED, 8, "lift")
        place_yaw = yaw + self.rng.uniform(-0.5, 0.5)
        self.waypoint([*goal, z + 0.10], place_yaw, CLOSED, 12, "carry")
        self.waypoint([*goal, z + self.place_dz], place_yaw, CLOSED, 8, "lower")
        self.waypoint([*goal, z + self.place_dz], place_yaw, OPEN, 4, "release")
        self.waypoint([*goal, z + 0.09], place_yaw, OPEN, 6, "retreat")

    def reach(self):
        xy = sample_cube_xy(self.rng)
        z = self.rng.uniform(0.04, 0.20)
        self.waypoint([*xy, z], self.rng.uniform(-1.0, 1.0), self.rng.uniform(CLOSED, OPEN), 12, "reach")

    def free_motion(self):
        q = self.env.joints() if not self.plan else self.plan[-1][0]
        for _ in range(self.rng.integers(2, 5)):
            q = np.clip(q + self.rng.normal(0, 0.25, 6), self.env.ctrl_lo, self.env.ctrl_hi)
            q[1:4] = np.clip(q[1:4], REST[1:4] - 0.9, REST[1:4] + 0.9)
            self.plan.append((q.copy(), 4, "free"))

    def next_task(self):
        r = self.rng.uniform()
        if r < 0.6:
            self.pick_and_place()
        elif r < 0.85:
            self.reach()
        else:
            self.free_motion()

    def act(self):
        """The next action (joint target change relative to the measured joints)."""
        while not self.plan:
            self.next_task()
        target, steps, self.phase = self.plan[0]
        q = self.env.joints()
        action = np.clip(target - q, -MAX_ACTION * 0.6, MAX_ACTION * 0.6)
        action += self.rng.normal(0, self.noise, 6)
        steps -= 1
        if steps <= 0 or np.abs(target - q).max() < 0.03:  # a closing gripper waits out its steps
            self.plan.pop(0)
        else:
            self.plan[0] = (target, steps, self.phase)
        return action

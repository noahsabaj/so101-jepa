"""SO-101 environment: 5 Hz control, scene and wrist views, joint-space actions.

An action is the change of the 6 joint targets (rad) relative to the measured joints. During the
0.2 s step the targets move linearly to measured + action (as a teleoperation stream would), and
the per-episode joint offset of the domain (calibration error, backlash) is added to them.
"""

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from scene import CUBE_HALF, IMAGE, JOINTS, Domain, apply_domain, build_spec, sample_domain

CONTROL_DT = 0.2
REST = np.array([0.0, -1.2, 1.1, 1.1, 0.0, 0.3])  # joints: arm up and back, fingers half open
MAX_ACTION = np.array([0.25, 0.25, 0.25, 0.3, 0.4, 0.4])  # rad per step


class SO101Env:
    def __init__(self, randomize=True):
        self.randomize = randomize
        self.model = build_spec().compile()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, IMAGE, IMAGE)
        self.qadr = np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self.vadr = np.array([self.model.joint(j).dofadr[0] for j in JOINTS])
        self.cube_qadr = self.model.joint("cube").qposadr[0]
        self.site = self.model.site("gripperframe").id
        self.base_kp = self.model.actuator_gainprm[:, 0].copy()
        self.ctrl_lo, self.ctrl_hi = self.model.actuator_ctrlrange.T.copy()
        self.substeps = int(round(CONTROL_DT / self.model.opt.timestep))
        self.domain: Domain | None = None

    # --- state -------------------------------------------------------------------------------
    def joints(self):
        return self.data.qpos[self.qadr].copy()

    def cube_pose(self):
        return self.data.qpos[self.cube_qadr:self.cube_qadr + 7].copy()

    def site_pose(self):
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def get_state(self):
        return dict(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(), ctrl=self.data.ctrl.copy(),
                    target=self.target.copy())

    def set_state(self, state):
        self.data.qpos[:], self.data.qvel[:] = state["qpos"], state["qvel"]
        self.data.ctrl[:], self.target = state["ctrl"], state["target"].copy()
        mujoco.mj_forward(self.model, self.data)

    # --- episode -----------------------------------------------------------------------------
    def reset(self, seed):
        """Start an episode. The domain, the cube pose and the joints come from `seed`."""
        rng = np.random.default_rng(seed)
        self.domain = sample_domain(rng) if self.randomize else None
        if self.domain is not None:
            apply_domain(self.model, self.domain, self.base_kp)
        mujoco.mj_resetData(self.model, self.data)
        cube_xy = sample_cube_xy(rng)
        self.set_cube(cube_xy, rng.uniform(-np.pi / 4, np.pi / 4))
        q = REST + rng.uniform(-0.1, 0.1, 6)
        self.data.qpos[self.qadr] = q
        self.target = q.copy()
        self.data.ctrl[:] = self._command(q)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.substeps):  # let the cube and the servos settle
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    def set_cube(self, xy, yaw, z=CUBE_HALF):
        a = self.cube_qadr
        self.data.qpos[a:a + 3] = [xy[0], xy[1], z]
        self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qvel[self.model.joint("cube").dofadr[0]:][:6] = 0.0

    def _command(self, target):
        offset = self.domain.joint_offset if self.domain is not None else 0.0
        return np.clip(target + offset, self.ctrl_lo, self.ctrl_hi)

    def step(self, action):
        action = np.clip(action, -MAX_ACTION, MAX_ACTION)
        start = self.target
        self.target = np.clip(self.joints() + action, self.ctrl_lo, self.ctrl_hi)
        for k in range(self.substeps):
            self.data.ctrl[:] = self._command(start + (self.target - start) * (k + 1) / self.substeps)
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    # --- observation -------------------------------------------------------------------------
    def render(self):
        """Scene view and wrist view side by side: uint8 (IMAGE, 2 * IMAGE, 3).

        The wrist view renders on a second GL context in a worker thread while this thread renders
        the scene view (mjr_render releases the GIL). Both scenes are updated here, as
        mjv_updateScene may use the MjData stack. The pixels are the same as rendering the two views
        one after the other; 1.07x faster on llvmpipe (RENDER_HILLCLIMB.md)."""
        if getattr(self, "_wrist", None) is None:
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wrist-render")
            # made in the worker thread, so that its GL context is current there
            self._wrist = pool, pool.submit(mujoco.Renderer, self.model, IMAGE, IMAGE).result()
        pool, wrist = self._wrist
        self.renderer.update_scene(self.data, camera="scene")
        wrist.update_scene(self.data, camera="wrist")
        job = pool.submit(wrist.render)
        views = [self.renderer.render(), job.result()]
        return np.concatenate(views, axis=1)

    def observe(self):
        return dict(pixels=self.render(), proprio=self.joints().astype(np.float32))


def sample_cube_xy(rng):
    """A cube position in the band in front of the arm where top-down grasps are reliable. Past
    26 cm the arm is almost straight and the expert lifts the cube in under half of its tries."""
    r, a = rng.uniform(0.15, 0.25), rng.uniform(-np.pi / 3, np.pi / 3)
    return np.array([r * np.cos(a), r * np.sin(a)])

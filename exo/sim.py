"""MuJoCo wrapper: treadmill gait world with a physics-driven affected (right) knee."""
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import mujoco

from .config import MODEL_XML, GaitConfig, SimConfig, clamp
from .gait import FOOT_FLAT, HEEL_OFF, TOE_OFF, GaitReference, right_weight_share

SERVO_ACTUATORS = {"l_hip": "l_hip_servo", "l_knee": "l_knee_servo", "l_ankle": "l_ankle_servo",
                   "r_hip": "r_hip_servo", "r_ankle": "r_ankle_servo",
                   "l_shoulder": "l_arm_servo", "r_shoulder": "r_arm_servo"}
_MUSCLE_IDLE = np.array([0.60, 0.42, 0.45, 1.0])
_MUSCLE_ACTIVE = np.array([1.0, 0.10, 0.06, 1.0])
_MOTOR_IDLE = np.array([0.05, 0.55, 0.95, 1.0])
_MOTOR_HOT = np.array([1.0, 0.55, 0.05, 1.0])

# An OpenGL context belongs to one thread, but Streamlit reruns scripts on new threads.
# All rendering therefore goes through this single persistent GL worker thread.
_GL_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco-gl")
_WITH_RENDERER: "weakref.WeakSet[GaitSim]" = weakref.WeakSet()


def _close_all_renderers():
    """Free GL contexts on their own thread before the interpreter shuts the worker down."""
    for sim in list(_WITH_RENDERER):
        try:
            _GL_THREAD.submit(sim._close).result(timeout=5)
        except Exception:
            pass


threading._register_atexit(_close_all_renderers)   # runs before the executor's own exit hook


def _free_renderer(r: mujoco.Renderer):
    """Close a renderer without breaking the others on the GL thread.

    mujoco.Renderer.close() destroys its GL context first and frees its MjrContext afterwards, while
    another renderer's context is current. Object ids are per context, so that free deletes the OTHER
    renderer's framebuffer and every later frame it renders is black. Free the MjrContext in its own
    context first."""
    ctx, mjr = getattr(r, "_gl_context", None), getattr(r, "_mjr_context", None)
    if ctx is not None and mjr is not None:
        ctx.make_current()
        mjr.free()
        r._mjr_context = None
    r.close()


class GaitSim:
    def __init__(self, sim_cfg: SimConfig | None = None, gait_cfg: GaitConfig | None = None,
                 xml_path=MODEL_XML):
        self.cfg = sim_cfg or SimConfig()
        self.gait_cfg = gait_cfg or GaitConfig()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = self.cfg.dt
        self.data = mujoco.MjData(self.model)
        self.ref = GaitReference(self.gait_cfg.stride_period)

        m = self.model
        self.servo_act = {j: m.actuator(a).id for j, a in SERVO_ACTUATORS.items()}
        self.qadr = {j.name: m.jnt_qposadr[j.id] for j in (m.joint(i) for i in range(m.njnt))}
        self.dadr = {j.name: m.jnt_dofadr[j.id] for j in (m.joint(i) for i in range(m.njnt))}
        self.knee_dof = self.dadr["r_knee"]
        self.act_patient = m.actuator("patient_knee").id
        self.act_exo = m.actuator("exo_knee").id
        self.pelvis_mocap = m.body_mocapid[m.body("pelvis").id]
        self.stripe_mocap = [m.body_mocapid[m.body(f"stripe{i}").id] for i in range(9)]
        self.site = {n: m.site(n).id for n in ("r_heel", "r_toe", "l_heel", "l_toe")}
        self.body = {n: m.body(n).id for n in ("r_thigh", "r_shank", "r_foot", "l_thigh", "l_shank", "l_foot",
                                               "pelvis", "l_upperarm", "r_upperarm")}
        self.ten_quad = m.tendon("quadriceps").id
        self.ten_ham = m.tendon("hamstrings").id
        self.geom_motor = m.geom("exo_motor").id
        # computed-torque servo gains (force = ctrl - kp*q - kv*qd)
        self.servo_kp = {j: -float(m.actuator_biasprm[a][1]) for j, a in self.servo_act.items()}
        self.servo_kv = {j: -float(m.actuator_biasprm[a][2]) for j, a in self.servo_act.items()}

        self.id_table = self._inverse_dynamics_table()
        self.belt_speed = self._estimate_belt_speed()
        self.phase = 0.0
        self._renderer = None
        self._render_size = None
        self._drive = (0.0, 0.0, 0.0)   # quad, ham, exo colour drive in [0,1]
        self.reset()

    # ------------------------------------------------------------ setup
    ID_SAMPLES = 400

    def _inverse_dynamics_table(self) -> dict:
        """Joint torques needed to follow the normative gait (mj_inverse), vs phase.
        Used as servo feedforward and as the patient's internal model of the knee."""
        d = mujoco.MjData(self.model)
        joints = list(self.servo_act) + ["r_knee"]
        table = {j: np.zeros(self.ID_SAMPLES + 1) for j in joints}
        for i, p in enumerate(np.linspace(0, 1, self.ID_SAMPLES + 1)):
            for j, (q, qd, qdd) in self.ref.kinematics(p).items():
                d.qpos[self.qadr[j]], d.qvel[self.dadr[j]], d.qacc[self.dadr[j]] = q, qd, qdd
            mujoco.mj_inverse(self.model, d)
            for j in joints:
                table[j][i] = d.qfrc_inverse[self.dadr[j]]
        return {j: v.tolist() for j, v in table.items()}

    def id_torque(self, joint: str, phase: float) -> float:
        tab = self.id_table[joint]
        x = (phase % 1.0) * self.ID_SAMPLES
        i = int(x)
        return tab[i] + (x - i) * (tab[i + 1] - tab[i])

    def _estimate_belt_speed(self) -> float:
        """Belt speed so the stance foot is stationary on the belt (from normative kinematics)."""
        d = mujoco.MjData(self.model)
        phases = np.linspace(0.15, 0.45, 13)
        xs = []
        for p in phases:
            pose = self.ref.pose(p)
            for j in ("r_hip", "r_knee", "r_ankle"):
                d.qpos[self.qadr[j]] = pose[j]
            mujoco.mj_kinematics(self.model, d)
            xs.append(d.xpos[self.model.body("r_foot").id][0])
        slope = np.polyfit(phases * self.ref.stride_period, xs, 1)[0]
        return float(-slope)

    def reset(self, phase: float = 0.0):
        mujoco.mj_resetData(self.model, self.data)
        self.phase = phase
        for j, (q, qd) in self.ref.pose_full(phase).items():
            self.data.qpos[self.qadr[j]] = q
            self.data.qvel[self.dadr[j]] = qd
        self._apply_servo_targets(phase)
        self.data.mocap_pos[self.pelvis_mocap] = [0, 0, 1.0]
        for _ in range(6):                      # settle pelvis height onto the belt
            mujoco.mj_kinematics(self.model, self.data)
            self._update_pelvis(gain=1.0)
        for i, mid in enumerate(self.stripe_mocap):
            self.data.mocap_pos[mid] = [-0.85 + 0.21 * i, 0, self.gait_cfg.belt_top + 0.0013]
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------ sensors / state
    @property
    def time(self) -> float:
        return self.data.time

    @property
    def knee_angle(self) -> float:
        return float(self.data.qpos[self.qadr["r_knee"]])

    @property
    def knee_vel(self) -> float:
        return float(self.data.qvel[self.knee_dof])

    @property
    def thigh_angle(self) -> float:
        return float(self.data.qpos[self.qadr["r_hip"]])

    @property
    def thigh_vel(self) -> float:
        return float(self.data.qvel[self.dadr["r_hip"]])

    def foot_height(self, side: str = "r") -> float:
        sx = self.data.site_xpos
        return float(min(sx[self.site[f"{side}_heel"]][2], sx[self.site[f"{side}_toe"]][2]) - self.gait_cfg.belt_top)

    def toe_height(self, side: str = "r") -> float:
        return float(self.data.site_xpos[self.site[f"{side}_toe"]][2] - self.gait_cfg.belt_top)

    SCUFF_FROM = TOE_OFF + 0.03   # after the toe-off roll: the toe should be clearing the belt

    def toe_scuff(self, side: str = "r", depth: float = 0.02) -> bool:
        """Swing-phase toe drag (e.g. stiff-knee gait): toe more than `depth` below the belt."""
        p = (self.phase + (0.5 if side == "l" else 0.0)) % 1.0
        return p >= self.SCUFF_FROM and self.toe_height(side) < -depth

    def insole(self, side: str = "r") -> tuple[bool, bool]:
        """Ideal heel / toe insole switches: from the gait events of this leg (Perry timing),
        plus physics-based toe-scuff contacts during swing. Sensor latency, noise and
        bounce are added by the perception sensor model."""
        p = (self.phase + (0.5 if side == "l" else 0.0)) % 1.0
        heel = p < HEEL_OFF
        toe = FOOT_FLAT <= p < TOE_OFF or self.toe_scuff(side)
        return heel, toe

    def foot_contact(self, side: str = "r") -> bool:
        heel, toe = self.insole(side)
        return heel or toe

    def exo_torque(self) -> float:
        return float(self.data.actuator_force[self.act_exo])

    def patient_torque(self) -> float:
        return float(self.data.actuator_force[self.act_patient])

    def skeleton(self) -> dict:
        """3D keypoints for plotting (pelvis, hips, knees, ankles, heels, toes, shoulders)."""
        x, s = self.data.xpos, self.data.site_xpos
        b = self.body
        return {
            "pelvis": x[b["pelvis"]].copy(),
            "r_hip": x[b["r_thigh"]].copy(), "r_knee": x[b["r_shank"]].copy(), "r_ankle": x[b["r_foot"]].copy(),
            "r_heel": s[self.site["r_heel"]].copy(), "r_toe": s[self.site["r_toe"]].copy(),
            "l_hip": x[b["l_thigh"]].copy(), "l_knee": x[b["l_shank"]].copy(), "l_ankle": x[b["l_foot"]].copy(),
            "l_heel": s[self.site["l_heel"]].copy(), "l_toe": s[self.site["l_toe"]].copy(),
            "r_shoulder": x[b["r_upperarm"]].copy(), "l_shoulder": x[b["l_upperarm"]].copy(),
        }

    # ------------------------------------------------------------ stepping
    def _apply_servo_targets(self, phase: float):
        """Computed-torque control of the normative joints: PD on (q_ref, qd_ref) + inverse dynamics."""
        pose = self.ref.pose_full(phase)
        ctrl = self.data.ctrl
        for j, a in self.servo_act.items():
            q, qd = pose[j]
            ctrl[a] = self.servo_kp[j] * q + self.servo_kv[j] * qd + self.id_torque(j, phase)

    def _update_pelvis(self, gain: float = 0.5):
        """BWS harness: set the pelvis height so the loaded foot rests on the belt.
        The load is shared between feet in double support (Perry), so the weighted
        foot height is driven to zero; a buckling knee therefore lowers the body."""
        w = right_weight_share(self.phase)
        h = w * self.foot_height("r") + (1 - w) * self.foot_height("l")
        self.data.mocap_pos[self.pelvis_mocap][2] -= gain * h

    def _update_belt(self, elapsed: float):
        for mid in self.stripe_mocap:
            x = self.data.mocap_pos[mid][0] - self.belt_speed * elapsed
            self.data.mocap_pos[mid][0] = x + 1.89 if x < -0.945 else x

    def step(self, phase: float, tau_patient: float, tau_exo: float, tau_ext: float = 0.0, n: int = 1):
        """Advance n physics steps with constant inputs.

        phase: true gait phase driving the normative servos (hips, ankles, left leg).
        tau_patient / tau_exo: torques on the right knee (N·m, + = flexion).
        tau_ext: external stance load on the right knee from the (partial) body weight.
        """
        self.phase = phase
        self._apply_servo_targets(phase)
        c = self.data.ctrl
        c[self.act_patient] = tau_patient
        c[self.act_exo] = tau_exo
        self.data.qfrc_applied[self.knee_dof] = tau_ext
        mujoco.mj_step(self.model, self.data, nstep=n)
        mujoco.mj_kinematics(self.model, self.data)
        self._update_pelvis()
        self._update_belt(n * self.cfg.dt)

    # ------------------------------------------------------------ rendering
    def set_visual_drive(self, quad: float, ham: float, exo: float):
        """Colour drives in [0,1]: quadriceps & hamstring effort, exo motor torque magnitude."""
        self._drive = (clamp(float(quad), 0.0, 1.0), clamp(float(ham), 0.0, 1.0), clamp(float(exo), 0.0, 1.0))

    def apply_colors(self):
        q, h, e = self._drive
        m = self.model
        m.tendon_rgba[self.ten_quad] = (1 - q) * _MUSCLE_IDLE + q * _MUSCLE_ACTIVE
        m.tendon_rgba[self.ten_ham] = (1 - h) * _MUSCLE_IDLE + h * _MUSCLE_ACTIVE
        m.tendon_width[self.ten_quad] = 0.013 + 0.010 * q
        m.tendon_width[self.ten_ham] = 0.011 + 0.010 * h
        m.geom_rgba[self.geom_motor] = (1 - e) * _MOTOR_IDLE + e * _MOTOR_HOT

    def render(self, camera: str | None = None, width: int | None = None, height: int | None = None) -> np.ndarray:
        """RGB frame (H, W, 3). Thread-safe: executed on the dedicated GL thread."""
        return _GL_THREAD.submit(self._render, camera, width, height).result()

    def _render(self, camera, width, height) -> np.ndarray:
        w, h = width or self.cfg.render_width, height or self.cfg.render_height
        if self._renderer is None or self._render_size != (w, h):
            self._close()
            self._renderer = mujoco.Renderer(self.model, h, w)
            _WITH_RENDERER.add(self)
            self._render_size = (w, h)
        self.apply_colors()
        self._renderer.update_scene(self.data, camera=camera or self.cfg.camera)
        return self._renderer.render()

    def close(self):
        _GL_THREAD.submit(self._close).result()

    def __del__(self):
        # a GaitSim may be garbage-collected on any thread: hand its renderer to the GL thread
        r, self._renderer = getattr(self, "_renderer", None), None
        if r is not None:
            try:
                _GL_THREAD.submit(_free_renderer, r)
            except RuntimeError:          # interpreter shutting down
                pass

    def _close(self):
        if self._renderer is not None:
            _free_renderer(self._renderer)
            self._renderer = None

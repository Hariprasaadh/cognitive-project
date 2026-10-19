"""Closed-loop therapy session: world + virtual patient + exoskeleton controller."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import GaitConfig, PatientConfig, SensorConfig, SimConfig
from .controllers import Controller
from .patient import VirtualPatient
from .perception import Observation, Percept, Perception, SensorModel
from .sim import GaitSim
from . import metrics as M

LOG_COLUMNS = ["t", "phase", "stride", "knee", "knee_ref", "knee_vel", "thigh", "heel", "toe",
               "toe_height", "scuff", "tau_patient", "tau_vol", "tau_spastic", "tau_ext", "tau_exo",
               "fatigue", "effort", "capacity", "pelvis_z",
               # perception (estimates from noisy wearable sensors)
               "phase_hat", "phase_src", "heel_hat", "toe_hat", "T_hat", "subphase_hat",
               "tau_human_hat", "tau_patient_hat",
               # controller decisions
               "mode", "assist_gain", "tau_ff", "scuff_hat"]


@dataclass
class SessionResult:
    log: pd.DataFrame
    strides: pd.DataFrame
    metrics: dict
    perception: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    frames: list = field(default_factory=list)


class Session:
    """Run with `run()`, or step it incrementally with `advance()` for live dashboards."""

    def __init__(self, patient_cfg: PatientConfig, controller: Controller,
                 sim_cfg: SimConfig | None = None, gait_cfg: GaitConfig | None = None,
                 sensor_cfg: SensorConfig | None = None, log_hz: int = 100, skeleton_hz: float = 0.0):
        self.sim_cfg = sim_cfg or SimConfig()
        sensor_cfg = sensor_cfg or SensorConfig()      # one object shared by sensors and perception
        self.sim = GaitSim(self.sim_cfg, gait_cfg)
        self.patient = VirtualPatient(patient_cfg, self.sim, seed=self.sim_cfg.seed)
        self.sensors = SensorModel(sensor_cfg, self.sim_cfg.dt * self.sim_cfg.control_decimation,
                                   seed=self.sim_cfg.seed)
        self.perception = Perception(self.sim, sensor_cfg)
        self.controller = controller
        self.controller.reset(self.sim)
        self.T = self.sim.ref.stride_period     # nominal stride period
        # true gait phase: integrated with stride-to-stride variability
        self._rng = np.random.default_rng(self.sim_cfg.seed + 7)
        self.phase, self.stride, self.T_stride = 0.0, 0, self.T
        self.true_heel_strikes: list[float] = [0.0]
        self.percept: Percept | None = None
        self.measurement: Observation | None = None
        self.dec = self.sim_cfg.control_decimation
        self.ctrl_dt = self.sim_cfg.dt * self.dec
        self.log_every = max(1, int(round(1.0 / (log_hz * self.ctrl_dt))))
        self.frame_every = max(1, int(round(1.0 / (self.sim_cfg.render_fps * self.ctrl_dt))))
        self.rows: list[tuple] = []
        self.frames: list[np.ndarray] = []
        self.skeleton_frames: list[dict] = []
        self.skel_every = max(1, int(round(1.0 / (skeleton_hz * self.ctrl_dt)))) if skeleton_hz > 0 else 0
        self.tau_exo = 0.0
        self.last_info: dict = {}
        self._k = 0

    @property
    def time(self) -> float:
        return self.sim.time

    @property
    def done(self) -> bool:
        return self.sim.time >= self.sim_cfg.duration - 1e-9

    def observe(self) -> Observation:
        """Ground-truth sensor values; the exo only ever sees them through SensorModel."""
        s = self.sim
        heel, toe = s.insole("r")
        return Observation(t=s.time, knee=s.knee_angle, knee_vel=s.knee_vel, thigh=s.thigh_angle,
                           thigh_vel=s.thigh_vel, heel=heel, toe=toe, tau_exo=self.tau_exo)

    def _control_step(self, render_camera: str | None):
        s, p = self.sim, self.patient
        phase = self.phase                       # true gait phase (drives the world, not the exo)
        s.phase = phase
        self.measurement = self.sensors.measure(self.observe())   # noisy wearable sensors
        pc = self.percept = self.perception.update(self.measurement)
        cmd = self.controller.compute(pc)
        self.tau_exo = cmd.tau
        knee_ref = s.ref.fast("knee", phase)[0]
        pat = p.compute(phase, s.knee_angle, s.knee_vel)
        tau_ext = p.stance_load(phase, s.knee_angle, knee_ref)
        s.step(phase, pat["tau"], self.tau_exo, tau_ext, n=self.dec)
        p.update(pat["tau_vol"], self.tau_exo)
        self.last_info = cmd.info
        self._advance_phase()

        cap = max(p.capacity, 1.0)
        s.set_visual_drive(max(0.0, -pat["tau_vol"]) / cap, max(0.0, pat["tau_vol"]) / cap,
                           abs(self.tau_exo) / max(self.controller.cfg.tau_max, 1.0))
        if self._k % self.log_every == 0:
            # log state, phase and reference at the same instant (after the step)
            ph_now = self.phase
            ref_now = s.ref.fast("knee", ph_now)[0]
            s.phase = ph_now
            heel, toe = s.insole("r")
            self.rows.append((s.time, ph_now, self.stride, s.knee_angle, ref_now, s.knee_vel,
                              s.thigh_angle, heel, toe, s.toe_height("r"), s.toe_scuff("r"),
                              pat["tau"], pat["tau_vol"], pat["tau_spastic"], tau_ext, self.tau_exo,
                              p.fatigue, p.effort, p.capacity, float(s.data.mocap_pos[s.pelvis_mocap][2]),
                              pc.phase, pc.phase_source, pc.heel, pc.toe, pc.stride_period, pc.subphase,
                              pc.tau_human, pc.tau_patient,
                              cmd.info.get("mode", ""), cmd.info.get("gain", 0.0), cmd.info.get("tau_ff", 0.0), pc.toe_scuff))
        if self.skel_every and self._k % self.skel_every == 0:
            self.skeleton_frames.append({"t": round(s.time, 3), "tau_exo": self.tau_exo,
                                         "mode": cmd.info.get("mode", ""), "knee_deg": float(np.rad2deg(s.knee_angle)),
                                         "pts": {k: v.tolist() for k, v in s.skeleton().items()}})
        if render_camera and self._k % self.frame_every == 0:
            self.frames.append(s.render(render_camera))
        self._k += 1

    # ------------------------------------------------------------ live interventions (dashboard / viewer)
    def set_controller(self, controller: Controller):
        controller.reset(self.sim)
        self.controller = controller
        self.tau_exo = 0.0

    def inject_fatigue(self, amount: float = 0.3):
        self.patient.fatigue = float(min(0.95, self.patient.fatigue + amount))

    def fail_insole(self):
        self.sensors.cfg.insole_fault_at = self.sim.time

    def _advance_phase(self):
        """Integrate the true phase; each new stride gets a slightly different period."""
        self.phase += self.ctrl_dt / self.T_stride
        if self.phase >= 1.0:
            self.phase -= 1.0
            self.stride += 1
            self.true_heel_strikes.append(self.sim.time - self.phase * self.T_stride)
            cv = self.sim.gait_cfg.stride_cv
            self.T_stride = self.T * float(np.clip(1.0 + cv * self._rng.normal(), 0.85, 1.15))

    def advance(self, seconds: float, render_camera: str | None = None):
        end = min(self.sim.time + seconds, self.sim_cfg.duration)
        while self.sim.time < end - 1e-9:
            self._control_step(render_camera)

    def run(self, render_camera: str | None = None) -> SessionResult:
        self.advance(self.sim_cfg.duration, render_camera)
        return self.result()

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=LOG_COLUMNS)

    def result(self) -> SessionResult:
        log = self.dataframe()
        strides = M.stride_table(log)
        perc = M.perception_summary(log, self.true_heel_strikes, self.perception.heel_strikes)
        return SessionResult(log=log, strides=strides, metrics=M.summary(log, strides), perception=perc,
                             events=list(self.controller.events), frames=self.frames)

"""Perception: what the exoskeleton can infer from its own wearable sensors.

  SensorModel  - turns the true state into realistic measurements: quantised + noisy knee
                 encoder, biased + noisy thigh IMU, thigh + shank gyros (knee velocity),
                 delayed / bouncing / spuriously
                 triggering heel and toe insole switches, optional insole failure.
  Perception   - runs on those measurements only (no ground truth):
     * state filtering   - knee/thigh velocity and acceleration (Butterworth low-pass)
     * gait events       - debounced heel strike and toe off with refractory windows
     * gait phase        - event-based (time since heel strike / adaptive stride period),
                           with a self-calibrating thigh phase-portrait estimator
                           (Quintero et al. 2018) that takes over if the insoles fail
     * gait sub-phase    - Perry's 7 sub-phases + stance/swing (heel strike -> toe off)
     * toe scuff         - toe switch closing again during swing (foot drag, stiff-knee gait)
     * torque estimation - inverse dynamics of the leg model (mj_inverse) minus the known
                           exo torque minus the modelled body-weight load = the patient's
                           own muscle torque at the knee
"""
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.signal import butter

from .config import SensorConfig, clamp
from .gait import TOE_OFF, right_weight_share, subphase_index
from .patient import GRAVITY, HARNESS_CATCH, LOAD_LEVER

TWO_PI = 2.0 * np.pi
PHASE_SOURCES = ("insole events", "thigh phase portrait")


def wrap_phase(x):
    """Signed circular phase difference in [-0.5, 0.5)."""
    return (np.asarray(x) + 0.5) % 1.0 - 0.5


@dataclass
class Observation:
    """Sensor readings at one control step (true values before SensorModel, measured after)."""
    t: float
    knee: float           # knee encoder (rad, + = flexion)
    knee_vel: float       # rad/s (measured: shank gyro - thigh gyro)
    thigh: float          # thigh IMU angle from vertical (rad, + = flexion)
    thigh_vel: float      # gyro (rad/s)
    heel: bool            # insole heel switch
    toe: bool             # insole toe switch
    tau_exo: float        # exo motor torque (N·m)


@dataclass
class Percept:
    """The exoskeleton's belief about the user and the gait at one control step."""
    t: float
    knee: float
    knee_vel: float
    knee_acc: float
    thigh: float
    thigh_vel: float
    heel: bool                 # debounced insole switches
    toe: bool
    tau_exo: float
    phase: float               # estimated gait phase [0, 1), heel strike = 0
    phase_source: int          # index into PHASE_SOURCES
    stride_period: float       # estimated (s)
    stride: int                # detected strides
    subphase: int              # Perry sub-phase index (gait.SUBPHASES)
    stance: bool
    toe_scuff: bool            # toe contact during swing
    heel_strike: bool          # event detected at this step
    toe_off: bool
    insole_ok: bool
    tau_human: float           # net knee torque from the human side (muscles + body-weight load)
    tau_load: float            # modelled body-weight flexing moment
    tau_patient: float         # estimated patient muscle torque


class Biquad:
    """Causal 2nd-order Butterworth low-pass (direct form II transposed)."""

    def __init__(self, cutoff: float, fs: float):
        self.b, self.a = butter(2, cutoff, fs=fs)
        self.z1 = self.z2 = 0.0
        self.primed = False

    def __call__(self, x: float) -> float:
        b, a = self.b, self.a
        if not self.primed:      # start in steady state (unity DC gain) - no start-up transient
            self.primed = True
            self.z2 = x * (b[2] - a[2])
            self.z1 = x * (b[1] - a[1]) + self.z2
        y = b[0] * x + self.z1
        self.z1 = b[1] * x - a[1] * y + self.z2
        self.z2 = b[2] * x - a[2] * y
        return float(y)


# ====================================================================== sensors
class SensorModel:
    def __init__(self, cfg: SensorConfig | None, dt: float, seed: int = 0):
        self.cfg = cfg or SensorConfig()
        self.dt = dt
        self.rng = np.random.default_rng(seed + 1000)
        self.q_step = TWO_PI / 2 ** self.cfg.encoder_bits
        self._delay = deque(maxlen=max(1, int(round(self.cfg.insole_latency / dt))) + 1)
        self._last_ideal = None
        self._bounce_until = [-1.0, -1.0]
        self._pulse_until = -1.0

    def measure(self, obs: Observation) -> Observation:
        c, rng = self.cfg, self.rng
        knee = obs.knee + rng.normal(0, np.deg2rad(c.encoder_noise_deg))
        knee = round(knee / self.q_step) * self.q_step
        knee_vel = obs.knee_vel + rng.normal(0, c.gyro_noise * np.sqrt(2))   # two gyros
        thigh = obs.thigh + np.deg2rad(c.imu_bias_deg) + rng.normal(0, np.deg2rad(c.imu_noise_deg))
        thigh_vel = obs.thigh_vel + rng.normal(0, c.gyro_noise)
        heel, toe = self._insole(obs.t, obs.heel, obs.toe)
        tau = obs.tau_exo + rng.normal(0, c.torque_noise)
        return Observation(obs.t, knee, knee_vel, thigh, thigh_vel, heel, toe, tau)

    def _insole(self, t: float, heel: bool, toe: bool) -> tuple[bool, bool]:
        c = self.cfg
        if c.insole_fault_at is not None and t >= c.insole_fault_at:
            return False, False                      # connector unplugged: reads "no contact"
        self._delay.append((heel, toe))
        sig = list(self._delay[0])                   # delayed ideal switch states
        if self._last_ideal is not None:
            for i in range(2):
                if sig[i] != self._last_ideal[i]:
                    self._bounce_until[i] = t + c.insole_bounce
        self._last_ideal = tuple(sig)
        for i in range(2):                           # contact chatter after each transition
            if t < self._bounce_until[i] and self.rng.random() < 0.5:
                sig[i] = not sig[i]
        if self.rng.random() < c.insole_false_rate * self.dt:   # vibration-induced pulse
            self._pulse_until = t + self.rng.uniform(0.002, 0.006)
        if t < self._pulse_until:
            sig[0] = True
        return bool(sig[0]), bool(sig[1])


# ====================================================================== perception
class Debouncer:
    """A switch state is accepted only after it has been stable for n samples."""

    def __init__(self, n: int):
        self.n = n
        self.state = False
        self._count = 0

    def __call__(self, x: bool) -> bool:
        if x == self.state:
            self._count = 0
        else:
            self._count += 1
            if self._count >= self.n:
                self.state, self._count = x, 0
        return self.state


class PhasePortrait:
    """Thigh phase-portrait gait phase (Quintero et al. 2018), self-calibrating.

    phi = atan2(-thigh_vel_n, thigh_n) / 2pi, where thigh angle and velocity are centred
    and scaled by their range over the last stride. The map from portrait angle to gait
    phase is not exactly linear, so a 50-bin correction table is initialised from the
    normative gait model and refined online whenever the insole phase is trusted.
    """
    N_BINS = 50

    def __init__(self, ref, T: float, adapt_rate: float = 0.02):
        th = ref.curve("thigh")
        p = np.arange(400) / 400
        ang, vel = th(p), th(p, 1) / T
        self._norm = self._range(ang, vel)
        self.correction = np.zeros(self.N_BINS)
        counts = np.zeros(self.N_BINS)
        for pi, a, v in zip(p, ang, vel):      # prior: correction from the normative curve
            raw = self._raw(a, v)
            b = int(raw * self.N_BINS) % self.N_BINS
            self.correction[b] += wrap_phase(pi - raw)
            counts[b] += 1
        self.correction /= np.maximum(counts, 1)
        self.adapt_rate = adapt_rate
        self._buf_a, self._buf_v = [], []
        self._bin = 0
        self.calibrated_strides = 0

    @staticmethod
    def _range(ang, vel):
        a0, a1, v0, v1 = np.min(ang), np.max(ang), np.min(vel), np.max(vel)
        return ((a0 + a1) / 2, max((a1 - a0) / 2, 1e-3), (v0 + v1) / 2, max((v1 - v0) / 2, 1e-3))

    def _raw(self, ang: float, vel: float) -> float:
        ac, aa, vc, va = self._norm
        return float(np.arctan2(-(vel - vc) / va, (ang - ac) / aa) / TWO_PI) % 1.0

    def estimate(self, ang: float, vel: float) -> float:
        self._buf_a.append(ang)
        self._buf_v.append(vel)
        raw = self._raw(ang, vel)
        self._bin = int(raw * self.N_BINS) % self.N_BINS
        return float((raw + self.correction[self._bin]) % 1.0)

    def learn(self, estimate: float, trusted_phase: float):
        """Pull the current bin's correction toward the trusted (insole) phase."""
        self.correction[self._bin] += self.adapt_rate * float(wrap_phase(trusted_phase - estimate))

    def new_stride(self):
        """Re-normalise with the thigh range observed over the last stride."""
        if len(self._buf_a) > 50:
            self._norm = self._range(self._buf_a, self._buf_v)
            self.calibrated_strides += 1
        self._buf_a, self._buf_v = [], []


class Perception:
    DEBOUNCE_SAMPLES = 4             # 8 ms at 500 Hz
    REFRACTORY = 0.6                 # min time between heel strikes (fraction of stride)
    OVERDUE = 1.08                   # heel strike late by this much -> trust the portrait
    PLL_GAIN = 0.03                  # portrait phase tracker gain per sample (~70 ms)
    DRAG_MARGIN = 0.02               # toe contact this long after nominal toe off = scuff
    HARD_STOP = np.deg2rad(-4.5)     # near the hyperextension stop: impact forces, hold estimate
    FAULT_TIMEOUT = 1.5              # no heel strike for this many strides -> insole fault

    def __init__(self, sim, sensor_cfg: SensorConfig | None = None):
        self.sim = sim
        self.cfg = sensor_cfg or SensorConfig()
        self.dt = sim.cfg.dt * sim.cfg.control_decimation
        self.fs = 1.0 / self.dt
        self.T_nominal = sim.ref.stride_period
        self.ref = sim.ref
        self.weight = sim.gait_cfg.body_mass * GRAVITY * (1.0 - sim.gait_cfg.bws)
        self._d = mujoco.MjData(sim.model)     # private copy for inverse dynamics
        self._qa, self._da = sim.qadr, sim.dadr
        # compensate the known detection delay (debounce + datasheet latency)
        self.event_delay = self.DEBOUNCE_SAMPLES * self.dt + self.cfg.insole_latency
        self.reset()

    def reset(self):
        fs = self.fs
        self._f_knee_vel, self._f_knee_acc = Biquad(40, fs), Biquad(30, fs)
        self._f_thigh_vel, self._f_thigh_acc = Biquad(40, fs), Biquad(25, fs)
        self._f_tau = Biquad(25, fs)
        self._f_thigh = Biquad(12, fs)
        self._heel = Debouncer(self.DEBOUNCE_SAMPLES)
        self._toe = Debouncer(self.DEBOUNCE_SAMPLES)
        self.portrait = PhasePortrait(self.ref, self.T_nominal)
        self.T_hat = self.T_nominal
        self._durations: deque = deque(maxlen=3)
        self.t_hs = None
        self.stride = 0
        self.insole_ok = False
        self.heel_strikes: list[float] = []
        self.toe_offs: list[float] = []
        self._prev = None
        self._prev_heel = self._prev_toe = False
        self._toe_off_done = True
        self._phi_prev = None
        self._tau_patient = 0.0
        self.torque_valid = True

    # ------------------------------------------------------------ main update
    def update(self, m: Observation) -> Percept:
        t = m.t
        knee_vel = self._f_knee_vel(m.knee_vel)
        thigh_vel = self._f_thigh_vel(m.thigh_vel)
        if self._prev is None:
            knee_acc, thigh_acc = self._f_knee_acc(0.0), self._f_thigh_acc(0.0)
        else:
            knee_acc = self._f_knee_acc((knee_vel - self._prev[0]) / self.dt)
            thigh_acc = self._f_thigh_acc((thigh_vel - self._prev[1]) / self.dt)
        self._prev = (knee_vel, thigh_vel)

        heel, toe = self._heel(m.heel), self._toe(m.toe)
        hs, to = self._events(t, heel, toe)

        # ---- gait phase: insole events when trusted, else the thigh phase portrait
        phi_pp = self.portrait.estimate(self._f_thigh(m.thigh), thigh_vel)
        since_hs = t - self.t_hs if self.t_hs is not None else np.inf
        if self.insole_ok and since_hs > self.FAULT_TIMEOUT * self.T_hat:
            self.insole_ok = False                       # heel strikes stopped arriving
        use_events = self.insole_ok and since_hs < self.OVERDUE * self.T_hat
        if use_events:
            phase = min(since_hs / self.T_hat, 0.999)
            self.portrait.learn(phi_pp, phase)
        elif self._phi_prev is None:                     # start-up: first sample
            phase = phi_pp
        else:                                            # overdue / failed insole: phase tracker
            step = self.dt / self.T_hat                  # cadence prior
            pred = self._phi_prev + step
            # correct toward the portrait; phase only moves forward (0.3x - 2x nominal rate)
            dphi = step + self.PLL_GAIN * float(wrap_phase(phi_pp - pred))
            phase = (self._phi_prev + clamp(dphi, 0.3 * step, 2.0 * step)) % 1.0
            if self._phi_prev > 0.9 and phase < 0.1:
                self.portrait.new_stride()               # portrait-detected stride
        self._phi_prev = phase
        if self.insole_ok:   # stance from heel strike until the detected toe off
            stance = not self._toe_off_done and phase < TOE_OFF + self.DRAG_MARGIN
        else:
            stance = phase < TOE_OFF
        # toe still (or again) on the belt well after toe off is due: foot drag / scuff
        # (the switch reports event_delay late, so allow for it before calling it a drag)
        scuff = (self.insole_ok and toe and not heel
                 and phase > TOE_OFF + self.DRAG_MARGIN + self.event_delay / self.T_hat)

        # ---- torque estimation
        tau_human = self._inverse_dynamics(m, knee_vel, knee_acc, thigh_vel, thigh_acc, phase) - m.tau_exo
        tau_load = self._load(phase, m.knee) if stance else 0.0
        tau_f = self._f_tau(tau_human - tau_load)
        self.torque_valid = m.knee > self.HARD_STOP      # joint-stop impacts are not muscle torque
        if self.torque_valid:
            self._tau_patient = tau_f
        tau_patient = self._tau_patient

        return Percept(t=t, knee=m.knee, knee_vel=knee_vel, knee_acc=knee_acc, thigh=m.thigh,
                       thigh_vel=thigh_vel, heel=heel, toe=toe, tau_exo=m.tau_exo,
                       phase=float(phase), phase_source=0 if use_events else 1,
                       stride_period=self.T_hat, stride=self.stride, subphase=subphase_index(phase),
                       stance=bool(stance), toe_scuff=bool(scuff), heel_strike=hs, toe_off=to, insole_ok=self.insole_ok,
                       tau_human=float(tau_human), tau_load=float(tau_load), tau_patient=float(tau_patient))

    # ------------------------------------------------------------ gait events
    def _events(self, t: float, heel: bool, toe: bool) -> tuple[bool, bool]:
        hs = to = False
        if heel and not self._prev_heel:
            t_event = t - self.event_delay
            if self.t_hs is None or t_event - self.t_hs > self.REFRACTORY * self.T_hat:
                if self.t_hs is not None:
                    d = t_event - self.t_hs
                    if 0.6 * self.T_hat < d < 1.6 * self.T_hat:   # plausible stride
                        self._durations.append(d)
                        self.T_hat = float(np.median(self._durations))
                self.t_hs = t_event
                self.stride += 1
                self.heel_strikes.append(t_event)
                self.portrait.new_stride()
                self.insole_ok = True
                self._toe_off_done = False
                hs = True
        if (not toe and self._prev_toe and not heel and not self._toe_off_done
                and self.t_hs is not None and t - self.t_hs > 0.3 * self.T_hat):
            self.toe_offs.append(t - self.event_delay)
            self._toe_off_done = True                   # later toe drags are not toe-offs
            to = True
        self._prev_heel, self._prev_toe = heel, toe
        return hs, to

    # ------------------------------------------------------------ torque estimation
    def _inverse_dynamics(self, m, knee_vel, knee_acc, thigh_vel, thigh_acc, phase) -> float:
        """Generalised knee force required by the measured motion (rigid-body model)."""
        d, qa, da = self._d, self._qa, self._da
        bias = np.deg2rad(self.cfg.imu_bias_deg)        # calibrated mounting offset
        d.qpos[qa["r_hip"]], d.qvel[da["r_hip"]], d.qacc[da["r_hip"]] = m.thigh - bias, thigh_vel, thigh_acc
        d.qpos[qa["r_knee"]], d.qvel[da["r_knee"]], d.qacc[da["r_knee"]] = m.knee, knee_vel, knee_acc
        # ankle: not instrumented; it follows the normative pattern at the estimated phase
        q, qd = self.ref.fast("ankle", phase)
        h = 0.004
        qdd = (self.ref.fast("ankle", phase + h)[1] - self.ref.fast("ankle", phase - h)[1]) / (2 * h * self.T_hat)
        d.qpos[qa["r_ankle"]], d.qvel[da["r_ankle"]], d.qacc[da["r_ankle"]] = q, qd, qdd
        mujoco.mj_inverse(self.sim.model, d)
        return float(d.qfrc_inverse[da["r_knee"]])

    def _load(self, phase: float, knee: float) -> float:
        """Body-weight flexing moment at the knee from the known BWS setting: weight share
        x lever x sin(knee), released when the harness catches a collapse."""
        knee_ref = self.ref.fast("knee", phase)[0]
        lo, hi = HARNESS_CATCH
        catch = 1.0 - clamp((knee - knee_ref - lo) / (hi - lo), 0.0, 1.0)
        return LOAD_LEVER * self.weight * right_weight_share(phase) * catch * np.sin(max(knee, 0.0))

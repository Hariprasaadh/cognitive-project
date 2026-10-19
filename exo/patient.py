"""Virtual patient: the impaired right knee's neuromuscular torque and its stance load.

Healthy knee control is modelled as feedforward (inverse dynamics of the normative gait
plus the expected body-weight load) and feedback (intrinsic muscle impedance plus a
delayed stretch reflex). Impairments:
  * weakness   - torque capacity  s * (1 - fatigue) * tau_max and reduced neural gain
  * spasticity - rectus-femoris hyperreflexia: velocity-dependent resistance to knee
                 flexion in pre-/initial swing (stiff-knee gait); weak flexors
  * fatigue    - accumulates with relative effort, recovers slowly (simplified
                 3-compartment model, Xia & Frey Law 2008)
  * slacking   - effort drops when the exoskeleton carries the load (Reinkensmeyer)
  * motor noise - signal-dependent, low-pass filtered
"""
from collections import deque

import numpy as np

from .config import GaitConfig, PatientConfig, clamp
from .gait import right_weight_share

GRAVITY = 9.81
LOAD_LEVER = 0.30            # effective GRF moment arm per unit sin(knee angle) (m)
HARNESS_CATCH = (0.26, 0.70)  # excess flexion (rad) where the harness starts / fully unloads


class VirtualPatient:
    # healthy neuromuscular impedance (N·m/rad, N·m·s/rad); reflex acts with a delay
    K_INTRINSIC, D_INTRINSIC = 60.0, 4.0
    K_REFLEX, D_REFLEX = 190.0, 4.0
    REFLEX_DELAY = 0.025
    SPASTIC_THRESHOLD = 0.5  # rad/s
    SPASTIC_WINDOW = (0.50, 0.85)  # gait phases where the stretch reflex is hyperactive
    FLEXOR_RATIO = 0.6       # knee flexor / extensor strength ratio
    FATIGUE_RATE, RECOVERY_RATE = 0.012, 0.004  # 1/s
    SLACK_TAU = 3.0          # s, time constant of effort adaptation

    def __init__(self, cfg: PatientConfig, sim, gait_cfg: GaitConfig | None = None, seed: int = 0):
        self.cfg = cfg
        self.gait = gait_cfg or sim.gait_cfg
        self.ref = sim.ref
        self.dt = sim.cfg.dt * sim.cfg.control_decimation
        self.rng = np.random.default_rng(seed)
        self.weight = self.gait.body_mass * GRAVITY * (1.0 - self.gait.bws)
        self._tau_id = np.array(sim.id_table["r_knee"])   # internal model of the knee dynamics
        n_delay = max(1, int(round(self.REFLEX_DELAY / self.dt)))
        self._history = deque(maxlen=n_delay)
        self.reset()

    def reset(self):
        self.fatigue = 0.0
        self.effort = 1.0
        self._noise = 0.0
        self._assist_share = 0.0
        self._history.clear()

    # ------------------------------------------------------------ feedforward
    def tau_id(self, phase: float) -> float:
        x = (phase % 1.0) * (len(self._tau_id) - 1)
        i = int(x)
        return float(self._tau_id[i] + (x - i) * (self._tau_id[min(i + 1, len(self._tau_id) - 1)] - self._tau_id[i]))

    # ------------------------------------------------------------ environment
    def stance_load(self, phase: float, knee: float, knee_ref: float) -> float:
        """External flexing moment on the knee from the (partially supported) body weight.
        Grows with knee flexion (buckling feedback); the BWS harness catches a collapse."""
        w = right_weight_share(phase)
        if w <= 0.0:
            return 0.0
        lo, hi = HARNESS_CATCH
        catch = 1.0 - clamp((knee - knee_ref - lo) / (hi - lo), 0.0, 1.0)
        return float(LOAD_LEVER * self.weight * w * catch * np.sin(max(knee, 0.0)))

    # ------------------------------------------------------------ neuromuscular output
    @property
    def capacity(self) -> float:
        return self.cfg.strength * (1.0 - self.fatigue) * self.cfg.tau_max_healthy

    def compute(self, phase: float, knee: float, knee_vel: float) -> dict:
        """Patient's knee torque for this control step (+ = flexion)."""
        c = self.cfg
        ref, dref = self.ref.fast("knee", phase)
        self._history.append((knee, knee_vel))
        k_del, v_del = self._history[0]

        expected_load = LOAD_LEVER * self.weight * right_weight_share(phase) * np.sin(max(ref, 0.0))
        feedforward = self.tau_id(phase) - expected_load
        feedback = (self.K_INTRINSIC * (ref - knee) + self.D_INTRINSIC * (dref - knee_vel)
                    + self.K_REFLEX * (ref - k_del) + self.D_REFLEX * (dref - v_del))
        neural_gain = (0.5 + 0.5 * c.strength) * self.effort
        desired = neural_gain * (feedforward + feedback)

        cap = self.capacity
        voluntary = clamp(float(desired), -cap, self.FLEXOR_RATIO * cap)
        # signal-dependent motor noise, low-pass filtered (~8 Hz)
        a = self.dt / (self.dt + 0.02)
        self._noise += a * (self.rng.normal() * c.motor_noise * abs(voluntary) * 3.0 - self._noise)
        voluntary += self._noise
        # rectus-femoris hyperreflexia: phase-modulated, active from pre-swing to mid-swing
        gate = 1.0 if self.SPASTIC_WINDOW[0] <= phase % 1.0 < self.SPASTIC_WINDOW[1] else 0.0
        spastic = -gate * c.spasticity * max(0.0, knee_vel - self.SPASTIC_THRESHOLD)
        return {"tau": voluntary + spastic, "tau_vol": voluntary, "tau_spastic": spastic,
                "tau_desired": desired, "saturated": abs(desired) >= cap}

    def update(self, tau_vol: float, tau_exo: float):
        """Fatigue and slacking dynamics, once per control step."""
        c = self.cfg
        u = min(1.0, abs(tau_vol) / (c.strength * c.tau_max_healthy + 1e-6))
        F = self.FATIGUE_RATE * c.fatigue_rate
        R = self.RECOVERY_RATE * c.recovery_rate
        self.fatigue = clamp(self.fatigue + self.dt * (F * u * (1 - self.fatigue) - R * self.fatigue), 0.0, 0.95)
        share = abs(tau_exo) / (abs(tau_exo) + abs(tau_vol) + 1.0)
        a = self.dt / self.SLACK_TAU
        self._assist_share += a * (share - self._assist_share)
        target_effort = 1.0 - c.slacking * self._assist_share
        self.effort += a * (target_effort - self.effort)

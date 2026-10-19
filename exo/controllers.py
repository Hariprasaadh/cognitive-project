"""Exoskeleton controllers. They act only on the Percept (estimated state), never on ground truth.

  NoExo             - unpowered / transparent baseline
  FixedImpedance    - conventional phase-referenced impedance control (constant gains)
  CognitiveAdaptive - perceive -> reason -> act -> learn:
                        * assist-as-needed gains per gait-phase bin (Emken et al. 2007)
                        * iterative learning feedforward across strides (Arimoto ILC)
                        * explainable supervisor: ASSIST_AS_NEEDED / CHALLENGE /
                          FATIGUE_SUPPORT / SAFETY_HOLD, every decision logged with a reason
                        * within-stride anti-buckling reflex
                        * goal adaptation: toe scuffs raise the swing knee-flexion target
All controllers share the same safety layer (torque + slew limits, hyperextension guard,
overspeed cut-off).
"""
from dataclasses import dataclass, field

import numpy as np

from .config import ControllerConfig, clamp
from .perception import PHASE_SOURCES, Percept

MODES = ("ASSIST_AS_NEEDED", "CHALLENGE", "FATIGUE_SUPPORT", "SAFETY_HOLD")
MODE_COLORS = {"ASSIST_AS_NEEDED": "#2a9d8f", "CHALLENGE": "#7b61ff",
               "FATIGUE_SUPPORT": "#e9c46a", "SAFETY_HOLD": "#e76f51"}
def deadband(e, tol):
    """Error outside the tolerance tunnel (continuous, zero inside)."""
    return np.sign(e) * np.maximum(np.abs(e) - tol, 0.0)


MODE_SCALE = {"ASSIST_AS_NEEDED": 1.0, "CHALLENGE": 0.6, "FATIGUE_SUPPORT": 1.4, "SAFETY_HOLD": 1.3}


@dataclass
class Command:
    tau: float
    info: dict = field(default_factory=dict)


class Controller:
    name = "base"
    OVERSPEED = 12.0          # rad/s: knee velocity beyond this -> torque cut

    def __init__(self, cfg: ControllerConfig | None = None):
        self.cfg = cfg or ControllerConfig()
        self.events: list[tuple[float, str]] = []   # explainable decision log (t, message)

    def reset(self, sim):
        self.events.clear()
        self.ref = sim.ref
        self.dt = sim.cfg.dt * sim.cfg.control_decimation
        self._tau_prev = 0.0
        self._overspeed_logged = False

    def log(self, t: float, msg: str):
        self.events.append((round(float(t), 2), msg))

    def reference(self, p: Percept) -> tuple[float, float]:
        """Normative knee angle and velocity at the ESTIMATED phase and cadence."""
        q, qd = self.ref.fast("knee", p.phase)
        return q, qd * self.ref.stride_period / p.stride_period

    def safety(self, p: Percept, tau: float) -> tuple[float, str]:
        """Last line of defence, identical for every controller."""
        c, flag = self.cfg, ""
        if abs(p.knee_vel) > self.OVERSPEED:
            if not self._overspeed_logged:
                self.log(p.t, f"SAFETY: knee speed {p.knee_vel:.1f} rad/s > {self.OVERSPEED} -> torque cut")
                self._overspeed_logged = True
            tau, flag = 0.0, "overspeed"
        if p.knee < np.deg2rad(1.0) and tau < 0:           # never push into hyperextension
            tau *= max(0.0, (p.knee + np.deg2rad(2.0)) / np.deg2rad(3.0))
            flag = flag or "hyperextension guard"
        tau = clamp(float(tau), -c.tau_max, c.tau_max)
        step = c.rate_max * self.dt
        tau = clamp(tau, self._tau_prev - step, self._tau_prev + step)
        self._tau_prev = tau
        return tau, flag

    def compute(self, p: Percept) -> Command:
        raise NotImplementedError


class NoExo(Controller):
    """Exoskeleton worn but unpowered / transparent (zero torque)."""
    name = "No assistance"

    def compute(self, p: Percept) -> Command:
        return Command(0.0, {"mode": "OFF"})


class FixedImpedance(Controller):
    """Phase-referenced impedance with constant gains (the conventional approach)."""
    name = "Fixed impedance"

    def compute(self, p: Percept) -> Command:
        c = self.cfg
        ref, dref = self.reference(p)
        k = c.kp * (c.stance_stiffness_boost if p.stance else 1.0)
        tau_fb = c.fixed_gain * (k * (ref - p.knee) + c.kd * (dref - p.knee_vel))
        tau, flag = self.safety(p, tau_fb)
        return Command(tau, {"mode": "FIXED", "gain": c.fixed_gain, "tau_fb": tau_fb, "tau_ff": 0.0,
                             "knee_ref": ref, "safety": flag})


class CognitiveAdaptive(Controller):
    """Assist-as-needed + iterative learning + explainable supervisor + safety."""
    name = "Cognitive adaptive"
    ILC_LEAD = 1                 # bins of phase lead in the learning update (~2 % cycle)
    HOLD_STRIDES = 2             # SAFETY_HOLD lasts this many strides after a buckle
    MIN_DWELL = 2                # strides before another mode change (hysteresis)

    def reset(self, sim):
        super().reset(sim)
        n = self.cfg.n_bins
        self.n = n
        self.tol = np.deg2rad(self.cfg.tolerance_deg)
        self.gain = np.full(n, self.cfg.g_init)   # AAN gain per phase bin
        self.tol_bins = np.full(n, self.tol)       # per-bin tunnel width (narrowed after toe scuffs)
        self.swing_bins = np.arange(int(0.55 * n), int(0.80 * n))
        # toe-clearance goal: a smooth bump on the early-swing flexion target
        ph = (np.arange(n) + 0.5) / n
        self._bump = np.where((ph > 0.50) & (ph < 0.80), np.sin(np.pi * (ph - 0.50) / 0.30) ** 2, 0.0)
        self.clearance_deg = 0.0
        self.ref_offset = np.zeros(n)
        self.u_ff = np.zeros(n)              # learned feedforward per phase bin (N·m)
        self._new_stride_buffers()
        self.mode = "ASSIST_AS_NEEDED"
        self._mode_since = 0
        self._hold_left = 0
        self.stride_count = 0
        self.history: list[dict] = []        # per-stride beliefs (for the dashboard)
        self._effort_baseline = None
        self._rmse_ema = None
        self._good_strides = 0
        self._scuff_count = 0
        self._patience = 3                   # good strides required before challenging (learned)
        self._challenge_cause = "performance"
        self._mode_strides = 0
        self._rest_advised = False
        self._reflex = False
        self._insole_ok_prev = True
        self._last_phase = 0.0

    def _new_stride_buffers(self):
        self._e_sum = np.zeros(self.n)
        self._cnt = np.zeros(self.n)
        self._sq, self._k = 0.0, 0
        self._max_excess = 0.0
        self._effort, self._effort_n = 0.0, 0
        self._assist, self._patient = 0.0, 0.0
        self._scuffed = False

    # ------------------------------------------------------------ act (every step)
    def compute(self, p: Percept) -> Command:
        c = self.cfg
        # a stride ends at a detected heel strike, or at a phase wrap when the insoles failed
        wrapped = (not p.insole_ok) and self._last_phase > 0.9 and p.phase < 0.1
        if p.heel_strike or wrapped:
            if self._k > 50:
                self._end_stride(p)
            self._new_stride_buffers()
        self._last_phase = p.phase
        if self._insole_ok_prev and not p.insole_ok and self.stride_count > 0:
            self.log(p.t, "PERCEPTION: no heel strike for 1.5 strides -> insole fault; "
                          f"gait phase now from the {PHASE_SOURCES[1]}")
        elif not self._insole_ok_prev and p.insole_ok and self.stride_count > 0:
            self.log(p.t, "PERCEPTION: heel strikes back -> insole phase restored")
        self._insole_ok_prev = p.insole_ok

        ref, dref = self.reference(p)
        b = int(p.phase * self.n) % self.n
        ref += self.ref_offset[b]                      # adapted (toe-clearance) target
        e, de = ref - p.knee, dref - p.knee_vel

        # learn: accumulate this stride's errors and contributions
        self._e_sum[b] += e
        self._cnt[b] += 1
        self._sq += e * e
        self._k += 1
        if p.toe_scuff:
            self._scuffed = True
        if p.stance:
            self._max_excess = max(self._max_excess, -e)
            self._effort += max(0.0, -p.tau_patient)      # patient's own extensor effort
            self._effort_n += 1

        # reflex: knee collapsing in stance -> stiffen immediately for the rest of stance
        if p.stance and -e > np.deg2rad(c.buckle_threshold_deg):
            self._reflex = True
        if not p.stance:
            self._reflex = False

        scale = MODE_SCALE[self.mode]
        k = c.kp * (c.stance_stiffness_boost if p.stance else 1.0)
        g = clamp(float(self.gain[b]) * scale, c.g_min, c.g_max * 1.5)
        if self._reflex:
            g = max(g, 2.0)
        tol = float(self.tol_bins[b])
        e_out = e - tol if e > tol else e + tol if e < -tol else 0.0   # tunnel: free inside +-tol
        tau_fb = g * (k * e_out + c.kd * de)   # tunnel: free inside +-tol
        tau_ff = float(self.u_ff[b]) * scale
        tau, flag = self.safety(p, tau_fb + tau_ff)
        self._assist += abs(tau)
        self._patient += abs(p.tau_patient)
        return Command(tau, {"mode": self.mode, "gain": g, "tau_fb": tau_fb, "tau_ff": tau_ff,
                             "knee_ref": ref, "safety": flag or ("anti-buckling reflex" if self._reflex else "")})

    # ------------------------------------------------------------ learn + reason (per stride)
    def _end_stride(self, p: Percept):
        c = self.cfg
        self.stride_count += 1
        seen = self._cnt > 0
        e_bar = np.zeros(self.n)
        e_bar[seen] = self._e_sum[seen] / self._cnt[seen]
        if self._scuffed:        # foot drag: raise the flexion goal, tighten + boost swing support
            self.clearance_deg = min(self.clearance_deg + 2.0, 10.0)
            self.tol_bins[self.swing_bins] = np.maximum(self.tol_bins[self.swing_bins] * 0.5, np.deg2rad(1.0))
            self.gain[self.swing_bins] = np.minimum(self.gain[self.swing_bins] + 0.3, c.g_max)
            self._scuff_count += 1
            if self._scuff_count <= 3 or self._scuff_count % 10 == 0:
                self.log(p.t, f"TOE SCUFF #{self._scuff_count}: toe still on the belt in swing -> swing flexion "
                              f"target +{self.clearance_deg:.0f} deg, tunnel +-"
                              f"{np.rad2deg(self.tol_bins[self.swing_bins].mean()):.1f} deg")
        else:                    # no drag: slowly relax the extra flexion (keeps re-testing)
            self.clearance_deg *= 0.95
        self.ref_offset = np.deg2rad(self.clearance_deg) * self._bump
        e_out = deadband(e_bar, self.tol_bins)          # only errors outside the tunnel count
        # assist-as-needed: gains grow with error, are forgotten where the patient copes
        self.gain[seen] = np.clip(c.aan_forgetting * self.gain[seen] + c.aan_error_gain * np.abs(e_out[seen]),
                                  c.g_min, c.g_max)
        # iterative learning feedforward with forgetting (keeps testing the patient)
        lead = np.roll(e_out, -self.ILC_LEAD)
        lr = 0.0 if self.mode == "CHALLENGE" else 1.0     # challenge: stop learning, let it decay
        self.u_ff = np.clip(c.ilc_forgetting * self.u_ff + lr * c.ilc_gain * lead, -c.tau_max, c.tau_max)

        rmse = float(np.rad2deg(np.sqrt(self._sq / max(self._k, 1))))
        excess = float(np.rad2deg(self._max_excess))
        effort = self._effort / max(self._effort_n, 1)
        share = self._assist / (self._assist + self._patient + 1e-9)
        self._rmse_ema = rmse if self._rmse_ema is None else 0.6 * self._rmse_ema + 0.4 * rmse
        if self._effort_baseline is None and self.stride_count >= 3:
            self._effort_baseline = max(effort, 1.0)
        engagement = effort / self._effort_baseline if self._effort_baseline else 1.0
        self._reason(p.t, rmse, excess, engagement, self._scuffed)
        self.history.append({"t": p.t, "stride": self.stride_count, "rmse_deg": rmse, "excess_deg": excess,
                             "engagement": engagement, "assist_share": share, "mode": self.mode,
                             "mean_gain": float(self.gain.mean()), "ff_rms": float(np.sqrt(np.mean(self.u_ff ** 2))),
                             "gain_profile": self.gain.copy(), "ff_profile": self.u_ff.copy()})

    def _set_mode(self, t: float, mode: str, why: str):
        if mode != self.mode:
            self.log(t, f"{self.mode} -> {mode}: {why}")
            self.mode = mode
            self._mode_since = self.stride_count
            self._mode_strides = 0

    def _challenge_failed(self, why: str) -> str:
        """Remember a failed challenge and be more patient before the next one."""
        self._patience = min(2 * self._patience, 24)
        return f"{why}; next challenge only after {self._patience} good strides"

    def _reason(self, t, rmse, excess, engagement, scuffed=False):
        """Explainable supervisor: belief about the patient -> assistance mode.

        Low patient effort with low error is ambiguous: slacking (the patient lets the
        exo work) or fatigue (the patient cannot do more). The supervisor tests the
        slacking hypothesis by reducing assistance (CHALLENGE); if the knee then fails,
        it concludes fatigue and switches to FATIGUE_SUPPORT.
        """
        c = self.cfg
        buckled = excess > c.buckle_threshold_deg
        self._mode_strides = getattr(self, "_mode_strides", 0) + 1
        if self.mode == "CHALLENGE" and (buckled or rmse > 7.0 or scuffed):
            what = (f"knee buckled {excess:.0f} deg" if buckled else "toe scuffed" if scuffed
                    else f"error rose to {rmse:.1f} deg")
            if self._challenge_cause == "slacking" and not scuffed:
                self._set_mode(t, "FATIGUE_SUPPORT",
                               f"slacking test failed: with less assistance the {what}. Low effort "
                               f"({100 * engagement:.0f} % of baseline) means FATIGUE, not slacking -> "
                               f"assistance x{MODE_SCALE['FATIGUE_SUPPORT']}, no more challenges")
            elif buckled:
                self._hold_left = self.HOLD_STRIDES
                self._set_mode(t, "SAFETY_HOLD", self._challenge_failed(
                    f"{what} under challenge -> stance support x{MODE_SCALE['SAFETY_HOLD']}"))
            else:
                self._set_mode(t, "ASSIST_AS_NEEDED", self._challenge_failed(f"{what} under challenge"))
            return
        if buckled:
            if self.mode == "FATIGUE_SUPPORT":
                return                                   # already at the strongest support level
            self._hold_left = self.HOLD_STRIDES
            if self.mode != "SAFETY_HOLD":
                self._set_mode(t, "SAFETY_HOLD", f"knee buckled {excess:.0f} deg beyond normative in stance "
                                                 f"(> {c.buckle_threshold_deg:.0f}) -> stance support x{MODE_SCALE['SAFETY_HOLD']}")
            return
        if self.mode == "SAFETY_HOLD":
            self._hold_left -= 1
            if self._hold_left <= 0:
                self._set_mode(t, "ASSIST_AS_NEEDED", f"{self.HOLD_STRIDES} strides without buckling -> release hold")
            return
        if self.stride_count - self._mode_since < self.MIN_DWELL or self._effort_baseline is None:
            return
        self._good_strides = self._good_strides + 1 if (rmse < 5.0 and not scuffed) else 0
        if self.mode == "FATIGUE_SUPPORT":
            if self._mode_strides == 5 and not self._rest_advised:
                self._rest_advised = True
                self.log(t, "RECOMMENDATION: fatigue persists after 5 supported strides -> "
                            "suggest a 2-minute seated rest")
            if engagement > 0.95:
                self._set_mode(t, "ASSIST_AS_NEEDED", f"effort recovered ({100 * engagement:.0f} %) -> normal assistance")
        elif engagement < 0.7 and rmse > 5.0:
            self._set_mode(t, "FATIGUE_SUPPORT", f"patient extensor effort down to {100 * engagement:.0f} % of "
                                                 f"baseline AND error {rmse:.1f} deg -> fatigue, "
                                                 f"assistance x{MODE_SCALE['FATIGUE_SUPPORT']}")
        elif (self.mode == "ASSIST_AS_NEEDED" and engagement < 0.85 and not scuffed
              and self._good_strides >= min(self._patience, 3)):
            self._challenge_cause = "slacking"
            self._set_mode(t, "CHALLENGE", f"patient extensor effort down to {100 * engagement:.0f} % of baseline "
                                           f"while error is only {rmse:.1f} deg -> slacking? testing it: "
                                           f"assistance x{MODE_SCALE['CHALLENGE']}")
        elif self.mode == "ASSIST_AS_NEEDED" and self._good_strides >= self._patience:
            self._challenge_cause = "performance"
            self._set_mode(t, "CHALLENGE", f"error < 5 deg for {self._good_strides} strides -> reduce assistance "
                                           f"x{MODE_SCALE['CHALLENGE']} to keep the patient working")


CONTROLLERS = {"none": NoExo, "fixed": FixedImpedance, "cognitive": CognitiveAdaptive}


def make_controller(cfg: ControllerConfig | str) -> Controller:
    if isinstance(cfg, str):
        cfg = ControllerConfig(kind=cfg)
    try:
        return CONTROLLERS[cfg.kind](cfg)
    except KeyError:
        raise ValueError(f"unknown controller kind {cfg.kind!r}") from None

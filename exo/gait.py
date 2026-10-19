"""Normative sagittal-plane gait reference and gait sub-phases.

Joint-angle keypoints (degrees vs % gait cycle, heel strike = 0 %) follow the normative
adult curves in Winter (2009) and Perry & Burnfield (2010). A periodic cubic spline turns
them into smooth, differentiable trajectories; lookup tables make per-step queries cheap.

The ankle's return to neutral in early swing is slightly faster than the population
mean so the toe clears the belt in this sagittal-plane model (no pelvic obliquity).
"""
import numpy as np
from scipy.interpolate import CubicSpline

# (% gait cycle, angle in deg); first and last values match (periodic)
KEYPOINTS = {
    "hip": [(0, 30), (10, 27), (20, 20), (30, 12), (40, 4), (50, -5), (55, -8), (60, -2), (65, 6),
            (70, 15), (75, 22), (80, 27), (85, 31), (90, 32), (95, 31), (100, 30)],
    "knee": [(0, 4), (5, 11), (10, 17), (15, 19), (20, 17), (25, 13), (30, 9), (35, 6), (40, 5),
             (45, 6), (50, 10), (55, 18), (60, 31), (65, 46), (70, 58), (73, 61), (76, 61),
             (80, 55), (85, 41), (90, 23), (95, 8), (100, 4)],
    "ankle": [(0, 0), (5, -4), (8, -6), (15, -2), (30, 4), (40, 8), (47, 10), (52, 7), (57, -3),
              (62, -15), (66, -8), (70, 0), (78, 4), (90, 3), (100, 0)],
}

# Hip angles above are pelvis-relative; the model's pelvis stays upright, so the thigh's
# absolute angle is hip flexion minus the mean anterior pelvic tilt.
PELVIC_TILT_DEG = 12.0

# Perry's gait sub-phases: (name, start, end) as gait-cycle fractions
SUBPHASES = [
    ("Loading Response", 0.00, 0.12),
    ("Mid Stance", 0.12, 0.31),
    ("Terminal Stance", 0.31, 0.50),
    ("Pre-Swing", 0.50, 0.62),
    ("Initial Swing", 0.62, 0.75),
    ("Mid Swing", 0.75, 0.87),
    ("Terminal Swing", 0.87, 1.00),
]
SUBPHASE_SHORT = ["LR", "MSt", "TSt", "PSw", "ISw", "MSw", "TSw"]
FOOT_FLAT = 0.08               # forefoot contact (end of heel rocker)
HEEL_OFF = 0.40                # heel rise in terminal stance
TOE_OFF = 0.62
DOUBLE_SUPPORT = 0.12          # each double-support period lasts 12 % of the cycle


def subphase_index(phase: float) -> int:
    phase = phase % 1.0
    for i, (_, lo, hi) in enumerate(SUBPHASES):
        if phase < hi:
            return i
    return len(SUBPHASES) - 1


def is_stance(phase: float) -> bool:
    return (phase % 1.0) < TOE_OFF


def right_weight_share(phase: float) -> float:
    """Fraction of body weight on the right leg: ramps in during loading response and
    out during pre-swing (the two double-support periods)."""
    p = phase % 1.0
    if p < DOUBLE_SUPPORT:
        return p / DOUBLE_SUPPORT
    if p < TOE_OFF - DOUBLE_SUPPORT:
        return 1.0
    if p < TOE_OFF:
        return (TOE_OFF - p) / DOUBLE_SUPPORT
    return 0.0


class FourierCurve:
    """Periodic curve as a truncated Fourier series (the classic representation of gait
    kinematics), fitted to a densely sampled spline. Call as curve(phase, nu) like a
    scipy spline, where nu is the derivative order w.r.t. phase."""

    def __init__(self, func, n_harmonics: int = 10, n_samples: int = 1024):
        p = np.arange(n_samples) / n_samples
        c = np.fft.rfft(func(p)) / n_samples
        self.a0 = c[0].real
        self.k = np.arange(1, n_harmonics + 1)
        self.a = 2 * c[1:n_harmonics + 1].real
        self.b = -2 * c[1:n_harmonics + 1].imag

    def __call__(self, phase, nu: int = 0):
        w = 2 * np.pi * self.k
        arg = np.multiply.outer(np.asarray(phase, dtype=float), w)
        # d^nu/dphase^nu of a cos + b sin
        cos_t, sin_t = np.cos(arg + nu * np.pi / 2), np.sin(arg + nu * np.pi / 2)
        val = (self.a * cos_t + self.b * sin_t) @ (w ** nu)
        return val + (self.a0 if nu == 0 else 0.0)


class GaitReference:
    """Smooth periodic joint trajectories. Phase in [0, 1); angles in radians."""

    N_LUT = 2000

    def __init__(self, stride_period: float = 1.2):
        self.stride_period = stride_period
        self._splines = {}
        grid = np.arange(self.N_LUT + 1) / self.N_LUT
        self._lut = {}
        for joint, pts in KEYPOINTS.items():
            pct, deg = np.array(pts, dtype=float).T
            self._splines[joint] = CubicSpline(pct / 100.0, np.deg2rad(deg), bc_type="periodic")
        self._splines["thigh"] = self._thigh_spline()
        # smooth every curve with a truncated Fourier series (bounded accelerations).
        # The thigh gets fewer harmonics: it rotates smoothly through stance, and its
        # acceleration couples strongly into the knee (inertial hip-knee coupling).
        self._splines = {j: FourierCurve(sp, n_harmonics=5 if j == "thigh" else 10)
                         for j, sp in self._splines.items()}
        for joint, sp in self._splines.items():
            self._lut[joint] = (sp(grid).tolist(), sp(grid, 1).tolist())
        self._arm_offset = np.deg2rad(10) - np.deg2rad(PELVIC_TILT_DEG)

    def curve(self, joint: str):
        """The smooth periodic curve object: curve(phase, derivative_order)."""
        return self._splines[joint]

    def _thigh_spline(self) -> CubicSpline:
        """Thigh angle relative to vertical (what the hip servo tracks).

        Outside mid-stance: normative hip flexion minus the mean pelvic tilt.
        Foot-flat window (8-38 %): thigh = knee - ankle, i.e. the stance foot stays flat
        on the belt, which keeps stance kinematics consistent in this planar model.
        """
        p = np.linspace(0, 1, 401)
        hip, knee, ankle = (self._splines[j](p) for j in ("hip", "knee", "ankle"))
        normative = hip - np.deg2rad(PELVIC_TILT_DEG)
        foot_flat = knee - ankle

        def smoothstep(x, a, b):
            t = np.clip((x - a) / (b - a), 0, 1)
            return t * t * (3 - 2 * t)

        w = smoothstep(p, 0.02, 0.08) * (1 - smoothstep(p, 0.38, 0.46))
        thigh = w * foot_flat + (1 - w) * normative
        thigh[-1] = thigh[0]
        return CubicSpline(p, thigh, bc_type="periodic")

    # vectorised, exact (for analysis / plotting)
    def angle(self, joint: str, phase):
        return self._splines[joint](np.mod(phase, 1.0))

    def velocity(self, joint: str, phase):
        """d(angle)/dt in rad/s at the current stride period."""
        return self._splines[joint](np.mod(phase, 1.0), 1) / self.stride_period

    def acceleration(self, joint: str, phase):
        return self._splines[joint](np.mod(phase, 1.0), 2) / self.stride_period ** 2

    # scalar, fast (for the 500 Hz control loop)
    def fast(self, joint: str, phase: float) -> tuple[float, float]:
        """(angle, velocity) by linear interpolation of a lookup table."""
        x = (phase % 1.0) * self.N_LUT
        i = int(x)
        f = x - i
        ang, dang = self._lut[joint]
        return (ang[i] + f * (ang[i + 1] - ang[i]),
                (dang[i] + f * (dang[i + 1] - dang[i])) / self.stride_period)

    # model joint -> (curve, phase offset, scale); hips track the thigh-absolute angle,
    # arms swing opposite to the ipsilateral thigh
    JOINT_MAP = {
        "r_hip": ("thigh", 0.0, 1.0), "r_knee": ("knee", 0.0, 1.0), "r_ankle": ("ankle", 0.0, 1.0),
        "l_hip": ("thigh", 0.5, 1.0), "l_knee": ("knee", 0.5, 1.0), "l_ankle": ("ankle", 0.5, 1.0),
        "r_shoulder": ("thigh", 0.0, -0.6), "l_shoulder": ("thigh", 0.5, -0.6),
    }

    def _offset(self, joint: str) -> float:
        return 0.6 * self._arm_offset if joint.endswith("shoulder") else 0.0

    def pose_full(self, phase: float) -> dict:
        """{joint: (angle, velocity)} for a given RIGHT-leg phase (fast, for the control loop)."""
        out = {}
        for j, (curve, off, k) in self.JOINT_MAP.items():
            q, qd = self.fast(curve, phase + off)
            out[j] = (k * q + self._offset(j), k * qd)
        return out

    def pose(self, phase: float) -> dict:
        return {j: q for j, (q, _) in self.pose_full(phase).items()}

    def kinematics(self, phase: float) -> dict:
        """{joint: (q, qd, qdd)} exact, for inverse dynamics."""
        T = self.stride_period
        out = {}
        for j, (curve, off, k) in self.JOINT_MAP.items():
            c, p = self._splines[curve], (phase + off) % 1.0
            out[j] = (k * float(c(p)) + self._offset(j), k * float(c(p, 1)) / T, k * float(c(p, 2)) / T ** 2)
        return out

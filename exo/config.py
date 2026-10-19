"""Central configuration: simulation, gait, virtual-patient presets and controller settings."""
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_XML = Path(__file__).resolve().parent / "model" / "knee_exo.xml"
RESULTS_DIR = ROOT / "results"


@dataclass
class SimConfig:
    dt: float = 0.001             # physics step (s)
    control_decimation: int = 2   # controller runs every N physics steps (500 Hz)
    duration: float = 30.0        # session length (s)
    render_fps: int = 30
    render_width: int = 960
    render_height: int = 540
    camera: str = "side"
    seed: int = 0


@dataclass
class GaitConfig:
    stride_period: float = 1.2    # s per gait cycle (≈ 0.8-0.9 m/s comfortable rehab speed)
    body_mass: float = 70.0       # kg (matches the Winter anthropometrics in the model)
    bws: float = 0.30             # body-weight support fraction carried by the harness
    belt_top: float = 0.099       # z of the treadmill belt surface (m)
    stride_cv: float = 0.02       # stride-to-stride period variability (coefficient of variation)


@dataclass
class SensorConfig:
    """Wearable sensor imperfections (knee encoder, thigh IMU, heel/toe insole switches)."""
    encoder_bits: int = 14            # absolute knee encoder resolution
    encoder_noise_deg: float = 0.05
    imu_noise_deg: float = 0.5        # thigh orientation (sensor-fusion output) noise
    imu_bias_deg: float = 1.0         # mounting misalignment
    gyro_noise: float = 0.05          # rad/s
    insole_latency: float = 0.015     # s, switch + transmission delay
    insole_bounce: float = 0.006      # s of contact chatter after every transition
    insole_false_rate: float = 0.1    # spurious contact pulses per second (vibration)
    torque_noise: float = 0.3         # exo motor torque sensing noise (N·m)
    insole_fault_at: float | None = None  # time (s) the insole disconnects (fault-injection demo)


@dataclass
class PatientConfig:
    name: str = "moderate"
    strength: float = 0.45        # quadriceps/hamstring strength, fraction of healthy
    spasticity: float = 6.0       # swing-phase stretch-reflex gain resisting knee flexion (N·m·s/rad)
    fatigue_rate: float = 1.0     # multiplier on fatigue accumulation
    recovery_rate: float = 1.0    # multiplier on recovery
    motor_noise: float = 0.05     # signal-dependent torque noise
    slacking: float = 0.5         # effort reduction when the exo carries the load
    tau_max_healthy: float = 120.0  # healthy peak knee torque (N·m)


PATIENT_PRESETS = {
    "healthy": PatientConfig(name="healthy", strength=1.0, spasticity=0.0, motor_noise=0.02, slacking=0.2),
    "mild": PatientConfig(name="mild", strength=0.70, spasticity=1.5),
    "moderate": PatientConfig(name="moderate", strength=0.45, spasticity=6.0),
    "severe": PatientConfig(name="severe", strength=0.25, spasticity=10.0, motor_noise=0.08),
    # strong quadriceps but hyperactive rectus femoris: isolates the swing-phase deficit
    "stiff-knee": PatientConfig(name="stiff-knee", strength=0.75, spasticity=14.0),
}


@dataclass
class ControllerConfig:
    kind: str = "cognitive"       # "none" | "fixed" | "cognitive"
    kp: float = 120.0             # impedance stiffness (N·m/rad)
    kd: float = 6.0               # impedance damping (N·m·s/rad)
    stance_stiffness_boost: float = 1.5
    fixed_gain: float = 1.0
    tau_max: float = 60.0         # safety torque limit (N·m)
    rate_max: float = 2000.0      # torque slew limit (N·m/s)
    n_bins: int = 50              # phase bins for learned gains / feedforward
    tolerance_deg: float = 3.0    # assist-as-needed tunnel: no corrective torque within +-tol
    g_init: float = 0.8
    aan_forgetting: float = 0.90  # assist-as-needed: g <- f*g + k*|dead(e)|
    aan_error_gain: float = 2.0   # 1/rad
    g_min: float = 0.1
    g_max: float = 2.0
    ilc_gain: float = 40.0        # iterative learning: u <- f_ff*u + gamma*dead(e)  (N·m/rad)
    ilc_forgetting: float = 0.90
    buckle_threshold_deg: float = 12.0


def clamp(x: float, lo: float, hi: float) -> float:
    """Scalar clip (np.clip on Python floats is ~20x slower in the 500 Hz loop)."""
    return lo if x < lo else hi if x > hi else x


def patient_preset(name: str, **overrides) -> PatientConfig:
    return replace(PATIENT_PRESETS[name], **overrides)

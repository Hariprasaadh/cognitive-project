"""Phase-3 tests: sensor model, gait-event / phase estimation and torque estimation,
always checked against simulator ground truth."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exo.config import SensorConfig, SimConfig, patient_preset  # noqa: E402
from exo.controllers import NoExo  # noqa: E402
from exo.gait import subphase_index  # noqa: E402
from exo.perception import Biquad, Debouncer, Observation, SensorModel, wrap_phase  # noqa: E402
from exo.session import Session  # noqa: E402


def run(preset: str, duration: float = 15.0, sensor_cfg=None):
    return Session(patient_preset(preset), NoExo(), SimConfig(duration=duration), sensor_cfg=sensor_cfg).run()


@pytest.fixture(scope="module")
def results():
    return {"moderate": run("moderate"), "stiff-knee": run("stiff-knee"),
            "fault": run("moderate", 18.0, SensorConfig(insole_fault_at=8.0))}


# ------------------------------------------------------------ building blocks
def test_debouncer_rejects_bounce():
    d = Debouncer(4)
    chatter = [True, False, True, False, True, True, False]
    assert not any(d(x) for x in chatter)            # never stable for 4 samples
    assert [d(True) for _ in range(4)][-1]           # stable contact is accepted


def test_biquad_unity_dc_gain_and_no_startup_transient():
    f = Biquad(20, 500)
    assert all(abs(f(3.0) - 3.0) < 1e-9 for _ in range(50))


def test_insole_latency_and_fault():
    cfg = SensorConfig(insole_bounce=0.0, insole_false_rate=0.0, insole_latency=0.010, insole_fault_at=1.0)
    s = SensorModel(cfg, dt=0.002)
    obs = lambda t, heel: Observation(t, 0, 0, 0, 0, heel, False, 0)   # noqa: E731
    out = [s.measure(obs(k * 0.002, k >= 10)).heel for k in range(30)]
    assert out.index(True) == 15                      # 10 ms = 5 samples late
    assert not s.measure(obs(1.2, True)).heel         # disconnected insole reads no contact


def test_phase_wrap():
    assert wrap_phase(0.98 - 0.02) == pytest.approx(-0.04)
    assert wrap_phase(0.02 - 0.98) == pytest.approx(0.04)


# ------------------------------------------------------------ closed-loop accuracy
@pytest.mark.parametrize("preset", ["moderate", "stiff-knee"])
def test_phase_estimate_within_5_percent(results, preset):
    p = results[preset].perception
    assert p["Phase error RMS (% cycle)"] < 5.0
    assert p["Phase error MAE (% cycle)"] < 3.0


@pytest.mark.parametrize("preset", ["moderate", "stiff-knee"])
def test_heel_strikes_detected_despite_noise(results, preset):
    p = results[preset].perception                    # bounce + spurious pulses + toe drags
    assert p["Heel strikes detected (%)"] == 100.0
    assert p["False heel-strike detections"] == 0
    assert p["Heel-strike timing error (ms)"] < 20.0


def test_subphase_matches_truth(results):
    log = results["moderate"].log
    log = log[log["t"] > 1.5]
    truth = log["phase"].map(subphase_index)
    assert (truth == log["subphase_hat"]).mean() > 0.9


def test_insole_fault_falls_back_to_phase_portrait(results):
    r = results["fault"]
    p, log = r.perception, r.log
    before = log[(log["t"] > 1.5) & (log["t"] < 7.5)]["phase_src"]
    assert (before == 0).all()                                    # insoles trusted before
    assert (log[log["t"] > 10.0]["phase_src"] == 1).all()        # portrait after the fault
    assert p["Phase error MAE, phase portrait (% cycle)"] < 5.0
    assert p["Phase error RMS (% cycle)"] < 5.0


def test_patient_torque_estimate_tracks_truth(results):
    p = results["moderate"].perception
    assert p["Patient torque estimate correlation r"] > 0.9
    assert p["Patient torque estimate RMSE (N·m)"] < 0.35 * p["Patient torque RMS (true, N·m)"]
    assert np.isfinite(results["moderate"].log["tau_patient_hat"]).all()

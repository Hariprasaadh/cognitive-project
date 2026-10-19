"""Phase-1 tests: gait reference and the treadmill gait world."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exo.gait import SUBPHASES, GaitReference, right_weight_share, subphase_index  # noqa: E402
from exo.sim import GaitSim  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    return GaitSim()


def test_reference_is_periodic_and_normative():
    ref = GaitReference(1.2)
    for joint in ("hip", "knee", "ankle", "thigh"):
        assert ref.fast(joint, 0.0)[0] == pytest.approx(ref.fast(joint, 0.9999)[0], abs=1e-3)
    knee = np.rad2deg(ref.angle("knee", np.linspace(0, 1, 500)))
    assert 55 < knee.max() < 68            # peak swing flexion ~60 deg
    assert np.argmax(knee) / 500 > 0.62    # peak occurs in swing
    assert knee.min() < 6


def test_fast_lookup_matches_spline():
    ref = GaitReference(1.0)
    for p in np.random.default_rng(0).random(50):
        a, v = ref.fast("knee", p)
        assert a == pytest.approx(float(ref.angle("knee", p)), abs=1e-4)
        assert v == pytest.approx(float(ref.velocity("knee", p)), abs=2e-2)


def test_subphases_and_weight_share():
    assert SUBPHASES[subphase_index(0.05)][0] == "Loading Response"
    assert SUBPHASES[subphase_index(0.80)][0] == "Mid Swing"
    assert right_weight_share(0.3) == 1.0 and right_weight_share(0.8) == 0.0
    assert 0 < right_weight_share(0.06) < 1


def test_stance_foot_is_flat_kinematically(sim):
    ref = sim.ref
    for p in (0.12, 0.2, 0.3):
        pose = ref.pose(p)
        pitch = pose["r_hip"] - pose["r_knee"] + pose["r_ankle"]
        assert abs(np.rad2deg(pitch)) < 2.0


def test_walk_tracks_and_stays_finite(sim):
    sim.reset()
    T, err = sim.ref.stride_period, []
    while sim.time < 3.0:
        phase = (sim.time / T) % 1.0
        r, dr = sim.ref.fast("knee", phase)
        sim.step(phase, 400 * (r - sim.knee_angle) + 20 * (dr - sim.knee_vel), 0.0, n=2)
        if sim.time > 1.2:
            err.append(r - sim.knee_angle)
    assert np.all(np.isfinite(sim.data.qpos))
    assert np.rad2deg(np.sqrt(np.mean(np.square(err)))) < 4.0
    assert 0.9 < sim.data.mocap_pos[sim.pelvis_mocap][2] < 1.1
    assert 0.4 < sim.belt_speed < 1.5


def test_insole_timing(sim):
    sim.phase = 0.02
    assert sim.insole("r") == (True, False)      # heel strike: heel only
    sim.phase = 0.25
    assert sim.insole("r") == (True, True)       # foot flat
    sim.phase = 0.70
    assert sim.insole("r")[0] is False           # swing: heel off


def test_render_shape(sim):
    img = sim.render(width=320, height=180)
    assert img.shape == (180, 320, 3) and img.std() > 5

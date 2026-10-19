"""Phase-2 tests: virtual patient, session loop and metrics (no assistance)."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exo.config import SimConfig, patient_preset  # noqa: E402
from exo.controllers import NoExo  # noqa: E402
from exo.session import LOG_COLUMNS, Session  # noqa: E402


def run(preset: str, duration: float = 12.0, **kw):
    return Session(patient_preset(preset, **kw), NoExo(), SimConfig(duration=duration)).run()


@pytest.fixture(scope="module")
def results():
    return {p: run(p) for p in ("healthy", "mild", "moderate", "severe", "stiff-knee")}


def test_healthy_walks_normally(results):
    m = results["healthy"].metrics
    assert m["Knee RMSE vs normative (deg)"] < 6.0
    assert m["Strides with knee buckling (%)"] == 0.0
    assert 55 < m["Peak swing flexion (deg)"] < 68


def test_deficits_scale_with_severity(results):
    rmse = [results[p].metrics["Knee RMSE vs normative (deg)"] for p in ("healthy", "mild", "moderate", "severe")]
    assert rmse == sorted(rmse)
    assert results["moderate"].metrics["Strides with knee buckling (%)"] > 50
    assert results["severe"].metrics["Max stance excess flexion (deg)"] > results["moderate"].metrics["Max stance excess flexion (deg)"]


def test_stiff_knee_is_a_swing_deficit(results):
    sk, mild = results["stiff-knee"].metrics, results["mild"].metrics
    assert sk["Swing flexion deficit (deg)"] > mild["Swing flexion deficit (deg)"] + 3
    assert sk["Toe scuffs per stride"] > 0
    assert sk["Strides with knee buckling (%)"] == 0.0


def test_log_and_strides_are_consistent(results):
    r = results["moderate"]
    assert list(r.log.columns) == LOG_COLUMNS
    assert np.isfinite(r.log.select_dtypes("number").to_numpy()).all()
    assert len(r.strides) == r.log["stride"].max() - 1    # first + last stride excluded
    assert abs(len(r.strides) - (int(12.0 / 1.2) - 2)) <= 1   # stride periods vary by ~2 %
    f = r.log["fatigue"].to_numpy()                       # accumulates (with slight recovery dips)
    assert f[-1] > f[len(f) // 2] > f[len(f) // 10] > 0
    assert (r.log["tau_exo"] == 0).all()                  # NoExo is transparent


def test_bws_reduces_buckling():
    low = run("moderate", 8.0).metrics["Max stance excess flexion (deg)"]
    from exo.config import GaitConfig
    high = Session(patient_preset("moderate"), NoExo(), SimConfig(duration=8.0),
                   GaitConfig(bws=0.6)).run().metrics["Max stance excess flexion (deg)"]
    assert high < low


def test_incremental_advance_matches_run():
    a = run("moderate", 6.0).metrics
    s = Session(patient_preset("moderate"), NoExo(), SimConfig(duration=6.0))
    while not s.done:
        s.advance(0.37)
    b = s.result().metrics
    assert a["Knee RMSE vs normative (deg)"] == pytest.approx(b["Knee RMSE vs normative (deg)"], rel=1e-9)


def test_session_is_fast():
    t0 = time.perf_counter()
    run("moderate", 20.0)
    assert time.perf_counter() - t0 < 6.0

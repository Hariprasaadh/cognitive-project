"""Phase-4 tests: safety layer, controller behaviour and the cognitive supervisor (SIMULATED)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exo.config import ControllerConfig, SimConfig, patient_preset  # noqa: E402
from exo.controllers import MODES, CognitiveAdaptive, FixedImpedance, deadband, make_controller  # noqa: E402
from exo.perception import Percept  # noqa: E402
from exo.session import Session  # noqa: E402
from exo.sim import GaitSim  # noqa: E402


def percept(**kw) -> Percept:
    base = dict(t=0.0, knee=0.3, knee_vel=0.0, knee_acc=0.0, thigh=0.0, thigh_vel=0.0, heel=True, toe=True,
                tau_exo=0.0, phase=0.2, phase_source=0, stride_period=1.2, stride=1, subphase=1, stance=True,
                toe_scuff=False, heel_strike=False, toe_off=False, insole_ok=True, tau_human=0.0, tau_load=0.0,
                tau_patient=0.0)
    base.update(kw)
    return Percept(**base)


def run(preset, kind, duration=30.0, **pkw):
    s = Session(patient_preset(preset, **pkw), make_controller(kind), SimConfig(duration=duration))
    return s, s.run()


@pytest.fixture(scope="module")
def sim():
    return GaitSim()


@pytest.fixture(scope="module")
def moderate():
    return {k: run("moderate", k)[1] for k in ("none", "fixed", "cognitive")}


# ------------------------------------------------------------ building blocks
def test_make_controller_kinds():
    for kind, cls in (("fixed", FixedImpedance), ("cognitive", CognitiveAdaptive)):
        assert isinstance(make_controller(kind), cls)
    with pytest.raises(ValueError):
        make_controller("magic")


def test_deadband_tunnel():
    assert deadband(0.01, 0.05) == 0.0
    assert deadband(0.08, 0.05) == pytest.approx(0.03)
    assert deadband(-0.08, 0.05) == pytest.approx(-0.03)


def test_safety_layer_limits_torque_rate_and_hyperextension(sim):
    c = FixedImpedance(ControllerConfig(tau_max=40.0, rate_max=2000.0))
    c.reset(sim)
    taus = [c.safety(percept(), 500.0)[0] for _ in range(30)]
    assert max(taus) <= 40.0                                   # torque limit
    assert np.max(np.diff([0.0] + taus)) <= 2000.0 * c.dt + 1e-9   # slew limit
    c.reset(sim)
    assert c.safety(percept(knee=np.deg2rad(-3)), -30.0)[0] == 0.0   # never extend into hyperextension
    c.reset(sim)
    assert c.safety(percept(knee_vel=20.0), 30.0)[0] == 0.0          # overspeed cut-off


# ------------------------------------------------------------ closed-loop behaviour
def test_any_assistance_prevents_moderate_buckling(moderate):
    assert moderate["none"].metrics["Strides with knee buckling (%)"] > 50
    for k in ("fixed", "cognitive"):
        assert moderate[k].metrics["Strides with knee buckling (%)"] == 0
        assert moderate[k].metrics["Knee RMSE vs normative (deg)"] < 5.0


def test_cognitive_assists_as_needed(moderate):
    fix, cog = moderate["fixed"].metrics, moderate["cognitive"].metrics
    assert cog["Assistance ratio (%)"] < 0.75 * fix["Assistance ratio (%)"]   # less assistance
    assert cog["Final effort (%)"] > fix["Final effort (%)"]                   # less slacking
    assert cog["Exo peak torque (N·m)"] <= 60.0 + 1e-6


def test_cognitive_is_safer_for_severe_patient():
    fix = run("severe", "fixed", 40.0)[1].metrics
    cog = run("severe", "cognitive", 40.0)[1].metrics
    assert cog["Strides with knee buckling (%)"] < fix["Strides with knee buckling (%)"]
    assert cog["Knee RMSE vs normative (deg)"] < fix["Knee RMSE vs normative (deg)"]


def test_decisions_are_logged_and_explained(moderate):
    events = moderate["cognitive"].events
    assert events and all(isinstance(t, float) and len(msg) > 20 for t, msg in events)
    assert set(moderate["cognitive"].log["mode"].unique()) <= set(MODES)


def test_supervisor_tells_fatigue_from_slacking():
    s, r = run("moderate", "cognitive", 60.0, fatigue_rate=5.0)
    msgs = " ".join(m for _, m in r.events)
    assert "FATIGUE_SUPPORT" in msgs and "RECOMMENDATION" in msgs
    assert r.log["mode"].iloc[-1] == "FATIGUE_SUPPORT"
    assert len(s.controller.history) > 40 and np.isfinite(s.controller.gain).all()

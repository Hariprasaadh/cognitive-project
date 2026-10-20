"""Phase-5 smoke tests: the Streamlit dashboard renders and runs a session headlessly; viz helpers work."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exo  # noqa: E402,F401
from exo.config import SimConfig, patient_preset  # noqa: E402
from exo.controllers import make_controller  # noqa: E402
from exo.session import Session  # noqa: E402
from exo.viz import save_mp4, skeleton_animation  # noqa: E402

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_skeleton_animation_and_mp4(tmp_path):
    s = Session(patient_preset("moderate"), make_controller("cognitive"), SimConfig(duration=1.0), skeleton_hz=10)
    s.run(render_camera="side")
    fig = skeleton_animation(s.skeleton_frames, title="test")
    assert len(fig.frames) == len(s.skeleton_frames) >= 9
    out = save_mp4(s.frames[:10], tmp_path / "clip.mp4", fps=10)
    assert out.stat().st_size > 1000
    s.sim.close()


def test_dashboard_renders_and_runs_headless_session():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    assert any("SIMULATED" in c.value for c in at.sidebar.caption)  # simulation disclaimer in the sidebar
    at.sidebar.slider[-1].set_value(10)                                 # shortest session
    next(b for b in at.button if "headless" in b.label).click()
    at.run()
    assert not at.exception
    r = at.session_state["result"]["result"]
    assert r.metrics["Strides analysed"] >= 5

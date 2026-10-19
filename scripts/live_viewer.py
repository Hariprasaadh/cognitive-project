"""Interactive native MuJoCo 3D viewer of the cognitive knee exoskeleton (SIMULATED).

Run:  python scripts/live_viewer.py [--patient moderate] [--controller cognitive]

Mouse: rotate / zoom / pan the camera as in any MuJoCo viewer.
Keys:  1 = no assistance   2 = fixed impedance   3 = cognitive adaptive
       4 = inject +30 % fatigue   5 = unplug the insole (phase-portrait fallback)
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import exo  # noqa: E402,F401  (MSVC runtime preload before mujoco)
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from exo.config import PATIENT_PRESETS, ControllerConfig, SimConfig, patient_preset  # noqa: E402
from exo.controllers import make_controller  # noqa: E402
from exo.gait import SUBPHASES  # noqa: E402
from exo.perception import PHASE_SOURCES  # noqa: E402
from exo.session import Session  # noqa: E402

KINDS = {ord("1"): "none", ord("2"): "fixed", ord("3"): "cognitive"}
LABEL = {"none": "No assistance", "fixed": "Fixed impedance", "cognitive": "Cognitive adaptive"}
FIG_POINTS = 300                     # 3 s of knee history at 100 Hz


def make_figure() -> mujoco.MjvFigure:
    fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(fig)
    fig.title = "Knee angle (deg)"
    fig.flg_extend = 0
    fig.range[0][0], fig.range[0][1] = -FIG_POINTS / 100, 0     # last 3 s
    fig.range[1][0], fig.range[1][1] = -10, 80
    fig.flg_ticklabel[0], fig.flg_ticklabel[1] = 1, 1
    fig.gridsize[0], fig.gridsize[1] = 4, 5
    fig.figurergba[3] = 0.6
    fig.linergb[0] = (0.30, 0.66, 0.87)
    fig.linergb[1] = (1.0, 1.0, 1.0)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", default="moderate", choices=list(PATIENT_PRESETS))
    ap.add_argument("--controller", default="cognitive", choices=list(LABEL))
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-close after N s (0 = run until closed)")
    args = ap.parse_args()

    s = Session(patient_preset(args.patient), make_controller(args.controller), SimConfig(duration=1e9))
    pending: deque = deque()
    kind = {"now": args.controller}

    def on_key(key: int):
        pending.append(key)                       # handled in the main loop (thread-safe)

    knee_hist, ref_hist = deque(maxlen=FIG_POINTS), deque(maxlen=FIG_POINTS)
    fig = make_figure()
    print(__doc__)
    with mujoco.viewer.launch_passive(s.sim.model, s.sim.data, key_callback=on_key,
                                      show_left_ui=False, show_right_ui=False) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = s.sim.model.body("pelvis").id
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.0, 90.0, -8.0
        t_start = time.perf_counter()
        while v.is_running():
            tick = time.perf_counter()
            while pending:
                key = pending.popleft()
                if key in KINDS:
                    kind["now"] = KINDS[key]
                    events = list(s.controller.events)
                    s.set_controller(make_controller(ControllerConfig(kind=kind["now"])))
                    s.controller.events.extend(events + [(round(s.time, 2), f"USER: switched to {LABEL[kind['now']]}")])
                elif key == ord("4"):
                    s.inject_fatigue(0.3)
                    s.controller.log(s.time, "USER: fatigue injected (+30 %)")
                elif key == ord("5"):
                    s.fail_insole()
                    s.controller.log(s.time, "USER: insole unplugged")
            with v.lock():
                s.advance(1 / 60)
                s.sim.apply_colors()
            pc, info, p = s.percept, s.last_info, s.patient
            knee_hist.append(np.rad2deg(s.sim.knee_angle))
            ref_hist.append(np.rad2deg(s.sim.ref.fast("knee", s.phase)[0]))
            n = len(knee_hist)
            for line, hist in ((0, knee_hist), (1, ref_hist)):
                fig.linepnt[line] = n
                for i, y in enumerate(hist):
                    fig.linedata[line][2 * i] = (i - n) / 100
                    fig.linedata[line][2 * i + 1] = y
            last = s.controller.events[-1][1] if s.controller.events else "-"
            left = ("SIMULATED DATA\nController\nMode\nGait phase (est.)\nPhase source\nKnee / normative\n"
                    "Exo torque\nPatient fatigue\nPatient effort\nLast decision")
            right = (f"virtual patient: {args.patient}\n{LABEL[kind['now']]}\n{info.get('mode', 'OFF')}\n"
                     f"{100 * pc.phase:3.0f} %  {SUBPHASES[pc.subphase][0]}\n{PHASE_SOURCES[pc.phase_source]}\n"
                     f"{knee_hist[-1]:4.0f} / {ref_hist[-1]:4.0f} deg\n{s.tau_exo:+5.1f} N.m\n"
                     f"{100 * p.fatigue:4.1f} %\n{100 * p.effort:4.0f} %\n{last[:70]}")
            keys = "1  no assistance\n2  fixed impedance\n3  cognitive adaptive\n4  inject fatigue\n5  unplug insole"
            v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right),
                         (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, "Keys", keys)])
            vp = v.viewport
            if vp is not None:
                w, h = vp.width // 3, vp.height // 4
                v.set_figures((mujoco.MjrRect(10, 10, w, h), fig))      # bottom-left corner
            v.sync()
            if args.seconds and time.perf_counter() - t_start > args.seconds:
                break
            time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - tick)))


if __name__ == "__main__":
    main()

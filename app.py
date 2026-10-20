"""Cognitive Exoskeleton for Knee Rehabilitation - interactive Streamlit dashboard.

Run:  streamlit run app.py
Every number and plot on this dashboard comes from a physics SIMULATION (MuJoCo) with a
virtual patient - nothing is measured on a human.
"""
import exo  # noqa: F401  (Windows MSVC runtime preload - must be imported before mujoco)

import base64
import io
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

from exo.config import PATIENT_PRESETS, RESULTS_DIR, ControllerConfig, GaitConfig, SensorConfig, SimConfig, patient_preset
from exo.controllers import MODE_COLORS, MODES, make_controller
from exo.gait import SUBPHASES, TOE_OFF, GaitReference
from exo.metrics import NORMATIVE_PEAK_SWING_DEG, gait_cycle_curves, stride_table
from exo.perception import PHASE_SOURCES
from exo.session import Session
from exo.viz import HW_COMPONENTS, hardware_animation, save_mp4, skeleton_animation

st.set_page_config(page_title="Cognitive Knee Exoskeleton", page_icon="🦿", layout="wide")

KIND_LABEL = {"none": "No assistance", "fixed": "Fixed impedance", "cognitive": "Cognitive adaptive"}
KIND_COLOR = {"none": "#9e9e9e", "fixed": "#4ea8de", "cognitive": "#e76f51"}
SUB_NAMES = [s[0] for s in SUBPHASES]

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
.tile {background:#1a1d27; border:1px solid #2a2f3d; border-radius:12px; padding:10px 12px; min-height:112px;
       margin-bottom:.5rem; overflow:hidden;}
.tile, .tile * {word-break:normal; overflow-wrap:normal; hyphens:none;}   /* wrap at spaces, never mid-word */
.tile .lbl {font-size:.68rem; color:#9aa3b5; text-transform:uppercase; letter-spacing:.4px; line-height:1.3;
            min-height:1.8em;}
.tile .val {font-size:clamp(1rem, 1.5vw, 1.4rem); font-weight:700; color:#f1f3f8; line-height:1.2; margin:2px 0;}
.tile .val.long {font-size:clamp(.85rem, 1.1vw, 1.05rem); letter-spacing:.2px;}
.tile .sub {font-size:.72rem; color:#9aa3b5; line-height:1.3;}
.logline {font-family: ui-monospace, Consolas, monospace; font-size:.80rem; padding:3px 0;
          border-bottom:1px solid #222736;}
.logline b {color:#f4a261;}
</style>""", unsafe_allow_html=True)


def tile(col, label, value, sub="", color=None):
    style = f' style="color:{color}"' if color else ""
    cls = "val long" if len(str(value)) > 8 else "val"     # e.g. controller mode names
    col.markdown(f'<div class="tile"><div class="lbl">{label}</div><div class="{cls}"{style}>{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def log_html(events, n=10):
    if not events:
        return '<div class="logline">No decisions yet.</div>'
    return "".join(f'<div class="logline"><b>t={t:5.1f}s</b> &nbsp;{m}</div>' for t, m in events[-n:][::-1])


LIVE_FPS, LIVE_FRAMES = 10, 5          # each live tick = 0.5 s of simulation shown as one 10 fps clip


def clip_html(frames: list, fps: float) -> str:
    """Frames -> one inline animated WebP that plays once. One payload per tick instead of one per
    frame keeps the browser in step and avoids stale media-URL 404s."""
    imgs = [Image.fromarray(f) for f in frames]
    buf = io.BytesIO()
    imgs[0].save(buf, "WEBP", save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=1,
                 quality=72, method=0)
    return (f'<img src="data:image/webp;base64,{base64.b64encode(buf.getvalue()).decode()}" '
            f'style="width:100%;border-radius:10px;display:block">')


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


# ======================================================================== sidebar
def sidebar() -> dict:
    sb = st.sidebar
    sb.title("🦿 Cognitive Knee Exo")
    sb.caption("Software-only rehabilitation robotics demo · MuJoCo + Streamlit")
    sb.header("Patient")
    preset = sb.selectbox("Profile", list(PATIENT_PRESETS), index=2,
                          help="Virtual patients with different neuromuscular impairments")
    base = PATIENT_PRESETS[preset]
    with sb.expander("Customise impairment", expanded=False):
        strength = st.slider("Quadriceps strength (× healthy)", 0.15, 1.0, float(base.strength), 0.05)
        spasticity = st.slider("Spasticity (N·m·s/rad)", 0.0, 16.0, float(base.spasticity), 0.5)
        fatigue_rate = st.slider("Fatigue rate (×)", 0.0, 8.0, float(base.fatigue_rate), 0.5)
        slacking = st.slider("Slacking tendency", 0.0, 1.0, float(base.slacking), 0.05)
    sb.header("Treadmill")
    bws = sb.slider("Body-weight support (%)", 0, 60, 30, 5) / 100
    stride = sb.slider("Stride period (s)", 1.0, 1.6, 1.2, 0.05, help="Slower cadence = longer stride period")
    sb.header("Exoskeleton controller")
    kind = sb.radio("Controller", list(KIND_LABEL), index=2, format_func=KIND_LABEL.get)
    with sb.expander("Controller parameters", expanded=False):
        kp = st.slider("Stiffness K (N·m/rad)", 40, 250, 120, 10)
        tol = st.slider("Assist-as-needed tunnel (± deg)", 0.0, 8.0, 3.0, 0.5, help="Cognitive controller only")
        tau_max = st.slider("Safety torque limit (N·m)", 20, 80, 60, 5)
    sb.header("Sensors")
    fault = sb.checkbox("Unplug insole during session", help="Tests the thigh phase-portrait fallback")
    fault_at = sb.slider("…at time (s)", 3, 50, 12) if fault else None
    duration = sb.slider("Session length (s)", 10, 60, 30, 5)
    sb.markdown("---")
    sb.caption("All results are **SIMULATED**. Patient models are parameterised from the rehab literature "
               "(see *How it works*).")
    return {
        "patient": patient_preset(preset, strength=strength, spasticity=spasticity, fatigue_rate=fatigue_rate,
                                  slacking=slacking),
        "gait": GaitConfig(stride_period=stride, bws=bws),
        "ctrl": ControllerConfig(kind=kind, kp=float(kp), tolerance_deg=float(tol), tau_max=float(tau_max)),
        "sensor": SensorConfig(insole_fault_at=float(fault_at) if fault_at else None),
        "duration": float(duration), "preset": preset, "kind": kind,
    }


def with_kind(cfg: dict, kind: str) -> dict:
    return {**cfg, "kind": kind, "ctrl": ControllerConfig(**{**cfg["ctrl"].__dict__, "kind": kind})}


def make_session(cfg, skeleton_hz=20.0, duration=None) -> Session:
    sensor = SensorConfig(**cfg["sensor"].__dict__)              # fresh copy (sessions mutate it)
    return Session(cfg["patient"], make_controller(cfg["ctrl"]), SimConfig(duration=duration or cfg["duration"]),
                   cfg["gait"], sensor_cfg=sensor, skeleton_hz=skeleton_hz)


def pack_result(s: Session, cfg, video=None) -> dict:
    return {"result": s.result(), "skeleton": list(s.skeleton_frames),
            "history": list(getattr(s.controller, "history", [])),
            "label": f"{cfg['preset']} patient · {KIND_LABEL[cfg['kind']]}", "video": video,
            "tau_max": cfg["ctrl"].tau_max, "kind": cfg["kind"]}


# ======================================================================== live tab
def live_figure(df: pd.DataFrame, tol_deg: float, show_tunnel: bool, df_twin: pd.DataFrame | None = None) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.42, 0.33, 0.25],
                        subplot_titles=("Right knee angle (deg)", "Knee torques (N·m, + flexion)",
                                        "Gait phase: true vs estimated by the exo"))
    t = df["t"]
    ref = np.rad2deg(df["knee_ref"])
    if show_tunnel:
        fig.add_trace(go.Scatter(x=t, y=ref + tol_deg, line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=ref - tol_deg, fill="tonexty", fillcolor="rgba(244,162,97,0.14)",
                                 line=dict(width=0), name=f"assist-as-needed tunnel ±{tol_deg:g}°"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=ref, name="normative", line=dict(color="#e8eaf0", dash="dash", width=1.5)), 1, 1)
    if df_twin is not None and len(df_twin):
        fig.add_trace(go.Scatter(x=df_twin["t"], y=np.rad2deg(df_twin["knee"]), name="knee WITHOUT exo (same patient)",
                                 line=dict(color="#e63946", width=2, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=np.rad2deg(df["knee"]), name="knee WITH exo" if df_twin is not None else "knee (simulated)",
                             line=dict(color="#4ea8de", width=2.5)), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=df["tau_exo"], name="exo torque", line=dict(color="#f4a261", width=2)), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=df["tau_patient"], name="patient muscle torque (true)",
                             line=dict(color="#90be6d", width=1.5)), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=df["tau_patient_hat"], name="patient torque (exo estimate)",
                             line=dict(color="#90be6d", width=1.5, dash="dot")), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=df["phase"], name="true phase", line=dict(color="#6c757d", width=4)), 3, 1)
    fig.add_trace(go.Scatter(x=t, y=df["phase_hat"], name="estimated phase", line=dict(color="#e8eaf0", width=1.3)), 3, 1)
    modes = df["mode"].astype(str)
    for m in [m for m in MODES if (modes == m).any()]:
        sel = modes == m
        fig.add_trace(go.Scatter(x=t[sel], y=np.full(int(sel.sum()), 1.12), mode="markers", name=m,
                                 marker=dict(color=MODE_COLORS[m], size=5, symbol="square")), 3, 1)
    fig.update_yaxes(range=[-0.05, 1.2], row=3, col=1)
    fig.update_layout(height=600, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.08, font=dict(size=10)), uirevision="live",
                      paper_bgcolor="#0f1117", plot_bgcolor="#141824")
    return fig


def live_tab(cfg):
    ss = st.session_state
    c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1.2, 1.6])
    twin_on = c5.checkbox("Show the same patient WITHOUT the exo", value=True, key="twin_on",
                          help="Runs a second, identical virtual patient with no assistance next to the main one")
    if c1.button("▶ Start new session", type="primary", width="stretch"):
        ss.live = make_session(cfg)
        ss.twin = make_session(with_kind(cfg, "none")) if twin_on else None
        ss.live_cfg = cfg
        ss.running = True
        ss.pop("tick_cost", None)
    if c2.button("⏸ Pause / ▶ resume", width="stretch", disabled="live" not in ss):
        ss.running = not ss.get("running", False)
    ss.camera = c3.selectbox("Camera", ["side", "knee", "iso", "front"], label_visibility="collapsed")
    ss.speed = c4.select_slider("Speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0,
                                format_func=lambda v: f"{v:g}× real time", label_visibility="collapsed")
    st.caption("Start a session, then intervene live: inject fatigue, unplug the insole, or hot-swap the controller.")
    # The live loop never lets the script run finish, so Streamlit never clears elements left over from the
    # previous run. Keep the intro in one slot that a session run explicitly empties.
    intro = st.empty()
    if "live" not in ss:
        with intro.container():
            st.info("Configure the patient and controller in the sidebar, then press **Start new session**. "
                    "The MuJoCo world, the virtual patient, the sensors and the exoskeleton controller then run live.")
            img = RESULTS_DIR / "phase1_side.png"
            if img.exists():
                st.image(str(img), caption="The simulated world: treadmill, body-weight-support harness, "
                                           "right-knee exoskeleton (SIMULATED)", width="stretch")
        return
    # Not an st.fragment: a click inside a fragment waits for the page run to finish, and this loop never
    # lets it finish, so the live buttons were ignored. As part of the page run, a click interrupts the loop.
    live_panel()


def stride_stats(sess: Session) -> dict:
    """Per-stride metrics so far, recomputed only when a new stride has finished."""
    cache = getattr(sess, "_stats_cache", None)
    if cache is None or cache[0] != sess.stride:
        t = stride_table(sess.dataframe())
        stats = ({"n": len(t), "rmse": t["rmse_deg"].mean(), "buckled": int(t["buckled"].sum()),
                  "scuffs": t["scuffs"].mean(), "swing": t["peak_swing_deg"].mean()} if len(t) else {})
        sess._stats_cache = cache = (sess.stride, stats)
    return cache[1]


def twin_tiles(twin: Session, s: Session):
    """Live scoreboard: same patient without the exo (left value) vs with it (right value)."""
    a, b = stride_stats(twin), stride_stats(s)
    c = st.columns(4)
    if not a or not b:
        tile(c[0], "Why the exo matters", "…", "scoreboard appears after the first full strides")
        return
    n = a["n"]
    bad, good = "#e63946", "#90be6d"
    tile(c[0], "Knee error vs normal gait", f"{a['rmse']:.1f}° → {b['rmse']:.1f}°", "without → with exo", good if b["rmse"] < a["rmse"] else bad)
    tile(c[1], "Strides where the knee buckled", f"{a['buckled']} / {n} → {b['buckled']} / {b['n']}",
         "knee collapses in stance without help", good if b["buckled"] <= a["buckled"] else bad)
    tile(c[2], "Toe scuffs per stride", f"{a['scuffs']:.2f} → {b['scuffs']:.2f}", "foot dragging in swing", good if b["scuffs"] <= a["scuffs"] else bad)
    tile(c[3], "Peak swing knee flexion", f"{a['swing']:.0f}° → {b['swing']:.0f}°", f"normal gait {NORMATIVE_PEAK_SWING_DEG:.0f}°")


def live_panel():
    ss = st.session_state
    s: Session = ss.live
    cfg = ss.live_cfg
    twin: Session | None = ss.get("twin")
    size = (480, 270) if twin is not None else (720, 405)      # two renders per frame: keep real-time pace
    if twin is not None:       # side-by-side: the same patient with and without the exoskeleton
        v1, v2 = st.columns(2)
        v1.markdown("##### 🚫 Without exoskeleton")
        twin_ph = v1.empty()
        v2.markdown(f"##### ✅ With exoskeleton ({KIND_LABEL[cfg['kind']]})")
        img_ph = v2.empty()
        cmp_ph = st.empty()
    left, right = st.columns([1.05, 1])
    with left:
        if twin is None:
            img_ph = st.empty()
        b1, b2 = st.columns(2)
        if b1.button("💤 Inject fatigue (+30%)", width="stretch", disabled=s.done):
            s.inject_fatigue(0.3)
            if twin is not None:
                twin.inject_fatigue(0.3)
            s.controller.log(s.time, "USER: fatigue injected (+30 %) by the therapist")
        if b2.button("🔌 Unplug insole", width="stretch", disabled=s.done):
            s.fail_insole()
            s.controller.log(s.time, "USER: insole disconnected")
        swap = st.selectbox("Hot-swap controller", list(KIND_LABEL), format_func=KIND_LABEL.get,
                            index=list(KIND_LABEL).index(cfg["kind"]), key=f"swap_{id(s)}", disabled=s.done)
        if swap != cfg["kind"]:
            events = list(s.controller.events)
            cfg = ss.live_cfg = with_kind(cfg, swap)
            s.set_controller(make_controller(cfg["ctrl"]))
            s.controller.events.extend(events + [(round(s.time, 2), f"USER: controller switched to {KIND_LABEL[swap]}")])
        tiles_ph = st.empty()
        st.markdown("**Explainable decision log** (newest first)")
        log_ph = st.empty()
    with right:
        chart_ph = st.empty()

    # Single in-place loop: placeholders are overwritten each tick, so the page never reruns/flickers.
    # Clicking any widget interrupts the loop and the rerun restarts it with the new state.
    while True:
        # ---- simulate one tick, then send ONE animated clip + one UI update (browser-friendly)
        if ss.get("running") and not s.done:
            t_tick = time.perf_counter()
            frames, twin_frames = [], []
            for _ in range(LIVE_FRAMES):
                if s.done:
                    break
                s.advance(1 / LIVE_FPS)
                frames.append(s.sim.render(ss.camera, *size))
                if twin is not None:
                    twin.advance(1 / LIVE_FPS)
                    twin_frames.append(twin.sim.render(ss.camera, *size))
            # If a tick takes longer to compute than its clip lasts, stretch the clip to the real cost so
            # playback stays continuous (slightly slower than real time) instead of freezing between clips.
            play_fps = LIVE_FPS / ss.speed * min(1.0, (LIVE_FRAMES / LIVE_FPS / ss.speed) / max(ss.get("tick_cost", 0.0), 1e-3))
            img_ph.markdown(clip_html(frames, play_fps), unsafe_allow_html=True)
            if twin is not None:
                twin_ph.markdown(clip_html(twin_frames, play_fps), unsafe_allow_html=True)
            ss.last_frame = frames[-1] if frames else ss.get("last_frame")
            ss.tick_wall = (t_tick, len(frames) / LIVE_FPS / ss.speed)
        else:
            img_ph.markdown(clip_html([s.sim.render(ss.camera, *size)], 1), unsafe_allow_html=True)
            if twin is not None:
                twin_ph.markdown(clip_html([twin.sim.render(ss.camera, *size)], 1), unsafe_allow_html=True)

        if twin is not None:
            with cmp_ph.container():
                twin_tiles(twin, s)
        p, pc, info = s.patient, s.percept, s.last_info
        mode = info.get("mode", "OFF")
        insole_ok = pc is None or pc.insole_ok
        with tiles_ph.container():
            r1 = st.columns(4)
            tile(r1[0], "Controller mode", mode.replace("_", " "), KIND_LABEL[cfg["kind"]], MODE_COLORS.get(mode, "#9aa3b5"))
            if pc is not None:
                tile(r1[1], "Gait phase (estimated)", f"{100 * pc.phase:4.0f} %",
                     f"{SUB_NAMES[pc.subphase]} · via {PHASE_SOURCES[pc.phase_source]}")
            tile(r1[2], "Knee angle", f"{np.rad2deg(s.sim.knee_angle):4.0f}°",
                 f"normative {np.rad2deg(s.sim.ref.fast('knee', s.phase)[0]):.0f}°")
            tile(r1[3], "Exo torque", f"{s.tau_exo:+5.1f} N·m",
                 f"assist gain {info['gain']:.2f}" if "gain" in info else "unpowered")
            r2 = st.columns(4)
            tile(r2[0], "Sim time", f"{s.time:4.1f} s", f"of {s.sim_cfg.duration:.0f} s · stride {s.stride}")
            tile(r2[1], "Patient fatigue (true)", f"{100 * p.fatigue:4.1f} %", "simulated ground truth")
            tile(r2[2], "Patient effort (true)", f"{100 * p.effort:4.0f} %", "drops when the exo does the work")
            tile(r2[3], "Insole", "OK" if insole_ok else "FAULT",
                 "heel/toe switches" if insole_ok else "thigh phase portrait in use", "#90be6d" if insole_ok else "#e76f51")
        log_ph.markdown(log_html(s.controller.events, 9), unsafe_allow_html=True)
        if s.rows:
            df = s.dataframe().tail(600)
            df_twin = twin.dataframe().tail(600) if twin is not None else None
            chart_ph.plotly_chart(live_figure(df, cfg["ctrl"].tolerance_deg, cfg["kind"] == "cognitive", df_twin))
        if ss.get("running") and "tick_wall" in ss:
            t_tick, clip_len = ss.tick_wall
            cost = time.perf_counter() - t_tick
            ss.tick_cost = cost if "tick_cost" not in ss else 0.6 * ss.tick_cost + 0.4 * cost
            time.sleep(max(0.0, clip_len - cost))
        if s.done and ss.get("running"):
            ss.running = False
            ss.result = pack_result(s, cfg)
            st.toast("Session finished - open the Session Analysis tab", icon="✅")
            st.rerun()
        if not ss.get("running"):
            break


# ======================================================================== analysis tab
def curves_figure(log: pd.DataFrame) -> go.Figure:
    pct = np.linspace(0, 100, 101)
    k = np.rad2deg(gait_cycle_curves(log, "knee"))
    ref = np.rad2deg(GaitReference().angle("knee", pct / 100))
    fig = go.Figure()
    fig.add_vrect(x0=TOE_OFF * 100, x1=100, fillcolor="#ffffff", opacity=0.04, line_width=0,
                  annotation_text="swing", annotation_position="top left")
    fig.add_trace(go.Scatter(x=pct, y=ref, name="normative (Winter/Perry)", line=dict(color="#e8eaf0", dash="dash")))
    n = max(1, min(3, len(k) // 3))
    for sl, name, col in ((slice(0, n), f"first {n} strides", "#4ea8de"), (slice(-n, None), f"last {n} strides", "#e76f51")):
        fig.add_trace(go.Scatter(x=pct, y=k[sl].mean(0), name=name, line=dict(color=col, width=3)))
    fig.update_layout(template="plotly_dark", height=410, xaxis_title="gait cycle (%)", yaxis_title="knee flexion (deg)",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.22),
                      title="Knee angle over the gait cycle (heel strike = 0 %)")
    return fig


def stride_trend_figure(strides: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=strides["stride"], y=strides["rmse_deg"], name="stride RMSE (deg)",
                             line=dict(color="#4ea8de", width=2.5)), secondary_y=False)
    b = strides[strides["buckled"]]
    fig.add_trace(go.Scatter(x=b["stride"], y=b["rmse_deg"], mode="markers", name="knee buckled",
                             marker=dict(symbol="x", size=11, color="#e76f51")), secondary_y=False)
    fig.add_trace(go.Bar(x=strides["stride"], y=100 * strides["assist_ratio"], name="exo share (%)",
                         marker_color="rgba(244,162,97,0.45)"), secondary_y=True)
    fig.update_layout(template="plotly_dark", height=410, margin=dict(l=10, r=10, t=40, b=10),
                      title="Learning across strides", legend=dict(orientation="h", y=-0.22), xaxis_title="stride")
    fig.update_yaxes(title_text="RMSE (deg)", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="exo share (%)", secondary_y=True, range=[0, 100], dtick=25, showgrid=False)
    return fig


def gain_heatmap(history: list) -> go.Figure:
    G = np.array([h["gain_profile"] for h in history])
    fig = go.Figure(go.Heatmap(z=G.T, x=np.arange(1, len(G) + 1), y=np.linspace(0, 100, G.shape[1]),
                               colorscale="Inferno", zmin=0, zmax=2, colorbar=dict(title="gain")))
    fig.add_hline(y=TOE_OFF * 100, line_dash="dash", line_color="white", annotation_text="toe off")
    fig.update_layout(template="plotly_dark", height=410, margin=dict(l=10, r=10, t=40, b=10),
                      title="Learned assist-as-needed gain per gait-phase bin", xaxis_title="stride",
                      yaxis_title="gait cycle (%)")
    return fig


def perception_figure(log: pd.DataFrame) -> go.Figure:
    g = log[(log["t"] > log["t"].iloc[-1] - 4.0)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Gait phase: simulator truth vs exo estimate",
                                        "Patient torque: truth vs estimate (N·m)"))
    fig.add_trace(go.Scatter(x=g["t"], y=g["phase"], name="true phase", line=dict(color="#6c757d", width=5)), 1, 1)
    fig.add_trace(go.Scatter(x=g["t"], y=g["phase_hat"], name="estimated phase", line=dict(color="#e8eaf0")), 1, 1)
    fig.add_trace(go.Scatter(x=g["t"], y=g["tau_patient"], name="true torque", line=dict(color="#6c757d", width=5)), 2, 1)
    fig.add_trace(go.Scatter(x=g["t"], y=g["tau_patient_hat"], name="estimated torque", line=dict(color="#90be6d")), 2, 1)
    fig.update_layout(template="plotly_dark", height=430, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.12))
    return fig


def run_headless(cfg, render_video: bool):
    s = make_session(cfg)
    bar = st.progress(0.0, text="Simulating…")
    cam = st.session_state.get("camera", "side") if render_video else None
    while not s.done:
        s.advance(1.0, render_camera=cam)
        bar.progress(min(s.time / s.sim_cfg.duration, 1.0), text=f"Simulating… {s.time:.0f} / {s.sim_cfg.duration:.0f} s")
    video = None
    if render_video and s.frames:
        bar.progress(1.0, text="Encoding MP4…")
        path = Path(tempfile.gettempdir()) / f"cogexo_{int(time.time())}.mp4"
        video = save_mp4(s.frames, path, fps=s.sim_cfg.render_fps).read_bytes()
    bar.empty()
    st.session_state.result = pack_result(s, cfg, video)


def analysis_tab(cfg):
    ss = st.session_state
    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    vid = c2.checkbox("Also render an MP4 video", value=False, help="Adds ~1 s of compute per simulated second")
    if c1.button("⚡ Run headless session (sidebar settings)", type="primary", width="stretch"):
        run_headless(cfg, vid)
    c3.caption("Runs the full closed loop as fast as possible (≈5–8× real time) and analyses it. "
               "Sessions finished in the Live tab also appear here.")
    if "result" not in ss:
        st.info("No session yet - run one here or in the **Live Simulation** tab.")
        return
    R = ss.result
    r = R["result"]
    log, strides, m = r.log, r.strides, r.metrics
    st.subheader(f"Session: {R['label']}")
    if not m:
        st.warning("Session too short for stride statistics.")
        return
    t = st.columns(6)
    tile(t[0], "Knee RMSE", f"{m['Knee RMSE vs normative (deg)']:.1f}°", "vs normative gait")
    tile(t[1], "Buckling strides", f"{m['Strides with knee buckling (%)']:.0f} %", ">12° excess stance flexion")
    tile(t[2], "Peak swing flexion", f"{m['Peak swing flexion (deg)']:.0f}°", f"normative {NORMATIVE_PEAK_SWING_DEG:.0f}°")
    tile(t[3], "Toe scuffs / stride", f"{m['Toe scuffs per stride']:.2f}", "foot drag in swing")
    tile(t[4], "Exo share of torque", f"{m['Assistance ratio (%)']:.0f} %", f"peak {m['Exo peak torque (N·m)']:.0f} N·m")
    tile(t[5], "Patient effort / fatigue", f"{m['Final effort (%)']:.0f} / {m['Final fatigue (%)']:.0f} %", "at session end")
    st.write("")
    a, b = st.columns([1.15, 1])
    with a:
        if R["skeleton"]:
            st.plotly_chart(skeleton_animation(R["skeleton"][-300:], R["tau_max"],
                                               "Interactive 3D replay (drag to rotate)"), key="skel",
                            config={"scrollZoom": False})
    with b:
        if R["video"]:
            st.video(R["video"])
            st.caption("MuJoCo offscreen render (SIMULATED)")
        st.markdown("**Explainable decision log**")
        st.markdown(f'<div style="max-height:330px;overflow-y:auto">{log_html(r.events, 200)}</div>',
                    unsafe_allow_html=True)
    a, b = st.columns(2)
    a.plotly_chart(curves_figure(log), key="curves")
    if len(strides):
        b.plotly_chart(stride_trend_figure(strides), key="trend")
    a, b = st.columns(2)
    if R["history"]:
        a.plotly_chart(gain_heatmap(R["history"]), key="gains")
    else:
        a.info("Learned gains exist only for the cognitive controller.")
    b.plotly_chart(perception_figure(log), key="perc")
    with st.expander("Perception accuracy vs simulator ground truth"):
        st.dataframe(pd.DataFrame({"value": r.perception}).round(2), width="stretch")
    with st.expander("All metrics and per-stride table"):
        st.dataframe(pd.DataFrame({"value": m}).round(2), width="stretch")
        st.dataframe(strides.round(2), width="stretch")
    st.download_button("⬇ Download session log (CSV, SIMULATED)", log.to_csv(index=False).encode(),
                       file_name="cognitive_exo_session_SIMULATED.csv", mime="text/csv")


# ======================================================================== comparison tab
@st.cache_resource
def load_experiments():
    path = RESULTS_DIR / "experiments.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def comparison_tab(cfg):
    data = load_experiments()
    st.markdown("Same virtual patient, same simulator, three controllers. Pre-computed by "
                "`python scripts/run_experiments.py` (16 sessions, ≈1 min) - or run your own below.")
    custom = st.session_state.get("custom_cmp")
    options = []
    if custom:
        options.append("your sidebar patient (custom run)")
    if data:
        options += [f"{p} patient (40 s)" for p in data["patients"]] + ["fatiguing patient (5× fatigue, 60 s)"]
    if not options:
        st.warning("No results yet: run `python scripts/run_experiments.py` or the custom comparison below.")
    else:
        choice = st.radio("Scenario", options, horizontal=True)
        if choice.startswith("your"):
            runs = custom
        elif choice.startswith("fatiguing"):
            runs = {k: data["runs"][("fatigue", "moderate", k)] for k in data["kinds"]}
        else:
            p = choice.split(" patient")[0]
            runs = {k: data["runs"][("main", p, k)] for k in data["kinds"]}
        comparison_view(runs)
    st.markdown("---")
    if st.button(f"⚖ Run custom comparison: {cfg['preset']} patient, all three controllers, 30 s each"):
        out, bar = {}, st.progress(0.0)
        for i, k in enumerate(KIND_LABEL):
            r = make_session(with_kind(cfg, k), skeleton_hz=0, duration=30.0).run()
            out[k] = {"metrics": r.metrics, "strides": r.strides, "events": r.events,
                      "curves": {c: (v.mean(0), v.std(0)) for c, v in
                                 ((c, gait_cycle_curves(r.log, c)) for c in ("knee", "knee_ref"))}}
            bar.progress((i + 1) / 3, text=f"{KIND_LABEL[k]} done")
        st.session_state.custom_cmp = out
        st.rerun()


def comparison_view(runs: dict):
    keys = [("Knee RMSE vs normative (deg)", "Knee RMSE (deg)"), ("Strides with knee buckling (%)", "Buckling strides (%)"),
            ("Toe scuffs per stride", "Toe scuffs / stride"), ("Assistance ratio (%)", "Exo share of torque (%)"),
            ("Final effort (%)", "Patient effort at end (%)"), ("Exo torque RMS (N·m)", "Exo torque RMS (N·m)")]
    fmt = {"Toe scuffs per stride": "{:.2f}"}
    fig = make_subplots(rows=2, cols=3, subplot_titles=[k[1] for k in keys], vertical_spacing=0.2,
                        horizontal_spacing=0.07)
    for j, (key, _) in enumerate(keys):
        for k, run in runs.items():
            v = run["metrics"].get(key, np.nan)
            fig.add_trace(go.Bar(x=[KIND_LABEL[k].split()[0]], y=[v], marker_color=KIND_COLOR[k], name=KIND_LABEL[k],
                                 showlegend=(j == 0), text=[fmt.get(key, "{:.1f}").format(v)], textposition="outside",
                                 cliponaxis=False), j // 3 + 1, j % 3 + 1)
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(template="plotly_dark", height=560, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.08))
    st.plotly_chart(fig, key="cmp_bars")
    a, b = st.columns([1.2, 1])
    pct = np.linspace(0, 100, 101)
    f2 = go.Figure()
    ref = next(iter(runs.values()))["curves"]["knee_ref"][0]
    f2.add_trace(go.Scatter(x=pct, y=np.rad2deg(ref), name="normative", line=dict(color="#e8eaf0", dash="dash")))
    for k, run in runs.items():
        mu, sd = run["curves"]["knee"]
        f2.add_trace(go.Scatter(x=pct, y=np.rad2deg(mu + sd), line=dict(width=0), showlegend=False, hoverinfo="skip"))
        f2.add_trace(go.Scatter(x=pct, y=np.rad2deg(mu - sd), fill="tonexty", fillcolor=rgba(KIND_COLOR[k], 0.15),
                                line=dict(width=0), showlegend=False, hoverinfo="skip"))
        f2.add_trace(go.Scatter(x=pct, y=np.rad2deg(mu), name=KIND_LABEL[k], line=dict(color=KIND_COLOR[k], width=3)))
    f2.add_vrect(x0=TOE_OFF * 100, x1=100, fillcolor="#ffffff", opacity=0.04, line_width=0)
    f2.update_layout(template="plotly_dark", height=400, title="Knee angle over the gait cycle (mean ± SD)",
                     xaxis_title="gait cycle (%)", yaxis_title="knee flexion (deg)", margin=dict(l=10, r=10, t=40, b=10),
                     legend=dict(orientation="h", y=-0.2))
    a.plotly_chart(f2, key="cmp_curves")
    with b:
        st.markdown("**Cognitive controller - decision log**")
        ev = runs.get("cognitive", {}).get("events", [])
        st.markdown(f'<div style="max-height:360px;overflow-y:auto">{log_html(ev, 100)}</div>', unsafe_allow_html=True)
    with st.expander("Full metrics table"):
        st.dataframe(pd.DataFrame({KIND_LABEL[k]: run["metrics"] for k, run in runs.items()}).round(2), width="stretch")


# ======================================================================== how-it-works tab
ARCH = """
digraph G {
  rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, color="#3a4152",
        fontcolor="#e8eaf0", fillcolor="#1a1d27", margin="0.18,0.08"];
  edge [color="#8d99ae", fontcolor="#9aa3b5", fontsize=11, penwidth=1.4];
  subgraph cluster_world { label="SIMULATED WORLD  (MuJoCo, 500 Hz control)"; fontcolor="#f28482"; fontsize=13;
    color="#b91c1c"; style="dashed,rounded";
    pat [label="Virtual patient\\nweakness · spasticity · fatigue\\nslacking · delayed reflex"];
    sim [label="Treadmill + body-weight support\\nnormative hips / ankles / left leg\\nphysics-only RIGHT KNEE"]; }
  subgraph cluster_exo { label="COGNITIVE EXOSKELETON  (sees its sensors only)"; fontcolor="#f4a261"; fontsize=13;
    color="#f4a261"; style="dashed,rounded";
    sens [label="Wearable sensors: encoder · thigh IMU · gyros · insoles · motor torque\\n(noise, latency, bounce, faults)"];
    perc [label="PERCEIVE\\nheel strike · toe off · gait phase\\nphase-portrait fallback · toe drag\\npatient torque (inverse dynamics)", fillcolor="#1f2a3a"];
    reas [label="REASON\\nexplainable supervisor\\nASSIST · CHALLENGE · FATIGUE · HOLD\\nslacking-vs-fatigue test", fillcolor="#2a2238"];
    act  [label="ACT\\nimpedance in a ±tol tunnel\\n+ learned feedforward\\nanti-buckling reflex · safety layer", fillcolor="#33261f"];
    learn[label="LEARN (every stride)\\nassist-as-needed gains per phase bin · ILC feedforward\\ntoe-clearance goal · challenge patience", fillcolor="#1f3329"];
    {rank=same; perc; reas; act;} }
  pat -> sim [label=" muscle torque"]; sim -> pat [label=" knee state"];
  sim -> sens [label=" true state"]; sens -> perc; perc -> reas [label=" belief"]; reas -> act [label=" mode"];
  perc -> learn [label=" errors"]; reas -> learn; learn -> act [label=" g[b], u_ff[b]"];
  act -> sim [label=" exo torque", constraint=false];
}
"""


def how_tab():
    st.graphviz_chart(ARCH, width="stretch")
    a, b = st.columns(2)
    with a:
        st.markdown("""
### The simulated world
- **MuJoCo 3** physics at 1 kHz; control at 500 Hz. A 70 kg / 1.75 m body built from **Winter's anthropometric tables**.
- Clinical **treadmill + body-weight-support (Lokomat-style) setup**: hips, ankles and the healthy left leg follow
  **normative gait** (Winter/Perry, Fourier-smoothed) through computed-torque servos. The **right knee is driven only by
  physics**: patient muscle torque + exo motor torque + body-weight load.
- **Virtual patient**: feedforward + delayed (25 ms) stretch reflex; weakness, phase-gated spasticity (stiff-knee gait),
  3-compartment fatigue (Xia & Frey Law 2008), slacking (Reinkensmeyer), motor noise.

### Perception (no ground truth)
- Sensor model: 14-bit encoder, thigh IMU (bias, noise), gyros, insoles (15 ms latency, bounce, false pulses), faults.
- Heel strike = debounced heel switch, time-stamped back by the known delay; adaptive stride period.
- Gait phase = time since heel strike / stride period; **fallback: self-calibrating thigh phase portrait**
  (Quintero et al. 2018) if the insoles fail.
- Patient torque = `mj_inverse` on the measured motion − exo torque − modelled body-weight load.
""")
    with b:
        st.markdown("### The cognitive controller")
        st.latex(r"\tau_{exo} = \mathrm{sat}\Big( g[b]\,s_{mode}\big[K\,\mathrm{dead}(\theta^{*}-\theta,\,tol[b]) "
                 r"+ B\,\dot e\big] + s_{mode}\,u_{ff}[b] \Big),\quad b=\lfloor \hat\varphi N \rfloor")
        st.latex(r"g[b] \leftarrow 0.9\,g[b] + \kappa\,|\mathrm{dead}(\bar e[b])|")
        st.latex(r"u_{ff}[b] \leftarrow 0.9\,u_{ff}[b] + \gamma\,\mathrm{dead}(\bar e[b+1])")
        st.markdown("""
- **Assist-as-needed tunnel** (±3°): no corrective torque while the patient is close enough → patient keeps working.
- **AAN gains** (Emken et al. 2007) grow where the patient fails and are *forgotten* where they cope.
- **Iterative learning** feedforward anticipates the torque the patient lacks, stride after stride.
- **Goal adaptation**: a perceived toe drag raises the swing knee-flexion target (functional goal: clear the toe).
- **Explainable supervisor** (logged with reasons): SAFETY_HOLD after buckling; CHALLENGE when performance is good or
  slacking is suspected - and if the challenge fails, it concludes **fatigue, not slacking** → FATIGUE_SUPPORT + rest advice.
  Failed challenges double its patience.
- **Safety layer** (all controllers): ±60 N·m, slew-rate limit, hyperextension guard, overspeed cut-off.

### Honest limitations
Sagittal-plane model, kinematically driven hips/ankles, patient models from literature (not fitted to a person),
insole switches emulated from gait events. **All results are simulated** and are not clinical evidence.
""")
    for f, cap in (("experiments_summary.png", "Controller comparison (SIMULATED)"),
                   ("perception.png", "Perception accuracy vs ground truth (SIMULATED)")):
        if (RESULTS_DIR / f).exists():
            st.image(str(RESULTS_DIR / f), caption=cap, width="stretch")


# ======================================================================== hardware tab
HW_FLOW = """
digraph H {
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.5;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, color="#3a4152",
        fontcolor="#e8eaf0", fillcolor="#1a1d27", margin="0.14,0.07"];
  edge [color="#8d99ae", fontcolor="#9aa3b5", fontsize=10, penwidth=1.3];
  enc [label="④ Knee encoder\nknee angle, speed"]; imu [label="⑤ Thigh IMU\nthigh angle, swing speed"];
  ins [label="⑦ Insole switches\nheel strike, toe off"]; cur [label="Motor current\n(torque feedback)"];
  mcu [label="⑥ Microcontroller (500 Hz)\nPERCEIVE: gait phase, patient torque\nREASON: assist / challenge / fatigue\nACT: impedance torque command\nLEARN: gains per stride", fillcolor="#2a2238"];
  saf [label="⑧ Safety layer\n±60 N·m, slew limit,\nhard stops, E-stop", fillcolor="#33261f"];
  drv [label="Motor driver"]; mot [label="③ Knee motor + gearbox", fillcolor="#1f2a3a"]; leg [label="Patient's knee\n(extends / flexes)", fillcolor="#1f3329"];
  enc -> mcu; imu -> mcu; ins -> mcu; cur -> mcu; mcu -> saf [label=" torque command"]; saf -> drv -> mot -> leg;
  leg -> enc [style=dashed, label=" motion", constraint=false];
}
"""

STRIDE_STEPS = [
    ("Heel strike → loading response (0–12 %)", "Insole ⑦ senses the heel hit the ground. A weak quadriceps cannot hold the knee while body weight "
     "lands on it, so the knee buckles. Motor ③ pushes the knee into extension within a few ms of the encoder ④ seeing it fold "
     "(anti-buckling reflex)."),
    ("Mid and terminal stance (12–50 %)", "The knee only needs to stay near normal. While the encoder shows the angle inside the ±3° tunnel, the motor "
     "stays quiet so the patient does the work. If the patient drifts out, assistance grows."),
    ("Pre-swing → initial swing (50–75 %)", "Toe off is read by insole ⑦. A stiff or weak knee does not flex enough and the toe drags. "
     "The motor adds a flexion pulse (learned feedforward from earlier strides) so the foot clears the ground."),
    ("Mid and terminal swing (75–100 %)", "The shank swings forward and the knee straightens for the next heel strike. The thigh IMU ⑤ keeps "
     "the phase estimate on track; the motor mostly stays out of the way."),
]


@st.cache_data(show_spinner="Simulating a short walk for the hardware view…")
def hardware_demo_frames() -> list:
    s = Session(patient_preset("moderate"), make_controller("cognitive"), SimConfig(duration=9.0), GaitConfig(),
                skeleton_hz=20.0)
    s.run()
    return s.skeleton_frames[-140:]            # last 7 s: past the start-up transient


def hardware_tab():
    ss = st.session_state
    st.markdown("### How the exoskeleton is built and worn")
    st.caption("A proposed hardware design for the simulated device. Component values (60 N·m, 500 Hz, 14-bit encoder) "
               "match the simulation; nothing here has been built or tested on a person.")
    R = ss.get("result")
    if R and R["skeleton"] and R["kind"] != "none":      # an unpowered session would show a motor stuck at 0 N·m
        frames, src = R["skeleton"][-140:], f"Replaying the end of your last session ({R['label']})."
        tau_max = R["tau_max"]
    else:
        frames, src, tau_max = hardware_demo_frames(), "Replaying a 7 s demo walk (moderate patient, cognitive controller).", 60.0
    a, b = st.columns([1.5, 1])
    with a:
        st.plotly_chart(hardware_animation(frames, tau_max, "Right leg with the exoskeleton"), key="hw_anim",
                        config={"displayModeBar": False})
        st.caption(f"{src} Press ▶ Play. The knee motor ③ turns from blue to orange as it works harder; the insole "
                   "squares ⑦ turn green while that part of the foot presses on the ground.")
    with b:
        for n, name, where, what in HW_COMPONENTS:
            st.markdown(f'<div class="tile" style="min-height:0"><div class="val long"><span style="color:#f4a261">{n}</span> '
                        f'&nbsp;{name}</div><div class="sub"><i>{where}</i><br>{what}</div></div>', unsafe_allow_html=True)
    st.markdown("#### Signal path: from sensors to the knee")
    st.graphviz_chart(HW_FLOW, width="stretch")
    st.markdown("#### What the hardware does during one stride")
    for title, text in STRIDE_STEPS:
        st.markdown(f"**{title}**<br>{text}", unsafe_allow_html=True)
    st.markdown("#### Why it works")
    st.markdown("The patient's own muscles are too weak or too stiff to hold the knee in stance and bend it in swing. "
                "Sensors ④⑤⑦ tell the controller where the patient is in the stride, and motor ③ supplies only the missing "
                "torque at the right moment. That is the difference you see in the Live Simulation tab between the unassisted "
                "patient (buckling, dragging the toe) and the assisted one.")


# ======================================================================== main
def main():
    cfg = sidebar()
    st.markdown("## 🦿 Cognitive Exoskeleton for Knee Rehabilitation")
    tabs = st.tabs(["🎥 Live Simulation", "📊 Session Analysis", "⚖ Controller Comparison", "🔧 Hardware", "🧠 How it works"])
    with tabs[0]:
        live_tab(cfg)
    with tabs[1]:
        analysis_tab(cfg)
    with tabs[2]:
        comparison_tab(cfg)
    with tabs[3]:
        hardware_tab()
    with tabs[4]:
        how_tab()


main()

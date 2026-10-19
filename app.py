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
from exo.metrics import NORMATIVE_PEAK_SWING_DEG, gait_cycle_curves
from exo.perception import PHASE_SOURCES
from exo.session import Session
from exo.viz import save_mp4, skeleton_animation

st.set_page_config(page_title="Cognitive Knee Exoskeleton", page_icon="🦿", layout="wide")

KIND_LABEL = {"none": "No assistance", "fixed": "Fixed impedance", "cognitive": "Cognitive adaptive"}
KIND_COLOR = {"none": "#9e9e9e", "fixed": "#4ea8de", "cognitive": "#e76f51"}
SUB_NAMES = [s[0] for s in SUBPHASES]

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
.sim-banner {background: linear-gradient(90deg,#7f1d1d,#b91c1c); color: #fff; padding: 6px 14px;
             border-radius: 8px; font-weight: 600; letter-spacing: .3px; margin-bottom: .6rem;}
.tile {background:#1a1d27; border:1px solid #2a2f3d; border-radius:12px; padding:10px 14px; height:100%;}
.tile .lbl {font-size:.74rem; color:#9aa3b5; text-transform:uppercase; letter-spacing:.6px;}
.tile .val {font-size:1.45rem; font-weight:700; color:#f1f3f8; line-height:1.25;}
.tile .sub {font-size:.75rem; color:#9aa3b5;}
.logline {font-family: ui-monospace, Consolas, monospace; font-size:.80rem; padding:3px 0;
          border-bottom:1px solid #222736;}
.logline b {color:#f4a261;}
</style>""", unsafe_allow_html=True)


def banner(text="SIMULATED DATA — virtual patient in a MuJoCo physics simulation; not clinical measurements"):
    st.markdown(f'<div class="sim-banner">⚠️ {text}</div>', unsafe_allow_html=True)


def tile(col, label, value, sub="", color=None):
    style = f' style="color:{color}"' if color else ""
    col.markdown(f'<div class="tile"><div class="lbl">{label}</div><div class="val"{style}>{value}</div>'
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
def live_figure(df: pd.DataFrame, tol_deg: float, show_tunnel: bool) -> go.Figure:
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
    fig.add_trace(go.Scatter(x=t, y=np.rad2deg(df["knee"]), name="knee (simulated)",
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
    if c1.button("▶ Start new session", type="primary", width="stretch"):
        ss.live = make_session(cfg)
        ss.live_cfg = cfg
        ss.running = True
    if c2.button("⏸ Pause / ▶ resume", width="stretch", disabled="live" not in ss):
        ss.running = not ss.get("running", False)
    ss.camera = c3.selectbox("Camera", ["side", "knee", "iso", "front"], label_visibility="collapsed")
    ss.speed = c4.select_slider("Speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0,
                                format_func=lambda v: f"{v:g}× real time", label_visibility="collapsed")
    c5.caption("Start a session, then intervene live: inject fatigue, unplug the insole, or hot-swap the controller.")
    if "live" not in ss:
        st.info("Configure the patient and controller in the sidebar, then press **Start new session**. "
                "The MuJoCo world, the virtual patient, the sensors and the exoskeleton controller then run live.")
        img = RESULTS_DIR / "phase1_side.png"
        if img.exists():
            st.image(str(img), caption="The simulated world: treadmill, body-weight-support harness, "
                                       "right-knee exoskeleton (SIMULATED)", width="stretch")
        return
    st.fragment(live_panel)()


def live_panel():
    ss = st.session_state
    s: Session = ss.live
    cfg = ss.live_cfg
    left, right = st.columns([1.05, 1])
    with left:
        img_ph = st.empty()
        b1, b2, b3 = st.columns([1, 1, 1.4])
        if b1.button("💤 Inject fatigue (+30%)", width="stretch"):
            s.inject_fatigue(0.3)
            s.controller.log(s.time, "USER: fatigue injected (+30 %) by the therapist")
        if b2.button("🔌 Unplug insole", width="stretch"):
            s.fail_insole()
            s.controller.log(s.time, "USER: insole disconnected")
        swap = b3.selectbox("Hot-swap controller", list(KIND_LABEL), format_func=KIND_LABEL.get,
                            index=list(KIND_LABEL).index(cfg["kind"]), key=f"swap_{id(s)}")
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
    # Clicking any widget interrupts the loop and the fragment restarts it with the new state.
    while True:
        # ---- simulate one tick, then send ONE animated clip + one UI update (browser-friendly)
        if ss.get("running") and not s.done:
            t_tick = time.perf_counter()
            frames = []
            for _ in range(LIVE_FRAMES):
                if s.done:
                    break
                s.advance(1 / LIVE_FPS)
                frames.append(s.sim.render(ss.camera, 720, 405))
            img_ph.markdown(clip_html(frames, LIVE_FPS / ss.speed), unsafe_allow_html=True)
            ss.last_frame = frames[-1] if frames else ss.get("last_frame")
            ss.tick_wall = (t_tick, len(frames) / LIVE_FPS / ss.speed)
        else:
            img_ph.markdown(clip_html([s.sim.render(ss.camera, 720, 405)], 1), unsafe_allow_html=True)

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
            chart_ph.plotly_chart(live_figure(df, cfg["ctrl"].tolerance_deg, cfg["kind"] == "cognitive"))
        if ss.get("running") and "tick_wall" in ss:
            t_tick, clip_len = ss.tick_wall
            time.sleep(max(0.0, clip_len - (time.perf_counter() - t_tick)))
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
                                        "Patient muscle torque: truth vs inverse-dynamics estimate (N·m)"))
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
                                               "Interactive 3D replay (drag to rotate)"), key="skel")
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
    fig = make_subplots(rows=1, cols=len(keys), subplot_titles=[k[1] for k in keys])
    for j, (key, _) in enumerate(keys):
        for k, run in runs.items():
            v = run["metrics"].get(key, np.nan)
            fig.add_trace(go.Bar(x=[KIND_LABEL[k].split()[0]], y=[v], marker_color=KIND_COLOR[k], name=KIND_LABEL[k],
                                 showlegend=(j == 0), text=[f"{v:.1f}"], textposition="outside",
                                 cliponaxis=False), 1, j + 1)
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=-0.15))
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


# ======================================================================== main
def main():
    cfg = sidebar()
    st.markdown("## 🦿 Cognitive Exoskeleton for Knee Rehabilitation")
    banner()
    tabs = st.tabs(["🎥 Live Simulation", "📊 Session Analysis", "⚖ Controller Comparison", "🧠 How it works"])
    with tabs[0]:
        live_tab(cfg)
    with tabs[1]:
        analysis_tab(cfg)
    with tabs[2]:
        comparison_tab(cfg)
    with tabs[3]:
        how_tab()


main()

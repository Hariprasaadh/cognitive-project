"""Visualisation helpers: interactive Plotly 3D skeleton animation and MP4 export."""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .controllers import MODE_COLORS

BELT_TOP = 0.099
LEG = ["hip", "knee", "ankle", "heel", "toe", "ankle"]
EXO_OFFSET = np.array([0.0, -0.075, 0.0])      # exo bars sit on the lateral (camera) side of the right leg


def _leg(pts: dict, side: str) -> np.ndarray:
    return np.array([pts["pelvis"]] + [pts[f"{side}_{j}"] for j in LEG])


def _torso(pts: dict) -> np.ndarray:
    mid = (pts["l_shoulder"] + pts["r_shoulder"]) / 2
    head = mid + np.array([0.0, 0.0, 0.22])
    return np.array([pts["pelvis"], mid, pts["l_shoulder"], mid, pts["r_shoulder"], mid, head])


def _frame_traces(f: dict, tau_max: float) -> list:
    p = {k: np.asarray(v) for k, v in f["pts"].items()}
    L, R, T = _leg(p, "l"), _leg(p, "r"), _torso(p)
    exo = np.array([p["r_hip"] * 0.4 + p["r_knee"] * 0.6, p["r_knee"], p["r_knee"] * 0.6 + p["r_ankle"] * 0.4]) + EXO_OFFSET
    load = min(abs(f["tau_exo"]) / tau_max, 1.0)
    motor_col = f"rgb({int(40 + 215 * load)},{int(140 - 60 * load)},{int(240 - 220 * load)})"
    return [
        go.Scatter3d(x=L[:, 0], y=L[:, 1], z=L[:, 2], mode="lines+markers", line=dict(color="#8d99ae", width=9),
                     marker=dict(size=3, color="#8d99ae"), name="left leg (healthy)"),
        go.Scatter3d(x=R[:, 0], y=R[:, 1], z=R[:, 2], mode="lines+markers", line=dict(color="#edf2f4", width=10),
                     marker=dict(size=3, color="#edf2f4"), name="right leg (impaired)"),
        go.Scatter3d(x=T[:, 0], y=T[:, 1], z=T[:, 2], mode="lines+markers", line=dict(color="#8d99ae", width=11),
                     marker=dict(size=[3, 3, 3, 3, 3, 3, 14], color="#8d99ae"), name="torso"),
        go.Scatter3d(x=exo[:, 0], y=exo[:, 1], z=exo[:, 2], mode="lines", line=dict(color="#f4a261", width=14),
                     name="knee exoskeleton"),
        go.Scatter3d(x=[exo[1, 0]], y=[exo[1, 1]], z=[exo[1, 2]], mode="markers",
                     marker=dict(size=13, color=motor_col, line=dict(color="white", width=2)),
                     name="exo motor (colour = torque)"),
    ]


def _belt() -> go.Mesh3d:
    x0, x1, y0, y1 = -0.95, 0.95, -0.29, 0.29
    return go.Mesh3d(x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[BELT_TOP] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
                     color="#2b2d42", opacity=0.9, name="treadmill belt", hoverinfo="skip", showlegend=False)


def skeleton_animation(frames: list, tau_max: float = 60.0, title: str = "") -> go.Figure:
    """frames: [{"t", "pts": {joint: xyz}, "tau_exo", "mode", "knee_deg"}] -> rotatable animated 3D figure."""
    fig = go.Figure(data=_frame_traces(frames[0], tau_max) + [_belt()])
    fig.frames = [go.Frame(data=_frame_traces(f, tau_max), name=f"{f['t']:.2f}", traces=list(range(5)),
                           layout=go.Layout(title_text=f"{title}  t = {f['t']:4.1f} s | knee {f['knee_deg']:4.0f}° | "
                                                       f"exo {f['tau_exo']:+4.0f} N·m | {f['mode']}"))
                  for f in frames]
    dur = 1000 * (frames[1]["t"] - frames[0]["t"]) if len(frames) > 1 else 40
    stride = max(1, len(frames) // 60)
    steps = [dict(method="animate", label=f"{f['t']:.0f}" if i % 10 == 0 else "",
                  args=[[f"{f['t']:.2f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                 transition=dict(duration=0))])
             for i, f in enumerate(frames[::stride])]
    fig.update_layout(
        title=title, template="plotly_dark", height=620, margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="v", y=0.98, x=0.98, xanchor="right", yanchor="top", font=dict(size=10),
                    bgcolor="rgba(0,0,0,0.3)"),
        scene=dict(xaxis=dict(range=[-0.9, 0.9], title="x (m, walking direction)"),
                   yaxis=dict(range=[-0.6, 0.6], title="y (m)"), zaxis=dict(range=[0, 1.9], title="z (m)"),
                   aspectmode="manual", aspectratio=dict(x=1, y=0.67, z=1.05),
                   camera=dict(eye=dict(x=0.9, y=-1.25, z=0.35), center=dict(x=0, y=0, z=-0.05))),
        updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0.02, y=0.98, xanchor="left",
                          yanchor="top", pad=dict(r=6, t=0),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=dur, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(steps=steps, x=0.15, len=0.83, y=0.0, currentvalue=dict(prefix="t = ", suffix=" s"),
                      font=dict(size=9))],
    )
    return fig


# ---------------------------------------------------------------- hardware view (side-on, 2D)
HW_COMPONENTS = [
    # (number, name, where, what it does)
    (1, "Thigh cuff + upright", "mid-thigh, lateral",
     "Carbon-fibre upright held to the thigh by two padded Velcro cuffs. Carries the motor reaction torque into the thigh."),
    (2, "Shank cuff + upright", "shank, lateral",
     "Second upright and cuffs on the shank, with a strap at the ankle. The motor torque acts between the two uprights."),
    (3, "Knee actuator", "knee joint axis",
     "Brushless DC motor + low-ratio gearbox (quasi-direct drive, back-drivable). Pushes the shank to extend (anti-buckling) "
     "or flex (toe clearance). Capped at 60 N·m by the safety layer."),
    (4, "Joint encoder", "knee axis",
     "14-bit absolute magnetic encoder: knee angle and velocity. This is the controller's feedback signal."),
    (5, "Thigh IMU", "front of the thigh",
     "Accelerometer + gyro: thigh angle and swing speed. Gives the gait phase when the insole is unplugged "
     "(phase-portrait fallback)."),
    (6, "Electronics + battery", "back of the thigh upright",
     "Microcontroller running the 500 Hz Perceive → Reason → Act → Learn loop, motor driver with current sensing "
     "(torque ≈ k·current), and the battery pack."),
    (7, "Insole pressure sensors", "heel and forefoot, inside the shoe",
     "Heel and toe switches give heel strike and toe off, so the exo knows the gait phase. Green = foot pressing the ground."),
    (8, "Safety hardware", "knee joint and belt",
     "Mechanical hard stops at hyperextension and full flexion, motor-power cutoff, and a therapist E-stop."),
]
_METAL, _CUFF, _SKIN = "#cfd6e4", "#0e1016", "rgba(201,204,214,0.30)"


def _xz(p) -> np.ndarray:
    return np.array([p[0], p[2]], dtype=float)


def _poly(a, b, half_w: float, t0: float = 0.0, t1: float = 1.0, off: float = 0.0):
    """Rectangle along segment a->b (from fraction t0 to t1), half-width half_w, shifted sideways by `off`.
    Positive off = forward of a downward-pointing limb. Returns (x, z) lists closed with a None gap."""
    a, b = np.asarray(a), np.asarray(b)
    d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
    n = np.array([-d[1], d[0]])
    p0, p1 = a + (b - a) * t0 + n * off, a + (b - a) * t1 + n * off
    c = [p0 - n * half_w, p1 - n * half_w, p1 + n * half_w, p0 + n * half_w, p0 - n * half_w]
    return [q[0] for q in c] + [None], [q[1] for q in c] + [None]


def _at(a, b, t: float, off: float = 0.0) -> np.ndarray:
    a, b = np.asarray(a), np.asarray(b)
    d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
    return a + (b - a) * t + np.array([-d[1], d[0]]) * off


def _join(*polys):
    return [v for p in polys for v in p[0]], [v for p in polys for v in p[1]]


def _hardware_traces(f: dict, tau_max: float) -> list:
    p = {k: _xz(v) for k, v in f["pts"].items()}
    hip, knee, ank, heel, toe, pel = p["r_hip"], p["r_knee"], p["r_ankle"], p["r_heel"], p["r_toe"], p["pelvis"]
    sh = (p["l_shoulder"] + p["r_shoulder"]) / 2
    load = min(abs(f["tau_exo"]) / tau_max, 1.0)
    motor_col = f"rgb({int(40 + 215 * load)},{int(140 - 60 * load)},{int(240 - 220 * load)})"
    heel_on, toe_on = f["pts"]["r_heel"][2] < BELT_TOP + 0.012, f["pts"]["r_toe"][2] < BELT_TOP + 0.012

    def line(xs, zs, **kw):
        return go.Scatter(x=xs, y=zs, hoverinfo="skip", showlegend=False, **kw)

    thigh_x, thigh_z = _poly(hip, knee, 0.050)
    shank_x, shank_z = _poly(knee, ank, 0.042)
    foot_x, foot_z = [heel[0], toe[0], ank[0], heel[0], None], [heel[1], toe[1], ank[1], heel[1], None]
    leg_x, leg_z = _join((thigh_x, thigh_z), (shank_x, shank_z), (foot_x, foot_z))
    up_t = _poly(hip, knee, 0.012, 0.15, 0.97, off=-0.062)
    up_s = _poly(knee, ank, 0.012, 0.03, 0.88, off=-0.056)
    cuff_t = _join(_poly(hip, knee, 0.012 + 0.058, 0.20, 0.30, off=0), _poly(hip, knee, 0.058, 0.62, 0.72, off=0))
    cuff_s = _join(_poly(knee, ank, 0.052, 0.12, 0.22, off=0), _poly(knee, ank, 0.046, 0.70, 0.80, off=0))
    box = _join(_poly(hip, knee, 0.026, 0.30, 0.58, off=-0.100), _poly(hip, knee, 0.016, 0.22, 0.30, off=0.064))
    lk = [p["pelvis"], p["l_hip"], p["l_knee"], p["l_ankle"], p["l_heel"], p["l_toe"], p["l_ankle"]]
    badge = np.array([_at(hip, knee, 0.12, -0.17), _at(knee, ank, 0.62, -0.15), _at(knee, ank, 0.0, 0.12) + [0.04, 0.0],
                      _at(knee, ank, 0.0, -0.12) + [-0.04, 0.0], _at(hip, knee, 0.22, 0.13), _at(hip, knee, 0.44, -0.20),
                      (heel + toe) / 2 + [0.02, 0.15], _at(knee, ank, 0.22, 0.13)])
    return [
        line([q[0] for q in lk], [q[1] for q in lk], mode="lines", line=dict(color="#4a5160", width=7)),
        line([pel[0], sh[0], sh[0]], [pel[1], sh[1], sh[1] + 0.22], mode="lines+markers",
             line=dict(color="#4a5160", width=9), marker=dict(size=[0, 0, 26], color="#4a5160")),
        line(leg_x, leg_z, mode="lines", fill="toself", fillcolor=_SKIN, line=dict(color="#8d99ae", width=1.5)),
        line(*up_t, mode="lines", fill="toself", fillcolor=_METAL, line=dict(color=_METAL, width=1)),
        line(*up_s, mode="lines", fill="toself", fillcolor=_METAL, line=dict(color=_METAL, width=1)),
        line(*cuff_t, mode="lines", fill="toself", fillcolor=_CUFF, line=dict(color="#6b7385", width=2)),
        line(*cuff_s, mode="lines", fill="toself", fillcolor=_CUFF, line=dict(color="#6b7385", width=2)),
        line(*box, mode="lines", fill="toself", fillcolor="#2a3a55", line=dict(color="#4ea8de", width=2)),
        go.Scatter(x=[knee[0]], y=[knee[1]], mode="markers", showlegend=False, name="knee actuator",
                   hovertemplate=f"exo torque {f['tau_exo']:+.0f} N·m<extra>knee actuator</extra>",
                   marker=dict(size=42, color=motor_col, line=dict(color="white", width=3))),
        line([knee[0] - 0.0], [knee[1]], mode="markers",
             marker=dict(size=17, color="#0e1016", line=dict(color="#f4a261", width=3))),
        go.Scatter(x=[heel[0], toe[0]], y=[heel[1] - 0.012, toe[1] - 0.012], mode="markers", showlegend=False,
                   hovertemplate="%{customdata}<extra>insole sensor</extra>", customdata=["heel", "forefoot"],
                   marker=dict(size=13, symbol="square", line=dict(color="white", width=1.5),
                               color=["#90be6d" if heel_on else "#3b4152", "#90be6d" if toe_on else "#3b4152"])),
        go.Scatter(x=[knee[0] + 0.17], y=[knee[1] + 0.02], mode="text", hoverinfo="skip", showlegend=False,
                   text=[f"{f['tau_exo']:+.0f} N·m"], textfont=dict(color="#f4a261", size=15), textposition="middle right"),
        go.Scatter(x=badge[:, 0], y=badge[:, 1], mode="markers+text", hoverinfo="skip", showlegend=False,
                   text=[str(i) for i in range(1, 9)], textfont=dict(color="#0e1016", size=12, family="Arial Black"),
                   marker=dict(size=22, color="#f4a261", line=dict(color="white", width=1.5))),
    ]


def hardware_animation(frames: list, tau_max: float = 60.0, title: str = "") -> go.Figure:
    """Side-on animation of the right leg with the exoskeleton hardware drawn on it and numbered (see HW_COMPONENTS).
    frames: same dicts as `skeleton_animation`."""
    n_dyn = len(_hardware_traces(frames[0], tau_max))
    belt = go.Scatter(x=[-0.7, 0.7], y=[BELT_TOP, BELT_TOP], mode="lines", line=dict(color="#59607a", width=6),
                      hoverinfo="skip", showlegend=False)
    fig = go.Figure(data=_hardware_traces(frames[0], tau_max) + [belt])
    fig.frames = [go.Frame(data=_hardware_traces(f, tau_max), name=f"{f['t']:.2f}", traces=list(range(n_dyn)),
                           layout=go.Layout(title_text=f"{title}  t = {f['t']:4.1f} s | knee {f['knee_deg']:4.0f}° | "
                                                       f"motor {f['tau_exo']:+4.0f} N·m | {f['mode'] or 'unpowered'}"))
                  for f in frames]
    dur = 1000 * (frames[1]["t"] - frames[0]["t"]) if len(frames) > 1 else 50
    stride = max(1, len(frames) // 60)
    steps = [dict(method="animate", label=f"{f['t']:.0f}" if i % 10 == 0 else "",
                  args=[[f"{f['t']:.2f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                 transition=dict(duration=0))])
             for i, f in enumerate(frames[::stride])]
    fig.update_layout(
        title=title, template="plotly_dark", height=650, margin=dict(l=0, r=0, t=50, b=0), showlegend=False,
        paper_bgcolor="#0f1117", plot_bgcolor="#141824",
        xaxis=dict(range=[-0.42, 0.58], visible=False, fixedrange=True),
        yaxis=dict(range=[0.02, 1.22], visible=False, scaleanchor="x", fixedrange=True),
        updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0.02, y=0.98, xanchor="left",
                          yanchor="top", pad=dict(r=6, t=0),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=dur, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(steps=steps, x=0.15, len=0.83, y=0.0, currentvalue=dict(prefix="t = ", suffix=" s"),
                      font=dict(size=9))],
    )
    return fig


def mode_color(mode: str) -> str:
    return MODE_COLORS.get(mode, "#9e9e9e")


def save_mp4(frames: list, path, fps: int = 30) -> Path:
    import imageio.v2 as imageio
    path = Path(path)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=2) as w:
        for fr in frames:
            w.append_data(fr)
    return path

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
        legend=dict(orientation="v", y=0.9, x=0.0, font=dict(size=10), bgcolor="rgba(0,0,0,0.3)"),
        scene=dict(xaxis=dict(range=[-0.9, 0.9], title="x (m, walking direction)"),
                   yaxis=dict(range=[-0.6, 0.6], title="y (m)"), zaxis=dict(range=[0, 1.9], title="z (m)"),
                   aspectmode="manual", aspectratio=dict(x=1, y=0.67, z=1.05),
                   camera=dict(eye=dict(x=0.9, y=-1.25, z=0.35), center=dict(x=0, y=0, z=-0.05))),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.0, xanchor="left", yanchor="top",
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

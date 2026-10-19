"""Phase-3 check: how well the exoskeleton perceives gait from noisy wearable sensors.

Run:  python scripts/check_perception.py     (SIMULATED data -> results/perception*)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exo.config import PATIENT_PRESETS, RESULTS_DIR, SensorConfig, SimConfig, patient_preset  # noqa: E402
from exo.controllers import NoExo  # noqa: E402
from exo.session import Session  # noqa: E402

FAULT_AT = 10.0
KEYS = ["Phase error MAE (% cycle)", "Phase error MAE, phase portrait (% cycle)", "Heel strikes detected (%)",
        "False heel-strike detections", "Heel-strike timing error (ms)",
        "Patient torque estimate correlation r", "Patient torque estimate RMSE (N·m)"]


EVENT_DELAY = 4 * 0.002 + SensorConfig().insole_latency   # debounce + datasheet latency


def _break_wraps(x):
    """NaN at 1->0 wraps so phase plots show a clean sawtooth."""
    x = np.asarray(x, dtype=float).copy()
    x[1:][np.diff(x) < -0.5] = np.nan
    return x


def capture(preset: str, duration: float = 6.0) -> pd.DataFrame:
    """Control-rate (500 Hz) trace of raw vs debounced insole and the thigh signals."""
    s = Session(patient_preset(preset), NoExo(), SimConfig(duration=duration))
    rows = []
    while not s.done:
        true_heel = s.observe().heel
        s.advance(s.ctrl_dt)
        m, p = s.measurement, s.percept
        rows.append((s.time, true_heel, m.heel, p.heel, p.heel_strike, m.thigh, p.thigh_vel, s.phase))
    return pd.DataFrame(rows, columns=["t", "heel_true", "heel_raw", "heel_hat", "hs", "thigh", "thigh_vel", "phase"])


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for name in PATIENT_PRESETS:
        r = Session(patient_preset(name), NoExo(), SimConfig(duration=20.0)).run()
        rows.append({"scenario": name, **r.perception})
        if name == "moderate":
            moderate = r
    fault = Session(patient_preset("moderate"), NoExo(), SimConfig(duration=20.0),
                    sensor_cfg=SensorConfig(insole_fault_at=FAULT_AT)).run()
    rows.append({"scenario": f"moderate, insole fault at {FAULT_AT:.0f} s", **fault.perception})
    table = pd.DataFrame(rows).set_index("scenario")
    table.round(3).to_csv(RESULTS_DIR / "perception_metrics.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[KEYS].round(2).to_string())

    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))

    # (a) phase estimate across an insole failure
    a = ax[0, 0]
    g = fault.log[(fault.log.t > 6) & (fault.log.t < 15)]
    a.plot(g.t, _break_wraps(g.phase), color="0.6", lw=3, label="true phase (simulator)")
    a.plot(g.t, _break_wraps(g.phase_hat), color="#264653", lw=1.2, label="estimated phase")
    fb = g.phase_src.to_numpy() == 1
    a.fill_between(g.t, 0, 1, where=fb, color="#e76f51", alpha=0.12, label="thigh phase-portrait fallback")
    a.axvline(FAULT_AT, color="#e76f51", ls="--")
    a.text(FAULT_AT + 0.1, 1.03, "insole disconnected", color="#e76f51", fontsize=9)
    p = fault.perception
    a.set(xlabel="time (s)", ylabel="gait phase", ylim=(0, 1.12),
          title=f"Gait phase: insole MAE {p['Phase error MAE, insole (% cycle)']:.1f} %, "
                f"portrait MAE {p['Phase error MAE, phase portrait (% cycle)']:.1f} % of cycle")
    a.legend(loc="lower left", fontsize=8, ncol=3, frameon=True)

    # (b) raw vs debounced heel switch around one heel strike
    cap = capture("moderate")
    a = ax[0, 1]
    hs_t = cap.t[cap.heel_true.astype(int).diff() == 1].iloc[2]
    w = cap[(cap.t > hs_t - 0.06) & (cap.t < hs_t + 0.08)]
    t_ms = (w.t - hs_t) * 1000
    a.step(t_ms, w.heel_true + 2.4, where="post", color="0.5", lw=2, label="true contact")
    a.step(t_ms, w.heel_raw + 1.2, where="post", color="#e76f51", lw=1.5, label="raw insole (latency + bounce)")
    a.step(t_ms, w.heel_hat.astype(int), where="post", color="#264653", lw=2, label="debounced")
    for t_det in w[w.hs].t:
        delay = 1000 * (t_det - hs_t)
        a.axvline(delay, color="#2a9d8f", ls=":", lw=2)
        a.annotate("", xy=(delay - 1000 * EVENT_DELAY, 0.6), xytext=(delay, 0.6),
                   arrowprops=dict(arrowstyle="->", color="#2a9d8f", lw=1.5))
        a.text(delay + 2, 0.65, f"detected +{delay:.0f} ms,\ntime-stamped back by "
               f"{1000 * EVENT_DELAY:.0f} ms", color="#2a9d8f", fontsize=8)
    mp = moderate.perception
    a.set(xlabel="time from true heel strike (ms)", yticks=[], ylim=(-0.2, 3.6),
          title=f"Heel strikes: {mp['Heel strikes detected (%)']:.0f} % detected, "
                f"{mp['False heel-strike detections']} false, timing error {mp['Heel-strike timing error (ms)']:.1f} ms")
    a.legend(loc="upper left", fontsize=8)

    # (c) thigh phase portrait
    a = ax[1, 0]
    c = cap[cap.t > 1.5]
    th, om = np.rad2deg(c.thigh), np.rad2deg(c.thigh_vel)
    sc = a.scatter(th, om, c=c.phase, cmap="twilight", s=3)
    fig.colorbar(sc, ax=a, label="true gait phase")
    a.set(xlabel="thigh angle (deg, IMU)", ylabel="thigh angular velocity (deg/s, gyro)",
          title="Thigh phase portrait: phase = polar angle (Quintero 2018)")

    # (d) patient torque estimation
    a = ax[1, 1]
    g = moderate.log[(moderate.log.t > 8) & (moderate.log.t < 10.4)]
    a.plot(g.t, g.tau_patient, color="0.55", lw=3, label="true patient muscle torque")
    a.plot(g.t, g.tau_patient_hat, color="#264653", lw=1.3, label="estimated (inverse dynamics)")
    a.plot(g.t, g.tau_ext, color="#e9c46a", lw=1, ls="--", label="body-weight load (true)")
    mp = moderate.perception
    a.set(xlabel="time (s)", ylabel="knee torque (N·m, + flexion)",
          title=f"Patient torque estimate: r = {mp['Patient torque estimate correlation r']:.2f}, "
                f"RMSE {mp['Patient torque estimate RMSE (N·m)']:.1f} N·m")
    a.legend(fontsize=8, loc="lower left")

    fig.suptitle("SIMULATED DATA - exoskeleton perception from noisy wearable sensors (moderate patient)",
                 fontsize=11, color="#b00020")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "perception.png", dpi=140)
    print(f"Saved {RESULTS_DIR / 'perception.png'} and perception_metrics.csv")


if __name__ == "__main__":
    main()

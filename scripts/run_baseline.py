"""Phase-2 baseline: every patient preset walking with the exoskeleton unpowered.

Run:  python scripts/run_baseline.py        (SIMULATED data -> results/baseline_*)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exo.config import PATIENT_PRESETS, RESULTS_DIR, SimConfig, patient_preset  # noqa: E402
from exo.controllers import NoExo  # noqa: E402
from exo.gait import GaitReference, TOE_OFF  # noqa: E402
from exo.metrics import gait_cycle_curves  # noqa: E402
from exo.session import Session  # noqa: E402

COLORS = {"healthy": "#2a9d8f", "mild": "#8ab17d", "moderate": "#e9c46a", "severe": "#e76f51", "stiff-knee": "#7b61ff"}


def main(duration: float = 30.0):
    RESULTS_DIR.mkdir(exist_ok=True)
    rows, curves = [], {}
    for name in PATIENT_PRESETS:
        res = Session(patient_preset(name), NoExo(), SimConfig(duration=duration)).run()
        curves[name] = np.rad2deg(gait_cycle_curves(res.log))
        rows.append({"patient": name, **res.metrics})
        print(f"{name:10s} RMSE {res.metrics['Knee RMSE vs normative (deg)']:5.1f} deg | "
              f"buckling {res.metrics['Strides with knee buckling (%)']:5.1f}% | "
              f"peak swing {res.metrics['Peak swing flexion (deg)']:5.1f} deg")
    table = pd.DataFrame(rows).set_index("patient")
    table.round(2).to_csv(RESULTS_DIR / "baseline_metrics.csv")

    pct = np.linspace(0, 100, 101)
    ref = np.rad2deg(GaitReference().angle("knee", pct / 100))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.axvspan(TOE_OFF * 100, 100, color="0.93", zorder=0)
    ax.text(31, 0.96, "STANCE", ha="center", color="0.45", fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(81, 0.96, "SWING", ha="center", color="0.45", fontsize=9, transform=ax.get_xaxis_transform())
    ax.plot(pct, ref, "k--", lw=2, label="normative")
    for name, c in curves.items():
        m, s = c.mean(0), c.std(0)
        ax.plot(pct, m, color=COLORS[name], lw=2, label=name)
        ax.fill_between(pct, m - s, m + s, color=COLORS[name], alpha=0.15)
    ax.set(xlabel="gait cycle (%)", ylabel="right knee flexion (deg)", xlim=(0, 100),
           title="Knee angle without assistance (mean ± SD over strides)")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 0.92))
    ax = axes[1]
    keys = ["Knee RMSE vs normative (deg)", "Max stance excess flexion (deg)", "Swing flexion deficit (deg)"]
    x = np.arange(len(keys))
    w = 0.8 / len(table)
    for i, (name, r) in enumerate(table.iterrows()):
        ax.bar(x + i * w - 0.4 + w / 2, [r[k] for k in keys], w, color=COLORS[name], label=name)
    ax.set_xticks(x, ["RMSE", "stance excess\n(buckling)", "swing deficit\n(stiff knee)"])
    ax.set(ylabel="degrees", title="Gait deficits by patient profile")
    ax.axhline(0, color="k", lw=0.8)
    fig.suptitle("SIMULATED DATA - virtual patients, exoskeleton unpowered", fontsize=11, color="#b00020")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "baseline_knee_curves.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'baseline_knee_curves.png'} and baseline_metrics.csv")

    # snapshot of a severe patient's knee buckling in mid-stance
    s = Session(patient_preset("severe"), NoExo(), SimConfig(duration=6.0))
    T = s.T
    s.advance(4 * T + 0.25 * T)
    imageio.imwrite(RESULTS_DIR / "baseline_severe_buckling.png", s.sim.render("side", 1280, 720))
    s.sim.close()


if __name__ == "__main__":
    main()

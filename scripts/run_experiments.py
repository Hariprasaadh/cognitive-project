"""Phase-4 experiments: controller x patient comparison + fatigue and sensor-fault scenarios.

Run:  python scripts/run_experiments.py      (SIMULATED data -> results/experiments_* , results/experiments.pkl)
"""
import pickle
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exo.config import RESULTS_DIR, SensorConfig, SimConfig, patient_preset  # noqa: E402
from exo.controllers import MODE_COLORS, MODES, make_controller  # noqa: E402
from exo.gait import TOE_OFF  # noqa: E402
from exo.metrics import gait_cycle_curves  # noqa: E402
from exo.session import Session  # noqa: E402

PATIENTS = ["mild", "moderate", "severe", "stiff-knee"]
KINDS = ["none", "fixed", "cognitive"]
LABEL = {"none": "No assistance", "fixed": "Fixed impedance", "cognitive": "Cognitive adaptive"}
COLOR = {"none": "#9e9e9e", "fixed": "#457b9d", "cognitive": "#e76f51"}
DURATION = 40.0
BANNER = "SIMULATED DATA - virtual patients in a MuJoCo treadmill simulation"


def run_one(scenario: str, patient: str, kind: str, duration: float, patient_kw=None, sensor_kw=None) -> dict:
    s = Session(patient_preset(patient, **(patient_kw or {})), make_controller(kind),
                SimConfig(duration=duration), sensor_cfg=SensorConfig(**(sensor_kw or {})))
    r = s.run()
    log = r.log
    curves = {c: gait_cycle_curves(log, c) for c in ("knee", "knee_ref", "tau_exo", "tau_vol")}
    hist = getattr(s.controller, "history", [])
    return {"scenario": scenario, "patient": patient, "kind": kind, "metrics": r.metrics,
            "perception": r.perception, "strides": r.strides, "events": r.events,
            "curves": {k: (v.mean(0), v.std(0)) for k, v in curves.items()},
            "history": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in h.items()} for h in hist],
            "trace": log[["t", "phase", "knee", "knee_ref", "tau_exo", "tau_vol", "fatigue", "effort",
                          "mode", "phase_hat", "phase_src"]].iloc[::2].reset_index(drop=True)}


def jobs():
    for p in PATIENTS:
        for k in KINDS:
            yield ("main", p, k, DURATION, None, None)
    for k in KINDS:      # fatiguing patient: 5x faster fatigue, 60 s
        yield ("fatigue", "moderate", k, 60.0, {"fatigue_rate": 5.0}, None)
    yield ("insole fault", "moderate", "cognitive", DURATION, None, {"insole_fault_at": 20.0})


# ---------------------------------------------------------------------------------- figures
def fig_summary(runs):
    panels = [("Knee RMSE vs normative (deg)", "Knee RMSE vs normative (deg)"),
              ("Strides with knee buckling (%)", "Strides with knee buckling (%)"),
              ("Toe scuffs per stride", "Toe scuffs per stride"),
              ("Assistance ratio (%)", "Exo share of knee torque (%)"),
              ("Final effort (%)", "Patient effort at end (%)"),
              ("Exo torque RMS (N·m)", "Exo torque RMS (N·m)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = np.arange(len(PATIENTS))
    w = 0.27
    for ax, (key, title) in zip(axes.flat, panels):
        for i, k in enumerate(KINDS):
            vals = [runs[("main", p, k)]["metrics"][key] for p in PATIENTS]
            bars = ax.bar(x + (i - 1) * w, vals, w, color=COLOR[k], label=LABEL[k])
            if key in ("Strides with knee buckling (%)", "Assistance ratio (%)"):
                ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
        ax.set_xticks(x, PATIENTS)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if key == "Final effort (%)":
            ax.set_ylim(60, 102)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{BANNER}\nController comparison, {DURATION:.0f} s sessions (lower is better except patient effort)",
                 fontsize=11, color="#b00020")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiments_summary.png", dpi=140)
    plt.close(fig)


def fig_curves(runs):
    pct = np.linspace(0, 100, 101)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    for ax, p in zip(axes, PATIENTS):
        ax.axvspan(TOE_OFF * 100, 100, color="0.94", zorder=0)
        ref = runs[("main", p, "none")]["curves"]["knee_ref"][0]
        ax.plot(pct, np.rad2deg(ref), "k--", lw=2, label="normative")
        for k in KINDS:
            m, s = runs[("main", p, k)]["curves"]["knee"]
            ax.plot(pct, np.rad2deg(m), color=COLOR[k], lw=2, label=LABEL[k])
            ax.fill_between(pct, np.rad2deg(m - s), np.rad2deg(m + s), color=COLOR[k], alpha=0.15)
        ax.set(title=f"{p} patient", xlabel="gait cycle (%)", xlim=(0, 100))
    axes[0].set_ylabel("right knee flexion (deg)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{BANNER} - knee angle over the gait cycle (mean ± SD, shaded = swing)", fontsize=11, color="#b00020")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiments_knee_curves.png", dpi=140)
    plt.close(fig)


def _mode_strip(ax, trace):
    t, modes = trace["t"].to_numpy(), trace["mode"].to_numpy()
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or modes[i] != modes[start]:
            ax.axvspan(t[start], t[i - 1], color=MODE_COLORS.get(modes[start], "0.8"), lw=0)
            start = i
    ax.set_yticks([])
    ax.set_xlim(t[0], t[-1])


def fig_adaptation(runs, key, fname, title):
    cog, fix = runs[key], runs[(key[0], key[1], "fixed")]
    hist = pd.DataFrame(cog["history"])
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 0.25, 1.3], width_ratios=[1.5, 1], hspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    for r, k in ((fix, "fixed"), (cog, "cognitive")):
        st = r["strides"]
        ax.plot(st["t_start"], st["rmse_deg"], "o-", ms=3, color=COLOR[k], label=LABEL[k])
        b = st[st["buckled"]]
        ax.plot(b["t_start"], b["rmse_deg"], "x", color="k", ms=7, mew=2)
    ax.set(ylabel="stride RMSE (deg)", xlabel="time (s)", title="Tracking error per stride (x = knee buckled)")
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    for r, k in ((fix, "fixed"), (cog, "cognitive")):
        st = r["strides"]
        ax.plot(st["t_start"], 100 * st["assist_ratio"], "o-", ms=3, color=COLOR[k], label=f"{LABEL[k]}: exo share")
        ax.plot(st["t_start"], 100 * st["effort"], "--", color=COLOR[k], lw=1.2, label=f"{LABEL[k]}: patient effort")
    ax.plot(cog["trace"]["t"], 100 * cog["trace"]["fatigue"], color="#6d597a", lw=1, label="true fatigue (cognitive run)")
    ax.set(ylabel="%", ylim=(0, 135), title="Assistance share, patient effort and fatigue")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    ax = fig.add_subplot(gs[2, 0], sharex=ax)
    _mode_strip(ax, cog["trace"])
    ax.set_title("Cognitive supervisor mode (x axis: time in s)", fontsize=10)
    for m in MODES:
        ax.plot([], [], "s", color=MODE_COLORS[m], label=m)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.4))
    ax = fig.add_subplot(gs[3, 0])
    G = np.array(hist["gain_profile"].tolist())
    im = ax.imshow(G.T, aspect="auto", origin="lower", cmap="magma", extent=[1, len(G), 0, 100], vmin=0, vmax=2)
    ax.axhline(TOE_OFF * 100, color="w", ls="--", lw=1)
    ax.set(xlabel="stride", ylabel="gait cycle (%)", title="Learned assist-as-needed gain per phase bin")
    fig.colorbar(im, ax=ax, pad=0.01, label="gain")
    ax = fig.add_subplot(gs[:, 1])
    ax.axis("off")
    wrapped = []
    for t, msg in cog["events"][:22]:
        wrapped += textwrap.wrap(f"t = {t:5.1f} s  {msg}", 62, subsequent_indent="             ")
    ax.text(0, 1, "Explainable decision log (cognitive controller)\n\n" + "\n".join(wrapped[:60]),
            va="top", family="monospace", fontsize=7.2)
    fig.suptitle(f"{BANNER}\n{title}", fontsize=11, color="#b00020")
    fig.savefig(RESULTS_DIR / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    out = Parallel(n_jobs=-1)(delayed(run_one)(*j) for j in jobs())
    runs = {(r["scenario"], r["patient"], r["kind"]): r for r in out}
    print(f"{len(out)} sessions simulated in {time.perf_counter() - t0:.0f} s")

    rows = [{"scenario": r["scenario"], "patient": r["patient"], "controller": LABEL[r["kind"]], **r["metrics"]}
            for r in out]
    table = pd.DataFrame(rows)
    table.round(3).to_csv(RESULTS_DIR / "experiments_metrics.csv", index=False)
    cols = ["Knee RMSE vs normative (deg)", "Strides with knee buckling (%)", "Toe scuffs per stride",
            "Assistance ratio (%)", "Final effort (%)", "Final fatigue (%)"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(table[["scenario", "patient", "controller"] + cols].round(1).to_string(index=False))

    fig_summary(runs)
    fig_curves(runs)
    fig_adaptation(runs, ("main", "severe", "cognitive"), "experiments_adaptation_severe.png",
                   "Severe patient: learning, safety holds and patience (cognitive vs fixed)")
    fig_adaptation(runs, ("fatigue", "moderate", "cognitive"), "experiments_fatigue.png",
                   "Fatiguing patient (5x fatigue rate, 60 s): slacking-vs-fatigue reasoning")
    with open(RESULTS_DIR / "cognitive_decision_logs.txt", "w", encoding="utf-8") as f:
        for r in out:
            if r["kind"] == "cognitive":
                f.write(f"=== {r['scenario']} / {r['patient']} (SIMULATED) ===\n")
                f.writelines(f"t={t:6.2f} s  {m}\n" for t, m in r["events"])
                f.write("\n")
    with open(RESULTS_DIR / "experiments.pkl", "wb") as f:
        pickle.dump({"runs": runs, "patients": PATIENTS, "kinds": KINDS, "duration": DURATION}, f)
    print(f"Saved results/experiments_*.png, experiments_metrics.csv, experiments.pkl, cognitive_decision_logs.txt "
          f"({time.perf_counter() - t0:.0f} s total)")


if __name__ == "__main__":
    main()

"""Clinical-style gait and assistance metrics (computed on SIMULATED data)."""
import numpy as np
import pandas as pd

from .gait import TOE_OFF

NORMATIVE_PEAK_SWING_DEG = 61.0
BUCKLE_DEG = 12.0        # stance flexion this far above normative counts as a buckling event


def _rms(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def stride_table(log: pd.DataFrame) -> pd.DataFrame:
    """One row per complete stride (first, transient stride excluded)."""
    if log.empty:
        return pd.DataFrame()
    rows = []
    last = log["stride"].max()
    for k, g in log.groupby("stride"):
        if k == 0 or k == last:      # skip the start-up and the incomplete final stride
            continue
        err = np.rad2deg(g["knee"] - g["knee_ref"])
        st = g[g["phase"] < TOE_OFF]
        sw = g[g["phase"] >= TOE_OFF]
        excess = np.rad2deg((st["knee"] - st["knee_ref"]).max()) if len(st) else 0.0
        rows.append({
            "stride": int(k), "t_start": float(g["t"].iloc[0]),
            "rmse_deg": _rms(err),
            "peak_swing_deg": float(np.rad2deg(sw["knee"].max())) if len(sw) else np.nan,
            "stance_excess_deg": float(excess),
            "buckled": bool(excess > BUCKLE_DEG),
            "min_toe_clearance_cm": float(sw["toe_height"].min() * 100) if len(sw) else np.nan,
            "scuffs": int(g["scuff"].astype(int).diff().clip(lower=0).sum()),
            "rom_deg": float(np.rad2deg(g["knee"].max() - g["knee"].min())),
            "exo_rms_nm": _rms(g["tau_exo"]),
            "patient_rms_nm": _rms(g["tau_vol"]),
            "assist_ratio": float(np.abs(g["tau_exo"]).sum() /
                                  (np.abs(g["tau_exo"]).sum() + np.abs(g["tau_vol"]).sum() + 1e-9)),
            "fatigue": float(g["fatigue"].iloc[-1]),
            "effort": float(g["effort"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def gait_cycle_curves(log: pd.DataFrame, column: str = "knee", n: int = 101) -> np.ndarray:
    """Resample each complete stride onto 0-100 % of the gait cycle -> array (strides, n)."""
    pct = np.linspace(0, 1, n)
    curves = []
    last = log["stride"].max()
    for k, g in log.groupby("stride"):
        if k == 0 or k == last or len(g) < 10:
            continue
        curves.append(np.interp(pct, g["phase"].to_numpy(), g[column].to_numpy()))
    return np.array(curves)


def summary(log: pd.DataFrame, strides: pd.DataFrame) -> dict:
    if strides.empty:
        return {}
    rom = strides["rom_deg"].mean()
    dt = float(np.median(np.diff(log["t"]))) if len(log) > 1 else 0.0
    exo_power = log["tau_exo"] * log["knee_vel"]
    return {
        "Knee RMSE vs normative (deg)": strides["rmse_deg"].mean(),
        "Peak swing flexion (deg)": strides["peak_swing_deg"].mean(),
        "Swing flexion deficit (deg)": NORMATIVE_PEAK_SWING_DEG - strides["peak_swing_deg"].mean(),
        "Max stance excess flexion (deg)": strides["stance_excess_deg"].mean(),
        "Strides with knee buckling (%)": 100.0 * strides["buckled"].mean(),
        "Toe scuffs per stride": strides["scuffs"].mean(),
        "Knee ROM (deg)": rom,
        "Exo torque RMS (N·m)": _rms(log["tau_exo"]),
        "Exo peak torque (N·m)": float(np.abs(log["tau_exo"]).max()),
        "Exo positive work per stride (J)": float(exo_power.clip(lower=0).sum() * dt / max(len(strides), 1)),
        "Patient torque RMS (N·m)": _rms(log["tau_vol"]),
        "Assistance ratio (%)": 100.0 * strides["assist_ratio"].mean(),
        "Final fatigue (%)": 100.0 * float(log["fatigue"].iloc[-1]),
        "Final effort (%)": 100.0 * float(log["effort"].iloc[-1]),
        "Strides analysed": int(len(strides)),
    }


def perception_summary(log: pd.DataFrame, true_hs: list, detected_hs: list, warmup: float = 1.5) -> dict:
    """Accuracy of the exo's estimates against simulator ground truth."""
    if log.empty:
        return {}
    g = log[log["t"] > warmup]
    err = 100.0 * ((g["phase_hat"] - g["phase"] + 0.5) % 1.0 - 0.5)
    out = {"Phase error MAE (% cycle)": float(np.abs(err).mean()),
           "Phase error RMS (% cycle)": _rms(err)}
    for src, name in ((0, "insole"), (1, "phase portrait")):
        e = err[g["phase_src"] == src]
        out[f"Phase error MAE, {name} (% cycle)"] = float(np.abs(e).mean()) if len(e) else float("nan")
    out["Time on phase-portrait fallback (%)"] = 100.0 * float((g["phase_src"] == 1).mean())

    t_end = float(log["t"].iloc[-1])
    truth = np.array([t for t in true_hs if 0.2 < t < t_end - 0.1])
    det = np.array(detected_hs)
    matched, timing = 0, []
    for t in truth:
        if det.size and np.min(np.abs(det - t)) < 0.15:
            matched += 1
            timing.append(1000.0 * float(det[np.argmin(np.abs(det - t))] - t))
    n_det = int(np.sum((det > 0.2) & (det < t_end - 0.1))) if det.size else 0
    out["Heel strikes detected (%)"] = 100.0 * matched / max(len(truth), 1)
    out["False heel-strike detections"] = max(0, n_det - matched)
    out["Heel-strike timing error (ms)"] = float(np.mean(np.abs(timing))) if timing else float("nan")

    tau, tau_hat = g["tau_patient"].to_numpy(), g["tau_patient_hat"].to_numpy()
    out["Patient torque estimate RMSE (N·m)"] = _rms(tau_hat - tau)
    out["Patient torque estimate correlation r"] = float(np.corrcoef(tau, tau_hat)[0, 1]) if len(tau) > 2 else float("nan")
    out["Patient torque RMS (true, N·m)"] = _rms(tau)
    return out

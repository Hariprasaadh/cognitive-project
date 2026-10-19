"""Phase-1 check: treadmill gait world, physics knee, speed, renders, threaded rendering.

Run:  python scripts/check_model.py
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exo.config import RESULTS_DIR, GaitConfig, SimConfig  # noqa: E402
from exo.gait import TOE_OFF  # noqa: E402
from exo.sim import GaitSim  # noqa: E402


def run_gait(sim: GaitSim, seconds: float, knee_mode: str, frames=None, camera="side"):
    """Walk for `seconds`. knee_mode: 'track' (ideal healthy PD on the patient actuator) or 'passive'."""
    sim.reset()
    dec = sim.cfg.control_decimation
    T = sim.ref.stride_period
    frame_every = int(round(1 / (sim.cfg.render_fps * sim.cfg.dt * dec)))
    log, k = [], 0
    while sim.time < seconds:
        phase = (sim.time / T) % 1.0
        tau = 0.0
        if knee_mode == "track":
            ref, dref = sim.ref.fast("knee", phase)
            e, de = ref - sim.knee_angle, dref - sim.knee_vel
            tau = 400 * e + 20 * de
        sim.step(phase, tau, 0.0, 0.0, n=dec)
        sim.set_visual_drive(max(0, -tau) / 40, max(0, tau) / 40, 0.0)
        log.append((sim.time, phase, sim.knee_angle, sim.ref.fast("knee", phase)[0],
                    sim.foot_contact("r"), sim.data.mocap_pos[sim.pelvis_mocap][2],
                    sim.toe_height("r"), sim.foot_height("r")))
        if frames is not None and k % frame_every == 0:
            frames.append(sim.render(camera))
        k += 1
    return np.array(log, dtype=float)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    sim = GaitSim(SimConfig(render_width=960, render_height=540), GaitConfig(stride_period=1.2))
    m = sim.model
    print(f"Model OK: nq={m.nq} nu={m.nu} nmocap={m.nmocap} dt={m.opt.timestep}s")
    print(f"Belt speed (from normative kinematics): {sim.belt_speed:.2f} m/s, stride {sim.ref.stride_period}s")
    print(f"Initial pelvis height: {sim.data.mocap_pos[sim.pelvis_mocap][2]:.3f} m")

    t0 = time.perf_counter()
    log = run_gait(sim, 10.0, "track")
    wall = time.perf_counter() - t0
    print(f"Speed: 10 s of gait in {wall:.2f} s  ->  {10 / wall:.1f}x real time")

    steady = log[log[:, 0] > 2.4]
    rmse = np.rad2deg(np.sqrt(np.mean((steady[:, 2] - steady[:, 3]) ** 2)))
    print(f"Ideal knee tracking RMSE: {rmse:.2f} deg, knee range "
          f"{np.rad2deg(steady[:, 2].min()):.0f}-{np.rad2deg(steady[:, 2].max()):.0f} deg")
    swing = steady[:, 1] >= TOE_OFF + 0.03
    flat = (steady[:, 1] > 0.12) & (steady[:, 1] < 0.36)
    print(f"Insole contact duty {steady[:, 4].mean() * 100:.0f}% (normative 62%); "
          f"min swing toe clearance {steady[swing, 6].min() * 100:.1f} cm; "
          f"mid-stance foot height {steady[flat, 7].max() * 100:.1f} cm above belt (foot flat)")
    print(f"Pelvis bob: {steady[:, 5].min():.3f}-{steady[:, 5].max():.3f} m")

    plog = run_gait(sim, 4.0, "passive")
    p = plog[plog[:, 0] > 1.2]
    print(f"Passive knee (no torque): RMSE {np.rad2deg(np.sqrt(np.mean((p[:, 2] - p[:, 3]) ** 2))):.1f} deg "
          f"(physics, not scripted)")

    frames = []
    run_gait(sim, 3.6, "track", frames=frames, camera="side")
    imageio.mimsave(RESULTS_DIR / "phase1_walk_side.gif", frames[::2], duration=1 / 15, loop=0)
    imageio.imwrite(RESULTS_DIR / "phase1_side.png", frames[len(frames) // 3])
    sim.reset(0.3)
    imageio.imwrite(RESULTS_DIR / "phase1_iso.png", sim.render("iso", 1280, 720))
    sim.reset(0.72)
    imageio.imwrite(RESULTS_DIR / "phase1_knee.png", sim.render("knee", 1280, 720))
    sim.close()

    # Streamlit runs scripts in a worker thread: make sure offscreen rendering works there
    result = {}

    def worker():
        try:
            s = GaitSim(SimConfig(render_width=320, render_height=180))
            result["shape"] = s.render().shape
            s.close()
        except Exception as exc:  # noqa: BLE001
            result["error"] = repr(exc)

    th = threading.Thread(target=worker)
    th.start()
    th.join()
    print(f"Render from worker thread: {result}")
    print(f"Saved renders to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

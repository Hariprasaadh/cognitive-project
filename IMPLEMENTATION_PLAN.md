# Cognitive Exoskeleton for Knee Rehabilitation: Implementation Plan

> Software-only Cognitive Robotics course project. **All results are SIMULATED.** No hardware and no human data are involved.

## 1. Goal
Simulate a person with a walking disability doing **treadmill gait rehabilitation with body-weight support** (the clinical Lokomat-style setup) while wearing a **powered knee exoskeleton** on the affected right leg. The exoskeleton runs a cognitive **Perceive → Reason → Act → Learn** loop:

| Stage | What the exo does |
|---|---|
| **Perceive** | Reads its sensors (knee encoder, thigh IMU angle, foot-contact switch, motor torque) to track the knee angle, **detect gait events and phase**, and **estimate the patient's own knee torque** |
| **Reason** | Keeps a belief about the patient: where in the gait cycle they struggle, fatigue, slacking, buckling risk. Picks an assistance mode and logs a human-readable reason |
| **Act** | Gait-phase-dependent impedance control plus feedforward torque, inside a safety layer |
| **Learn** | Stride-by-stride **assist-as-needed** gain adaptation and **iterative learning** of feedforward torque. Help goes where it is needed and fades where it is not |

A Streamlit dashboard shows a 3D animated simulation, real-time graphs, adjustable parameters and performance metrics.

## 2. Key design decisions (why it will be reliable within 1 day)
- **Simulator:** MuJoCo 3.16 (already installed and verified). It is lightweight, has a built-in offscreen renderer and an interactive viewer, and runs at more than 40× real time.
- **Walking without falling:** fully dynamic biped walking takes weeks to tune. Instead we use the real clinical setup: the pelvis hangs in a **body-weight-support harness** above a **treadmill**.
  - Hips, ankles and the healthy left leg follow **normative gait data** through **computed-torque servos** (PD on reference position and velocity plus `mj_inverse` feedforward).
  - The pelvis height adapts kinematically so the loaded foot stays on the belt. The load is shared between feet during double support (Perry).
  - The **insole heel and toe switches** are emulated from the gait events, plus physics-based **toe-scuff** contacts in swing. The Phase 3 sensor model adds latency, bounce, spurious pulses and an optional disconnect fault.
  - **The exo never sees ground truth.** Controllers receive only a `Percept` built from noisy wearable sensors (knee encoder, thigh IMU, thigh and shank gyros, insoles, motor torque). Stride periods vary by 2% from stride to stride, so the exo has to estimate the cadence.
  - The affected **right knee is fully physics-simulated**: shank and foot inertia, gravity, a **stance body-weight load** that can make the knee buckle, the patient's impaired muscle torque, and the exo motor torque.
- **Reused models and data, nothing invented from scratch:**
  - Segment lengths and masses from **Winter's anthropometric tables** (H = 1.75 m, M = 70 kg).
  - Normative hip, knee and ankle angles from **Winter / Perry gait data**, as keypoints fitted with a periodic cubic spline.
  - Perry's **gait sub-phases**.
  - **Assist-as-needed** adaptation from Emken et al. (2007).
  - **Iterative learning control** (Arimoto).
  - **Phase-portrait gait phase estimation** (Quintero et al. 2018).
  - Inverse dynamics with `mujoco.mj_inverse`.
- **Disability model (virtual patient, right knee):**
  - Quadriceps weakness (strength %), which causes **knee buckling in stance**.
  - Rectus-femoris spasticity (resistance that grows with knee flexion velocity), which causes **stiff-knee gait in swing**.
  - Fatigue that builds stride by stride.
  - Slacking: effort drops when the exo carries the load.
  - Presets: mild / moderate / severe.

## 3. Architecture
```
 ┌──────────── MuJoCo world (1 kHz) ─────────────┐
 │ treadmill + BWS harness + body + right-knee exo│
 │ hips/ankles/left leg ← normative gait servos   │
 │ right knee ← patient τ + exo τ + stance load   │
 └───────┬───────────────────────────▲────────────┘
         │ sensors (noisy)           │ τ_exo
 ┌───────▼────────┐  ┌──────────────┴───────────────┐
 │ PERCEPTION      │  │ ACTION: impedance + feedfwd  │
 │ • knee angle/vel│  │ + safety (τ, velocity, range)│
 │ • gait events   │  └──────────────▲───────────────┘
 │ • phase φ̂       │                 │
 │ • τ_patient est │  ┌──────────────┴───────────────┐
 └───────┬────────┘  │ COGNITION                     │
         └──────────►│ belief: per-phase deficit,    │
                     │ fatigue, slacking, buckling   │
                     │ modes + explanations          │
                     │ LEARN: AAN gains g[φ], ILC u_ff│
                     └───────────────────────────────┘
```

## 4. Project structure
```
cognitive-project/
  IMPLEMENTATION_PLAN.md   README.md   requirements.txt
  exo/
    __init__.py            # Windows MSVC-runtime preload fix for mujoco
    config.py              # SimConfig, GaitConfig, PatientConfig (+presets), ControllerConfig
    model/knee_exo.xml     # treadmill, harness, body (Winter anthropometrics), right-knee exo
    gait.py                # normative gait reference (periodic splines), Perry sub-phases
    sim.py                 # MuJoCo wrapper: servos, pelvis height, treadmill animation, sensors, render
    patient.py             # impaired knee: weakness, spasticity, fatigue, slacking, stance load
    perception.py          # SensorModel (noise/latency/bounce/fault) + Perception: events, phase, sub-phase, torque
    controllers.py         # NoExo (transparent), FixedImpedance, CognitiveAdaptive (AAN + ILC + supervisor + safety)
    session.py             # closed-loop runner → pandas log, per-stride table, metrics, video
    metrics.py             # clinical-style metrics
  scripts/
    check_model.py         # Phase-1 render/speed check
    run_baseline.py        # Phase-2 no-exo baseline per patient profile
    check_perception.py    # Phase-3 perception accuracy vs ground truth
    run_experiments.py     # controller × patient + fatigue + insole-fault scenarios → results/experiments_*
    live_viewer.py         # interactive native MuJoCo 3D viewer
  app.py                   # Streamlit dashboard
  tests/                   # pytest: model, gait, perception, controllers, session
  results/
```

## 5. Controller summary (for the README and report)
- Reference: `θ_ref(φ̂)` is the normative knee angle at the **estimated** gait phase and cadence. Controllers see only the `Percept`.
- **Fixed impedance (baseline):** `τ = K(θ_ref − θ) + B(θ̇_ref − θ̇)`, with K = 120 N·m/rad (×1.5 in stance) and B = 6.
- **Cognitive adaptive** (Perceive → Reason → Act → Learn): `τ = g[b]·s_mode·[K·dead(θ_ref + Δ_clear − θ, tol[b]) + B·ė] + s_mode·u_ff[b]`
  - **Assist-as-needed tunnel:** no corrective torque while the knee stays within ±3° of the target (Banala/Duschau-Wicke style). This keeps the patient doing the work.
  - **AAN gains** per phase bin (50 bins), updated at every detected heel strike: `g[b] ← 0.9·g[b] + 2·|dead(ē[b])|` (Emken 2007). Assistance grows where the patient fails and fades where they cope.
  - **ILC feedforward:** `u_ff[b] ← 0.9·u_ff[b] + 40·dead(ē[b+1])`, an anticipatory torque learned from stride to stride. It pauses in CHALLENGE.
  - **Goal adaptation:** a perceived toe drag raises the early-swing flexion target `Δ_clear` (+2° per scuff, up to 10°, slow decay) and narrows the swing tunnel. The goal is functional (clear the toe), not just tracking.
  - **Anti-buckling reflex:** stance excess beyond 12° makes the stance stiffness at least ×2 at once.
  - **Explainable supervisor**, run once per stride with hysteresis. Every transition is logged with its reason.
    - `SAFETY_HOLD` (×1.3) after a buckle.
    - `CHALLENGE` (×0.6) after N good strides, or when patient effort drops while error is low, i.e. slacking is suspected.
      - **Hypothesis test:** if a slacking-motivated challenge makes the knee fail, the supervisor concludes **fatigue, not slacking** and switches to `FATIGUE_SUPPORT` (×1.4), with a rest recommendation.
      - **Learned patience:** every failed challenge doubles the number of good strides required before the next one.
    - Perception events (insole fault and recovery) are logged as well.
- **Safety layer** (same for every controller): ±60 N·m, 2000 N·m/s slew limit, never pushes into hyperextension, torque cut on overspeed.

## 6. Dashboard (Streamlit)
- **Live Simulation:** a streamed MuJoCo 3D render with live knee angle vs reference, exo and patient torque, gait phase, assist mode, metric tiles and a decision log.
- **Session Analysis:**
  - An **interactive, rotatable 3D Plotly animation** of the leg and exo, plus an MP4 render.
  - Gait-cycle-normalised knee curves (first vs last strides vs normative).
  - Heatmap of learned assistance gains over phase and stride number.
  - Torque estimation vs ground truth, and phase estimation vs ground truth.
  - Metrics table.
- **Controller Comparison:** no exo vs fixed vs cognitive on the same patient, with bar charts and knee curves.
- **How it works:** architecture diagram, controller equations, assumptions.
- A "SIMULATED DATA" banner on every page and in every figure title.

## 7. Phases (stop for approval after each)
| # | Phase | Deliverables | Done when |
|---|---|---|---|
| 1 | Simulation world + gait reference | `knee_exo.xml`, `gait.py`, `config.py`, `sim.py`, `check_model.py`, Streamlit render spike | the body walks on the treadmill driven by normative gait; right knee is physical; renders OK; offscreen render works inside Streamlit; well above real-time speed |
| 2 | Virtual patient + session loop | `patient.py`, `session.py`, `metrics.py`, NoExo baseline, tests | a no-exo session shows buckling and stiff-knee gait that scale with severity; a 30 s session runs in a few seconds |
| 3 | Perception | `perception.py` (heel strike, phase estimation, sub-phases, torque estimation), tests | phase error < 5 % of the cycle; torque estimate tracks the true patient torque |
| 4 | Controllers + experiments | `controllers.py` (fixed, cognitive), `run_experiments.py`, figures | the cognitive controller brings the knee close to normative, with less assistance and more patient effort than fixed impedance; decisions logged |
| 5 | Dashboard + live viewer | `app.py` (4 tabs), `live_viewer.py` | all tabs work end to end; verified with browser screenshots |
| 6 | Docs + full test pass | README (architecture, controller explanation, demo instructions), full pytest, end-to-end run | everything runs from a clean command line following the README |
| 7 | Report + slides (requested earlier) | report (Claude Doc → Word/PDF), slides (→ .pptx) | they use the real simulated results |

## 8. Verification
- `python -m pytest tests/` passes.
- `python scripts/check_model.py` reports speed and writes renders.
- `python scripts/run_experiments.py` writes CSV and PNG files in under ~3 min.
- `streamlit run app.py`: every tab is checked with Playwright screenshots.
- `python scripts/live_viewer.py` opens the 3D viewer.

## 9. Status tracker
| # | Phase | Status | Notes |
|---|---|---|---|
| 0 | (old elbow-exo Phase 1) | 🔁 Superseded | scope changed to knee/gait; env setup and the mujoco DLL fix are kept |
| 1 | Simulation world + gait reference | ✅ Done | 44× real time, 7/7 tests pass, renders verified |
| 2 | Virtual patient + session loop | ✅ Done | 5 patient profiles, 14/14 tests pass, 30 s session ≈ 1 s |
| 3 | Perception | ✅ Done | phase error 1.0% (insole) / 1.9% (portrait fallback), 100% heel strikes, torque r = 0.96, 25/25 tests pass |
| 4 | Controllers + experiments | ✅ Done | cognitive: half the assistance of fixed for mild/moderate, buckling 34% → 9% (severe), scuffs 0.9 → 0.2 (stiff-knee), fatigue detected; 33/33 tests pass |
| 5 | Dashboard + live viewer | ✅ Done | 4-tab Streamlit app (live, interactive), native viewer with overlays, verified with Playwright |
| 6 | Docs + full test pass | ✅ Done | README (architecture, controllers, demo script, results), 35/35 tests pass |
| 7 | Report + slides | ✅ Done | report (Claude Doc, 7 sections, diagram, chart, figures) + 14-slide deck |

## 10. Progress log
### Environment (kept from the earlier elbow attempt)
- `.venv` (Python 3.12) has mujoco 3.16, streamlit, plotly, pandas, matplotlib, scikit-learn, scipy, imageio-ffmpeg, joblib and pytest.
- **mujoco DLL fix:** `import mujoco` failed with *WinError 1114* because Anaconda's bundled MSVC runtime is 14.29 and mujoco 3.16 needs 14.4x or newer. `exo/__init__.py` preloads `System32\msvcp140.dll` first.

### Phase 1: Simulation world + gait reference (✅ done, 2026-10-19)
- **`exo/model/knee_exo.xml`:**
  - Treadmill with handrails, console and moving belt markers.
  - Body-weight-support harness frame, vest and cable.
  - A 70 kg / 1.75 m person built from Winter's segment tables, with swinging arms.
  - Right-knee exoskeleton: thigh and shank cuffs, lateral bars, a glowing knee motor that is coloured by torque, and a battery.
  - X-ray right leg with quadriceps and hamstrings coloured by patient effort.
  - Cameras: `side`, `knee`, `iso`, `front`.
- **`exo/gait.py`:**
  - Normative hip, knee and ankle curves (Winter and Perry keypoints, periodic cubic splines, 2000-point lookup tables for speed).
  - Perry sub-phases, gait events (foot-flat 8%, heel-off 40%, toe-off 62%) and double-support weight sharing.
  - Thigh angle = normative hip minus 12° pelvic tilt, with a **foot-flat constraint in mid-stance** (thigh = knee − ankle).
- **`exo/sim.py` (`GaitSim`):**
  - Normative servos for the hips, ankles, left leg and arms. The **right knee is physics-only** (patient torque, exo torque and stance-load input).
  - Weight-sharing pelvis height, treadmill belt speed estimated from the kinematics, emulated insole and toe-scuff sensors.
  - 3D skeleton keypoints for Plotly, torque- and effort-based colours, offscreen rendering.
- **Issues found and fixed:**
  1. **Speed was 3.7× real time.** SciPy spline calls in the 500 Hz loop were the bottleneck. Lookup tables brought it to **44× real time**.
  2. **Foot contact timing was off.**
     - Geometric "lowest foot" contact fired about 0.2 cycle late, because pelvis-relative normative hip angles don't fit an upright planar pelvis.
     - The pelvic-tilt correction, the foot-flat constraint and weight-sharing pelvis height fixed the posture: stance foot flat within 0.2 cm.
     - Geometric switches still chattered: 1° of tracking error moves the toe by 1 cm. Insole switches are therefore emulated from gait events (standard sensor emulation), with toe scuff kept physics-based.
- **`scripts/check_model.py` (all results SIMULATED):**
  - Belt 0.97 m/s at a 1.2 s stride.
  - Ideal-tracking knee RMSE 2.0°, knee range 0–62°.
  - Contact duty 64% (normative 62%).
  - Pelvis bob 3.3 cm.
  - Passive knee (no torque) deviates by 13° RMSE, which shows the knee really is driven by physics.
  - Offscreen rendering works from a worker thread, so it is Streamlit-safe.
  - Output: `results/phase1_walk_side.gif`, `phase1_side.png`, `phase1_iso.png`, `phase1_knee.png`.
- **`tests/test_world.py`:** 7 tests pass (reference periodicity and normative ranges, lookup-table accuracy, sub-phases, foot-flat, tracked walk, insole timing, rendering).
### Phase 2: Virtual patient + session loop (✅ done, 2026-10-19)
- **`exo/patient.py` (`VirtualPatient`):** the affected right knee's neuromuscular torque.
  - Feedforward = the knee's inverse dynamics from the normative gait (`mj_inverse` table) minus the expected stance load.
  - Feedback = intrinsic muscle impedance plus a **25 ms delayed stretch reflex**.
  - **Weakness:** torque capacity = strength × (1 − fatigue) × 120 N·m, with reduced neural gain. Flexors are 60% as strong as extensors.
  - **Spasticity:** rectus-femoris hyperreflexia, phase-gated to pre-swing through mid-swing (velocity-dependent). This produces stiff-knee gait and toe scuffing.
  - **Fatigue:** simplified 3-compartment model (accumulates with relative effort, slow recovery).
  - **Slacking:** effort drops in proportion to the exo's share of the torque (3 s adaptation).
  - Signal-dependent motor noise.
  - **Stance load:** 0.30 m × body weight × (1 − BWS) × weight share × sin(knee). It grows with flexion, which is the buckling feedback, and the BWS harness catches a collapse beyond 15° excess flexion.
- **`exo/controllers.py`:** `Observation` (sensor-only data structure), `Command`, `Controller` base with an explainable `events` log, the `NoExo` transparent baseline, and `make_controller`.
- **`exo/session.py` (`Session`):** world + patient + controller closed loop at 500 Hz, with a 100 Hz log of 20 channels. `advance()` supports live dashboards (verified identical to `run()`). Optional frame capture.
- **`exo/metrics.py`:** a per-stride table and a summary:
  - knee RMSE vs normative, peak swing flexion and swing deficit
  - stance excess flexion and % of strides with buckling (more than 12° above normative)
  - toe clearance and scuffs
  - exo RMS, peak torque and positive work; patient torque RMS
  - assistance ratio, fatigue, effort
  - plus gait-cycle-normalised curves.
- **Patient profiles:** healthy, mild, moderate, severe, and a new **stiff-knee** profile (strong quadriceps, high spasticity) that isolates the swing deficit.
- **No-exo baseline (`scripts/run_baseline.py` → `results/baseline_knee_curves.png`, `baseline_metrics.csv`, `baseline_severe_buckling.png`; SIMULATED):**

  | Profile | RMSE | Peak swing (normal 61.5°) | Stance excess | Buckling | Scuffs/stride |
  |---|---|---|---|---|---|
  | healthy | 3.8° | 62.8° | 5.6° | 0% | 0 |
  | mild | 4.5° | 60.7° | 9.0° | 0% | 0 |
  | moderate | 14.2° | 59.5° | 33° | 100% | 0.13 |
  | severe | 24.9° | 37.9° | 55° | 100% | 1.0 |
  | stiff-knee | 4.8° | 56.3° | 9.3° | 0% | 1.0 |
- **Issues found and fixed:**
  1. **The inverse-dynamics feedforward spanned ±200 N·m.** The spline keypoints and the foot-flat blend gave thigh accelerations up to 179 rad/s², which couple into the knee through inertia. Gait curves are now **truncated Fourier series** (Winter's representation): 10 harmonics, and 5 for the thigh. The feedforward is now −17 to +26 N·m and the foot stays flat within 1.5°.
  2. **Even a healthy patient deviated by 6°.** The normative joints used position servos with a 50 ms lag, which mistimed the hip-knee coupling. They are now **computed-torque servos** (`general` actuators with affine bias, references plus `mj_inverse` feedforward). Ideal knee tracking improved from 2.0° to 1.4°, and healthy patient RMSE from 6° to 3.8°.
  3. **Spasticity reduced buckling, which is unphysiological.** It is now phase-gated to swing, with weaker flexors.
  4. **Logging bug:** each row logged the phase from before the physics step but the state from after it, which broke the per-stride curves. Phase, reference and state are now logged for the same instant.
  5. **Fatigue rose 15% in 20 s even when healthy.** The rate is lowered (moderate now reaches about 14% in 30 s; there is a slider multiplier).
- **Tests:** `tests/test_patient_session.py` adds 7 tests (healthy normal, severity ordering, stiff-knee is a swing deficit, log and stride consistency, BWS reduces buckling, incremental advance = run, speed). **14/14 pass.**
### Phase 3: Perception (✅ done, 2026-10-19)
- **`exo/perception.py`:**
  - **`SensorModel`** turns the true state into wearable-sensor readings:
    - 14-bit knee encoder with noise
    - thigh IMU with 1° mounting bias and 0.5° noise
    - thigh and shank gyros, whose difference gives knee velocity
    - insole switches with 15 ms latency, 6 ms contact bounce and spurious vibration pulses
    - noisy motor-torque sensing
    - optional **insole disconnect fault**, for the demo.
  - **`Perception`** works from those measurements only:
    - Butterworth-filtered velocity and acceleration.
    - Debounced heel/toe switches (8 ms).
    - **Heel-strike detection** with a refractory window; the detection is time-stamped back by the known debounce and latency delay.
    - **Toe-off detection** that ignores swing-phase toe drags.
    - Adaptive stride period (median of the last 3 strides).
  - **Gait phase**, primary estimate: time since heel strike ÷ estimated stride period.
  - **Fallback: self-calibrating thigh phase portrait** (Quintero 2018). Phase is the polar angle of (thigh angle, −thigh velocity), normalised over each stride.
    - A 50-bin correction table starts from the normative gait model and is **learned online** while the insoles are trusted.
    - A forward-only phase tracker (cadence prior plus portrait correction) handles the ambiguous thigh plateau around heel strike.
    - The fallback takes over as soon as a heel strike is overdue, and the insole is flagged faulty after 1.5 strides with none.
  - **Sub-phase:** Perry's 7 sub-phases; stance/swing from the insoles, or from the phase if the insoles have failed.
  - **Torque estimation:**
    - Inverse dynamics (`mj_inverse` on a private copy of the leg model, using measured thigh and knee motion and the normative ankle at the estimated phase) gives the total knee torque.
    - Patient muscle torque = that total − measured exo torque − modelled body-weight load (known BWS, weight share and harness catch).
    - The estimate is held while the knee is at the hyperextension stop, because impact forces there are not muscle torque.
  - Controllers now receive a **`Percept`** (`controllers.py`), never ground truth.
- **`exo/session.py`:**
  - The true phase is integrated, with a **2% stride-to-stride period variability** (`GaitConfig.stride_cv`).
  - The session runs sensors → perception → controller each step and logs 8 extra perception channels.
  - `SessionResult.perception` holds the accuracy metrics (`metrics.perception_summary`).
- **Results** (`scripts/check_perception.py` → `results/perception.png`, `perception_metrics.csv`; SIMULATED, 20 s sessions):

  | Scenario | Phase MAE | HS detected | False HS | HS timing error | Patient torque r / RMSE |
  |---|---|---|---|---|---|
  | healthy | 1.0% | 100% | 0 | 5.8 ms | 0.95 / 7.7 N·m |
  | moderate | 1.0% | 100% | 0 | 5.3 ms | 0.96 / 8.1 N·m |
  | severe | 1.0% | 100% | 0 | 5.9 ms | 0.90 / 8.2 N·m |
  | stiff-knee (toe drags) | 1.1% | 100% | 0 | 6.0 ms | 0.95 / 10.1 N·m |
  | moderate, insole unplugged at 10 s | 1.5% overall; portrait 1.9% | n/a after the fault | 0 | – | 0.91 / 11.7 N·m |
- **Issues found and fixed:**
  1. **The torque estimate lagged** (r = 0.88, then r = 0.98 when shifted 30 ms). Double-differentiating the encoder needed heavy filtering. Knee velocity now comes from the shank and thigh gyros, as on real exos, and the filters are lighter: r = 0.96 with no lag compensation.
  2. **The severe patient's estimate was poor** (r = 0.54). Its knee buckles to 60°, then snaps into the −5° hyperextension stop, and the stop's impact forces looked like −280 N·m of "muscle" torque. The estimate is now held at the stop: r = 0.90.
  3. **The fallback phase was off by 5%.** After a missed heel strike the event phase saturated at 0.999, and the portrait kept learning from that wrong phase. Learning now stops once a heel strike is overdue. A forward-only tracker smooths the plateau near heel strike. Result: 1.9%.
- **Tests:** `tests/test_perception.py` adds 11 tests: debouncer, filter, insole latency and fault, phase wrap, phase error under 5% and heel-strike detection for moderate and stiff-knee, sub-phase agreement, insole-fault fallback, torque-estimate accuracy. The stride-count test was adapted to the variable stride. **25/25 pass.** The baseline was regenerated and is essentially unchanged with stride variability.
### Phase 4: Controllers + experiments (✅ done, 2026-10-19)
- **`exo/controllers.py`:**
  - `NoExo`, `FixedImpedance` and `CognitiveAdaptive` (see §5), plus a shared safety layer, `MODES` / `MODE_COLORS` and `make_controller("none"|"fixed"|"cognitive")`.
  - Per-step `info` (mode, gain, feedforward, target, safety flag) is logged.
  - The cognitive controller keeps a per-stride `history` (beliefs, gain and feedforward profiles) for the dashboard.
- **Perception additions:**
  - **Toe-drag detection:** the toe switch is still closed after toe off is due, with a latency-compensated margin. It is exposed as `Percept.toe_scuff`.
  - Stance now runs from heel strike to the detected toe off, so a dragging toe no longer counts as stance.
  - The physics scuff is counted from 3% after toe off, so the toe-off roll doesn't count as a scuff.
- **Experiments** (`scripts/run_experiments.py`): 16 sessions in parallel (joblib), about 1 min. Outputs:
  - `results/experiments_summary.png`, `experiments_knee_curves.png`
  - `experiments_adaptation_severe.png`, `experiments_fatigue.png`
  - `experiments_metrics.csv`, `cognitive_decision_logs.txt`
  - `experiments.pkl`, which the dashboard loads.
- **Results (SIMULATED, 40 s sessions; fatigue scenario 60 s at 5× fatigue rate):**

  | Patient | Controller | RMSE | Buckling | Scuffs/stride | Exo share | Patient effort |
  |---|---|---|---|---|---|---|
  | mild | fixed / cognitive | 1.9° / 3.0° | 0 / 0% | 0 / 0 | **26% / 13%** | **87% / 93%** |
  | moderate | none / fixed / cognitive | 14.4° / 2.8° / 3.9° | 100 / 0 / 0% | 0 / 0.3 / 0 | **30% / 15%** | **84% / 93%** |
  | severe | none / fixed / cognitive | 25.2° / 5.7° / **3.9°** | 100 / 34 / **9%** | 1.0 / 0.7 / **0.3** | 48% / 47% | 76% / 80% |
  | stiff-knee | none / fixed / cognitive | 4.9° / 2.9° / 2.5° | 0% | 1.0 / 0.9 / **0.2** | 23% / 29% | 87% / 87% |
  | fatiguing moderate | fixed / cognitive | 4.1° / **3.5°** | 22% / **4%** | 0.3 / 0.1 | 39% / 34% | 77% / 81% |
  | moderate, insole unplugged at 20 s | cognitive | 3.9° | 3% | 0 | 19% | 90% |

  The cognitive controller deliberately allows ±3° (the tunnel), so its RMSE is slightly higher than fixed for mild and moderate patients. In return it uses **half the assistance** and the patient **keeps working** (less slacking).
- **Tuning issues found and fixed:**
  1. **The first AAN version assisted more than fixed** (κ = 8 drove gains to the maximum). The ±3° tunnel and κ = 2 fixed it.
  2. **Slacking spiral:** ILC feedforward kept growing while the patient slacked (assistance 19% → 40%). Stronger ILC forgetting (0.9), an explicit slacking rule, and pausing ILC in CHALLENGE fixed it.
  3. **Stiff-knee scuffs persisted** even with 2–3° tracking error, because the toe clearance in early swing is very sensitive in this geometry and the spasticity resists up to 70 N·m. The fix was goal adaptation: raise the swing flexion target after a perceived drag.
  4. **Endless CHALLENGE → buckle → HOLD cycles for the severe patient:** fixed by learned patience, which doubles after each failure.
  5. **Fatigue mistaken for slacking:** fixed by the challenge-as-hypothesis-test rule, which switches to FATIGUE_SUPPORT and recommends a rest.
- **Tests:** `tests/test_controllers.py` adds 8 tests:
  - controller factory, deadband, safety layer (torque, slew, hyperextension, overspeed)
  - moderate buckling prevented
  - cognitive uses less assistance and keeps more patient effort
  - cognitive is safer for the severe patient
  - decisions are logged
  - fatigue is told apart from slacking.

  **33/33 pass.**
### Phase 5: Dashboard + live viewer (✅ done, 2026-10-19)
- **`app.py` (Streamlit, dark theme, "SIMULATED DATA" banner on every page):**
  - **Sidebar:** patient profile plus impairment sliders, body-weight support, cadence, controller and its parameters, insole fault, session length.
  - **🎥 Live Simulation:** an `st.fragment(run_every=…)` loop. Each 0.5 s tick simulates, then sends **one animated-WebP clip** (10 fps) plus one update of the tiles, the live Plotly chart (knee vs normative with the AAN tunnel, torques true vs estimated, phase true vs estimated, mode markers) and the decision log. It is paced to the wall clock (1× real time).
    - Live interventions: **inject fatigue**, **unplug insole**, **hot-swap controller**. Pause and resume, camera choice, speed 0.5–4×.
  - **📊 Session Analysis:** headless run (30 s in about 3 s) with an optional MP4. Shows:
    - metric tiles, a **rotatable 3D Plotly replay**, the decision log
    - gait-cycle curves (first vs last strides)
    - a learning-across-strides chart and the learned-gain heat-map
    - perception vs ground truth, plus a CSV download.
  - **⚖ Controller Comparison:** the pre-computed `experiments.pkl` (4 patients plus the fatigue scenario), or a custom three-controller run.
  - **🧠 How it works:** Graphviz architecture, LaTeX equations, method notes, limitations.
- **`exo/viz.py`:** Plotly 3D skeleton animation (legs, torso, exo bars, a motor coloured by torque, treadmill) and MP4 export.
- **`exo/session.py`:** skeleton recording, `set_controller`, `inject_fatigue`, `fail_insole`.
- **`scripts/live_viewer.py`:** native MuJoCo passive viewer, tracking camera.
  - Text overlay: mode, phase, sub-phase, phase source, knee, torque, fatigue, effort, last decision.
  - Live knee-angle figure.
  - Keys `1`/`2`/`3` controller, `4` fatigue, `5` insole fault. The number keys were chosen to avoid MuJoCo's letter shortcuts.
- **`scripts/make_demo_media.py`:** `results/demo_cognitive_side.gif` and `results/demo_comparison.mp4` (no exo vs cognitive, side by side).
- **Issues found and fixed:**
  1. **Cross-thread OpenGL segfault:** MuJoCo GL contexts are thread-bound, and Streamlit reruns on new threads. All rendering now runs on **one persistent GL worker thread** (`sim._GL_THREAD`), renderers are freed there on garbage collection, and they are closed before interpreter exit.
  2. **`np.clip` on scalars took about 20 % of the loop time.** A `config.clamp` helper replaced it; the loop is now about 6× real time with full perception.
  3. **The browser could not keep up with per-frame updates.** Media-URL 404s and dropped chart updates followed. One inline animated clip per 0.5 s tick fixed it.
  4. **Layout polish:** legend overlaps, the wide architecture graph (now top-to-bottom) and clipped bar labels.
- **Verified with Playwright screenshots** (`results/dashboard_live.png`): all four tabs, a live session to completion, fatigue injection leading to FATIGUE_SUPPORT, and insole unplugging leading to the phase-portrait fallback. Plus `tests/test_app.py` (AppTest headless run, 3D/MP4 export).
### Phase 6: Docs + full test pass (✅ done, 2026-10-19)
- **`README.md`:**
  - overview with the demo GIF, quick start, a 5-minute faculty demo script
  - Mermaid architecture, reused models and data, code map
  - patient, perception and controller explanations (equations, the supervisor mode table, hypothesis testing)
  - results table and figures, testing, limitations, references.
- `requirements.txt` cleaned (scikit-learn dropped, pillow added, streamlit ≥ 1.37); `.gitignore`; the empty `models/` folder removed.
- **Full test suite: 35/35 pass** (about 2.5 min). `pip install --dry-run -r requirements.txt` resolves.
### Phase 7: Report + slides (✅ done, 2026-10-19)
- **Project report** (Claude Doc, exportable to Word/PDF): https://claude.ai/code/artifact/1db0cc00-02a8-484a-a637-739a7ca41da2
  - Sections: abstract, introduction, related work, architecture (drawn diagram), methods (equations, mode table), results (baseline, perception, a paired fixed-vs-cognitive chart, fatigue and fault scenarios, dashboard figures), discussion and limitations, conclusion, references.
- **Slides** (14-slide deck, exports to PowerPoint/PDF): https://claude.ai/artifact/1BMvyj4LAzUvoYWVuJubck
  - problem → cognitive loop → architecture → world → patient → perception → controller → supervisor → headline results → visual evidence → live demo → limits → closing. Each slide has speaker notes.
  - Placeholders to fill in: `[Your name] · [Course code] · [Date]` on the cover.
- **Every number** in the report and slides comes from `results/*.csv` and is labelled SIMULATED.

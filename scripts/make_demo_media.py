"""Render demo media for the README / slides (SIMULATED).

Run:  python scripts/make_demo_media.py
  -> results/demo_cognitive_side.gif   (severe patient, cognitive exo, side view, ~5 s)
  -> results/demo_comparison.mp4       (same severe patient: no assistance vs cognitive exo, side by side)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import exo  # noqa: E402,F401
import imageio.v2 as imageio  # noqa: E402

from exo.config import RESULTS_DIR, SimConfig, patient_preset  # noqa: E402
from exo.controllers import make_controller  # noqa: E402
from exo.session import Session  # noqa: E402
from exo.viz import save_mp4  # noqa: E402

W, H, FPS = 640, 360, 25


def frames(kind: str, t0: float, t1: float, camera: str = "side") -> list:
    s = Session(patient_preset("severe"), make_controller(kind), SimConfig(duration=t1, render_fps=FPS,
                                                                          render_width=W, render_height=H))
    s.advance(t0)
    s.advance(t1 - t0, render_camera=camera)
    out = list(s.frames)
    s.sim.close()
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 26], fill=(15, 17, 23))
    d.text((10, 6), text, fill=(240, 240, 240))
    d.text((W - 150, 6), "SIMULATED DATA", fill=(255, 90, 90))
    return np.asarray(im)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    cog = frames("cognitive", 20.0, 25.0)
    gif = [label(f, "Severe patient + cognitive knee exoskeleton (MuJoCo)") for f in cog[::2]]
    imageio.mimsave(RESULTS_DIR / "demo_cognitive_side.gif", gif, duration=2 / FPS, loop=0)
    none = frames("none", 20.0, 25.0)
    n = min(len(none), len(cog))
    combo = [np.hstack([label(none[i], "No assistance: knee buckles"),
                        label(cog[i], "Cognitive exo: assist-as-needed")]) for i in range(n)]
    save_mp4(combo, RESULTS_DIR / "demo_comparison.mp4", fps=FPS)
    print("Saved results/demo_cognitive_side.gif and results/demo_comparison.mp4")


if __name__ == "__main__":
    main()

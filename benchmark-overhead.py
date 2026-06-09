"""
Overhead benchmark for the detector pipeline.

Cycles through loading → game → experience, running each phase's detector
stack on a reference image for PHASE_DURATION seconds.  Writes one row per
second to overhead.csv with FPS, CPU %, and RSS RAM.

Usage:  python3 benchmark-overhead.py
"""

import csv
import time

import cv2
import easyocr
import numpy as np
import psutil

from detectors import (
    ArmorDetector, ExperienceDetector, RankedLoadingDetector,
    ScreenClassifier, ShieldDetector, TopRightDetector, WeaponDetector,
)
from config import (
    ARMOR_LEFT_FRAC,  ARMOR_TOP_FRAC,  ARMOR_WIDTH_FRAC,  ARMOR_HEIGHT_FRAC,
    TR_LEFT_FRAC,     TR_TOP_FRAC,     TR_WIDTH_FRAC,     TR_HEIGHT_FRAC,
    WEAPON_LEFT_FRAC, WEAPON_TOP_FRAC, WEAPON_WIDTH_FRAC, WEAPON_HEIGHT_FRAC,
)

PHASE_DURATION  = 10.0   # seconds per phase
SAMPLE_INTERVAL = 1.0    # seconds between CSV rows
OUT_PATH        = "overhead.csv"
HEADERS         = ["phase", "elapsed_s", "fps", "cpu_pct", "ram_mb"]

PHASE_ORDER = ["loading", "game", "experience"]
PHASE_IMAGES = {
    "loading":    "images/ui-images/ranked-loading-1.png",
    "game":       "images/ui-images/game-purple-shield.png",
    "experience": "images/ui-images/experience-screen.png",
}

# ── Load reference images ─────────────────────────────────────────────────────
print("Loading reference images…")
imgs: dict[str, np.ndarray] = {}
for phase, path in PHASE_IMAGES.items():
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Reference image not found: {path}")
    imgs[phase] = img
    print(f"  {phase:12s}: {img.shape[1]}×{img.shape[0]}")

# ── Initialise detectors ──────────────────────────────────────────────────────
print("Loading EasyOCR (CPU)…")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)
reader.readtext(np.zeros((10, 10, 3), np.uint8))   # warm-up

classifier = ScreenClassifier()
ranked_det = RankedLoadingDetector(reader)
weapon_det = WeaponDetector(reader)
armor_det  = ArmorDetector(reader)
tr_det     = TopRightDetector(reader)
shield_det = ShieldDetector()
exp_det    = ExperienceDetector(reader)

# Resolve bar_bottom_y for the experience image once upfront.
_, exp_bar_y = classifier.classify(imgs["experience"])
print(f"  experience bar_bottom_y = {exp_bar_y}px\n")

# ── Per-phase frame functions ─────────────────────────────────────────────────
def _crop(img: np.ndarray, lf: float, tf: float, wf: float, hf: float) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(h * tf) : int(h * (tf + hf)),
               int(w * lf) : int(w * (lf + wf))]


def run_loading(img: np.ndarray) -> None:
    classifier.classify(img)
    ranked_det.detect_map_name(img)
    ranked_det.detect_squad_distribution(img)


def run_game(img: np.ndarray) -> None:
    classifier.classify(img)
    weapon_det.detect(_crop(img, WEAPON_LEFT_FRAC, WEAPON_TOP_FRAC,
                                  WEAPON_WIDTH_FRAC, WEAPON_HEIGHT_FRAC))
    armor_det.detect( _crop(img, ARMOR_LEFT_FRAC,  ARMOR_TOP_FRAC,
                                  ARMOR_WIDTH_FRAC,  ARMOR_HEIGHT_FRAC))
    tr_det.detect(    _crop(img, TR_LEFT_FRAC,     TR_TOP_FRAC,
                                  TR_WIDTH_FRAC,     TR_HEIGHT_FRAC))
    shield_det.detect(img)


def run_experience(img: np.ndarray) -> None:
    classifier.classify(img)
    exp_det.reset()          # bypass cache so each frame pays the full OCR cost
    exp_det.detect(img, exp_bar_y)


RUNNERS = {
    "loading":    run_loading,
    "game":       run_game,
    "experience": run_experience,
}

# ── Benchmark ─────────────────────────────────────────────────────────────────
proc = psutil.Process()
proc.cpu_percent()   # first call returns 0.0; discard it

rows: list[dict] = []

for phase in PHASE_ORDER:
    img    = imgs[phase]
    runner = RUNNERS[phase]
    print(f"[{phase.upper():12s}] running {PHASE_DURATION:.0f} s…")

    phase_start   = time.monotonic()
    window_start  = phase_start
    window_frames = 0

    while True:
        now       = time.monotonic()
        elapsed_s = now - phase_start
        if elapsed_s >= PHASE_DURATION:
            break

        runner(img)
        window_frames += 1

        if now - window_start >= SAMPLE_INTERVAL:
            dt  = now - window_start
            fps = window_frames / dt
            cpu = proc.cpu_percent()
            ram = proc.memory_info().rss / 1024 / 1024
            row = {
                "phase":     phase,
                "elapsed_s": round(elapsed_s, 1),
                "fps":       round(fps, 2),
                "cpu_pct":   round(cpu, 1),
                "ram_mb":    round(ram, 1),
            }
            rows.append(row)
            print(f"  t={elapsed_s:4.1f}s  fps={fps:5.2f}  "
                  f"cpu={cpu:5.1f}%  ram={ram:.0f} MB")
            window_start  = now
            window_frames = 0

    print()

# ── Write CSV ─────────────────────────────────────────────────────────────────
with open(OUT_PATH, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=HEADERS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows → {OUT_PATH}")

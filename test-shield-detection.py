"""
Correctness check for ShieldDetector.
Run with:  python3 test-shield-detection.py

Annotated output images are written to tests/.
"""

import os
import time
import cv2
import numpy as np

from detectors import ShieldDetector
from config import (
    SHIELD_BAR_X1, SHIELD_BAR_X2,
    SHIELD_STRIP_Y1, SHIELD_STRIP_Y2, SHIELD_BAR_MID_Y,
)

OUT_DIR = "tests"

SCREENS = [
    "images/ui-images/game-white-shield.png",
    "images/ui-images/game-blue-shield.png",
    "images/ui-images/game-purple-shield.png",
]

EXPECTED = {
    "game-white-shield.png": {
        "shield_type": "white",
        "shield_hp":   22,
        "flesh_hp":    100,
        "health":      122,
    },
    "game-blue-shield.png": {
        "shield_type": "blue",
        "shield_hp":   67,
        "flesh_hp":    100,
        "health":      167,
    },
    "game-purple-shield.png": {
        "shield_type": "purple",
        "shield_hp":   90,
        "flesh_hp":    100,
        "health":      190,
    },
}

SHIELD_COLOURS = {
    "white":  (230, 230, 230),
    "blue":   (220, 140,  40),   # BGR
    "purple": (200,  60, 180),
    "none":   ( 80,  80,  80),
}


def _annotate(img: np.ndarray, result: dict) -> np.ndarray:
    """Draw shield/health bar overlays and result text on a copy of img."""
    vis = img.copy()
    h, w = vis.shape[:2]

    x1 = int(w * SHIELD_BAR_X1)
    x2 = int(w * SHIELD_BAR_X2)
    sy1, sy2 = int(h * SHIELD_STRIP_Y1), int(h * SHIELD_BAR_MID_Y)
    hy1, hy2 = int(h * SHIELD_BAR_MID_Y), int(h * SHIELD_STRIP_Y2)

    stype  = result.get("shield_type", "none")
    s_col  = SHIELD_COLOURS.get(stype, SHIELD_COLOURS["none"])
    h_col  = (100, 220, 100)   # green for flesh

    # Shield bar region
    cv2.rectangle(vis, (x1, sy1), (x2, sy2), s_col, 2)
    # Health bar region
    cv2.rectangle(vis, (x1, hy1), (x2, hy2), h_col, 2)

    # Label the dividing line
    cv2.line(vis, (x1, int(h * SHIELD_BAR_MID_Y)),
                  (x2, int(h * SHIELD_BAR_MID_Y)), (180, 180, 180), 1)

    # Result text block — bottom-left, above the bars
    lines = [
        f"shield: {stype}  {result.get('shield_hp', '?')} HP",
        f"flesh:  {result.get('flesh_hp', '?')} HP",
        f"health: {result.get('health', '?')} HP",
    ]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
    pad, line_h = 12, 44
    block_h = len(lines) * line_h + pad * 2
    block_y = sy1 - block_h - 8
    cv2.rectangle(vis, (x1, block_y), (x1 + 460, sy1 - 8), (30, 30, 30), -1)
    for i, line in enumerate(lines):
        col = s_col if i == 0 else (h_col if i == 1 else (220, 220, 220))
        cv2.putText(vis, line, (x1 + pad, block_y + pad + (i + 1) * line_h),
                    font, scale, col, thick, cv2.LINE_AA)

    return vis


detector = ShieldDetector()

totals = {"first": [], "second": []}

for path in SCREENS:
    img = cv2.imread(path)
    if img is None:
        print(f"\n[SKIP] {path}")
        continue

    name = path.split("/")[-1]
    h, w = img.shape[:2]
    print(f"\n{'='*52}")
    print(f"  {name}  ({w}×{h})")
    print(f"{'='*52}")

    t0 = time.perf_counter()
    result = detector.detect(img)
    elapsed = time.perf_counter() - t0
    totals["first"].append(elapsed)
    print(f"  detect()  {elapsed*1000:.1f}ms")

    if result is None:
        print("  [no bars found]")
        continue

    exp = EXPECTED.get(name, {})
    ok = err = 0
    for key, val in result.items():
        if key not in exp:
            continue
        expected_val = exp[key]
        match = str(val) == str(expected_val)
        sym  = "✓" if match else "✗"
        tick = "✓" if match else f"✗ (expected {expected_val!r})"
        if match: ok += 1
        else:      err += 1
        print(f"    {sym} {key:<16} got={val!r}  {tick}")

    print(f"  score: {ok}/{ok+err} checked fields")

    out_path = os.path.join(OUT_DIR, f"shield_{name}")
    cv2.imwrite(out_path, _annotate(img, result))
    print(f"  → {out_path}")

print(f"\n{'='*52}")
print("  Second pass (same instance):")
for path in SCREENS:
    img = cv2.imread(path)
    if img is None:
        continue
    t0 = time.perf_counter()
    detector.detect(img)
    elapsed = time.perf_counter() - t0
    totals["second"].append(elapsed)
    print(f"    {path.split('/')[-1]:<35} {elapsed*1000:.1f}ms")

print(f"\n  avg first  pass:  {sum(totals['first'])/len(totals['first'])*1000:.1f}ms")
print(f"  avg second pass:  {sum(totals['second'])/len(totals['second'])*1000:.1f}ms")

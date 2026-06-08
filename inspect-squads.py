"""
Visual debugger for detect_squad_distribution.

For each reference ranked-loading screen, draws:
  - bar bounding box
  - per-segment boundaries, colour label, icon match, OCR count, width-prop count
  - icon search region above each segment

Saves annotated images to /tmp/squads_debug_{n}.png
Run with:  python3 inspect-squads.py
"""

import cv2
import easyocr
import numpy as np

from detectors import RankedLoadingDetector
from config import RANKED_DIST_ICON_SEARCH_PX, RANKED_SQUAD_TOTAL

SCREENS = {
    1: {"path": "images/ui-images/ranked-loading-1.png",
        "expected": {"Diamond": 13, "Master": 4, "Apex Predator": 3}},
    2: {"path": "images/ui-images/ranked-loading-2.png",
        "expected": {"Diamond": 16, "Master": 3, "Apex Predator": 1}},
    3: {"path": "images/ui-images/ranked-loading-3.png",
        "expected": {"Diamond": 16, "Master": 3, "Apex Predator": 1}},
    4: {"path": "images/ui-images/ranked-loading-4.png",
        "expected": {"Diamond": 20}},
}

print("Loading EasyOCR (CPU)…")
reader = easyocr.Reader(["en"], gpu=False)
det    = RankedLoadingDetector(reader)

for n, info in SCREENS.items():
    img      = cv2.imread(info["path"])
    h, w     = img.shape[:2]
    annotated = img.copy()

    # --- run detector internals to get bar + segments ---
    bar = det._find_dist_bar(img)
    if bar is None:
        print(f"screen {n}: bar NOT FOUND")
        continue
    bx1, by1, bx2, by2 = bar

    # Draw bar bounding box
    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 3)

    segments = det._segment_dist_bar(img, bx1, by1, bx2, by2)
    icon_y1  = max(0, by1 - RANKED_DIST_ICON_SEARCH_PX)
    icon_y2  = by1

    # Width-proportion counts
    total_w = sum(s["width"] for s in segments)
    raw     = [RANKED_SQUAD_TOTAL * s["width"] / total_w for s in segments]

    COLOURS = [(255,100,0),(0,200,255),(0,80,255),(0,255,180),(255,0,200)]
    for i, seg in enumerate(segments):
        col = COLOURS[i % len(COLOURS)]

        # Segment vertical line
        cx, x1, x2 = seg["cx"], seg["x1"], seg["x2"]
        cv2.rectangle(annotated, (x1, by1), (x2, by2), col, 2)
        cv2.line(annotated, (cx, by1), (cx, by2), (255,255,255), 1)

        # Icon search box
        half_w = min(icon_y2 - icon_y1, 100)
        ix1    = max(0, cx - half_w)
        ix2    = min(w,  cx + half_w)
        cv2.rectangle(annotated, (ix1, icon_y1), (ix2, icon_y2), col, 2)

        # Icon match
        icon_rank = det._match_dist_icon(img, cx, icon_y1, icon_y2)
        seg["rank"]     = icon_rank if icon_rank else seg["rank_colour"]
        seg["count_ocr"] = det._ocr_dist_count(img, x1, x2, by1, by2)

        # Label
        label = (f"{seg['rank_colour'][:3]}"
                 f"|icon:{(icon_rank or '?')[:3]}"
                 f"|ocr:{seg['count_ocr']}"
                 f"|w:{raw[i]:.1f}")
        cv2.putText(annotated, label, (x1 + 4, by1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    result   = det._reconcile_dist_counts(segments)
    expected = info["expected"]
    match    = result == expected

    out_path = f"/tmp/squads_debug_{n}.png"
    cv2.imwrite(out_path, annotated)

    print(f"\nscreen {n}: {'PASS ✓' if match else 'FAIL ✗'}")
    print(f"  result  : {result}")
    print(f"  expected: {expected}")
    print(f"  saved   : {out_path}")
    print(f"  segments: {[(s['rank_colour'], s['width'], s['count_ocr']) for s in segments]}")

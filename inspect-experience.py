"""
Inspect mode for ExperienceDetector.
Usage:  python3 inspect-experience.py <image_path>

Runs detection on the given image, draws:
  - Raw OCR hit bounding boxes (green=high conf, yellow=medium, red=low)
  - Column boundary guides (vertical lines per data column)
  - Row position guides (horizontal lines + tolerance bands)
Saves annotated image to tests/inspect_<name>.png and prints the full result.
"""

import os
import sys
import cv2
import easyocr
import numpy as np

from detectors import ScreenClassifier, ExperienceDetector
from config import (
    EXPERIENCE_COMBAT_COUNT_X, EXPERIENCE_COMBAT_RP_X,
    EXPERIENCE_BONUSES_COUNT_X, EXPERIENCE_BONUSES_RP_X,
    EXPERIENCE_PLACEMENT_TEXT_X, EXPERIENCE_PLACEMENT_RP_X,
    EXPERIENCE_BIG_ROW_GAP, EXPERIENCE_SMALL_ROW_GAP, EXPERIENCE_ROW_HALF_H,
    EXPERIENCE_NEXT_RANK_TOP, EXPERIENCE_NEXT_RANK_BOT,
    EXPERIENCE_POINT_CHANGE_TOP, EXPERIENCE_POINT_CHANGE_BOT,
    EXPERIENCE_POINT_CHANGE_X1, EXPERIENCE_POINT_CHANGE_X2,
    EXPERIENCE_CURRENT_RP_TOP, EXPERIENCE_CURRENT_RP_BOT,
    EXPERIENCE_CURRENT_RP_X1, EXPERIENCE_CURRENT_RP_X2,
)

if len(sys.argv) < 2:
    print("Usage: python3 inspect-experience.py <image_path>")
    sys.exit(1)

path = sys.argv[1]
img = cv2.imread(path)
if img is None:
    print(f"Cannot read: {path}")
    sys.exit(1)

sh, sw = img.shape[:2]
name = path.split("/")[-1].replace(".png", "")
print(f"Image: {name}  ({sw}×{sh})")

print("Initialising EasyOCR reader...")
reader     = easyocr.Reader(["en"], gpu=False)
classifier = ScreenClassifier()
detector   = ExperienceDetector(reader)

screen_type, bar_bottom_y = classifier.classify(img)
print(f"classifier → {screen_type.value}  bar_bottom_y={bar_bottom_y}")

# ── Raw OCR on sections band (keep full bboxes for drawing) ──────────────────
y0_band = int(sh * 0.45)
y1_band = int(sh * 0.80)
print("Running sections band OCR...")
raw_hits = reader.readtext(img[y0_band:y1_band, :])
print(f"  {len(raw_hits)} hits")

# ── Full detection ────────────────────────────────────────────────────────────
result = detector.detect(img, bar_bottom_y)

# ── Derive row positions from combat header ───────────────────────────────────
big_gap   = int(sh * EXPERIENCE_BIG_ROW_GAP)
small_gap = int(sh * EXPERIENCE_SMALL_ROW_GAP)
y_tol     = int(sh * EXPERIENCE_ROW_HALF_H)

# Re-derive header_y the same way the detector does
x_cutoff = sw * 0.40
header_y = None
for bbox, text, conf in raw_hits:
    cx = (bbox[0][0] + bbox[2][0]) / 2
    cy = (bbox[0][1] + bbox[2][1]) / 2 + y0_band
    if conf > 0.50 and text.upper().strip() == "COMBAT" and cx < x_cutoff:
        header_y = int(cy)
        break
if header_y is None:
    header_y = int(sh * 0.540)

totals_y = header_y + big_gap
row_a_y  = totals_y + big_gap
row_b_y  = row_a_y  + small_gap
row_c_y  = row_b_y  + small_gap
row_d_y  = row_c_y  + small_gap

# ── Draw ─────────────────────────────────────────────────────────────────────
vis = img.copy()

# -- Column boundary guides --
COLUMNS = [
    ("combat count",    EXPERIENCE_COMBAT_COUNT_X,    (255, 130,  80)),
    ("combat RP",       EXPERIENCE_COMBAT_RP_X,       (255, 200,   0)),
    ("bonuses count",   EXPERIENCE_BONUSES_COUNT_X,   ( 80, 220,  80)),
    ("bonuses RP",      EXPERIENCE_BONUSES_RP_X,      (  0, 180,  80)),
    ("placement text",  EXPERIENCE_PLACEMENT_TEXT_X,  (220,  80, 255)),
    ("placement RP",    EXPERIENCE_PLACEMENT_RP_X,    (140,   0, 220)),
]
for label, (x1f, x2f), color in COLUMNS:
    x1, x2 = int(sw * x1f), int(sw * x2f)
    cv2.line(vis, (x1, 0), (x1, sh), color, 1)
    cv2.line(vis, (x2, 0), (x2, sh), color, 1)
    cv2.putText(vis, label, (x1 + 4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

# -- Row guides (horizontal lines + tolerance bands) --
ROW_LABELS = [
    (header_y, "COMBAT header",  (100, 200, 255)),
    (totals_y, "totals",         (200, 200, 255)),
    (row_a_y,  "row_a",          (180, 180, 255)),
    (row_b_y,  "row_b",          (160, 160, 255)),
    (row_c_y,  "row_c",          (140, 140, 255)),
    (row_d_y,  "row_d",          (120, 120, 255)),
]
for ry, label, color in ROW_LABELS:
    # tolerance band (semi-transparent)
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, ry - y_tol), (sw, ry + y_tol), color, -1)
    cv2.addWeighted(overlay, 0.12, vis, 0.88, 0, vis)
    # centre line
    cv2.line(vis, (0, ry), (sw, ry), color, 1)
    cv2.putText(vis, label, (4, ry - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

# -- Other region boxes (rank strip, point_change, current_rp) --
content_h = sh - bar_bottom_y
REGIONS = [
    ("rank strip",    bar_bottom_y + int(content_h * EXPERIENCE_NEXT_RANK_TOP),
                      bar_bottom_y + int(content_h * EXPERIENCE_NEXT_RANK_BOT),
                      0, sw,                                                    (0, 220, 220)),
    ("point_change",  bar_bottom_y + int(content_h * EXPERIENCE_POINT_CHANGE_TOP),
                      bar_bottom_y + int(content_h * EXPERIENCE_POINT_CHANGE_BOT),
                      int(sw * EXPERIENCE_POINT_CHANGE_X1),
                      int(sw * EXPERIENCE_POINT_CHANGE_X2),                    (0, 180, 255)),
    ("current_rp",    bar_bottom_y + int(content_h * EXPERIENCE_CURRENT_RP_TOP),
                      bar_bottom_y + int(content_h * EXPERIENCE_CURRENT_RP_BOT),
                      int(sw * EXPERIENCE_CURRENT_RP_X1),
                      int(sw * EXPERIENCE_CURRENT_RP_X2),                      (0, 120, 255)),
]
for label, y1, y2, x1, x2, color in REGIONS:
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
    cv2.putText(vis, label, (x1 + 4, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

# -- Raw OCR hits (bounding boxes coloured by confidence) --
for bbox, text, conf in raw_hits:
    tl, tr, br, bl = bbox
    pts = np.array(
        [(int(x), int(y) + y0_band) for x, y in [tl, tr, br, bl]], dtype=np.int32
    )
    if conf >= 0.70:
        color = (0, 255, 0)
    elif conf >= 0.40:
        color = (0, 200, 255)
    else:
        color = (0, 0, 255)
    cv2.polylines(vis, [pts], True, color, 2)
    cv2.putText(vis, f"{text} {conf:.2f}", (pts[0][0], pts[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

# -- Legend --
legend = [
    ("OCR conf ≥0.70", (0, 255,   0)),
    ("OCR conf ≥0.40", (0, 200, 255)),
    ("OCR conf <0.40", (0,   0, 255)),
]
for i, (label, color) in enumerate(legend):
    cv2.putText(vis, label, (sw - 220, sh - 60 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("tests", exist_ok=True)
out_path = os.path.join("tests", f"inspect_{name}.png")
cv2.imwrite(out_path, vis)
print(f"\nAnnotated image → {out_path}")

# ── Print full structured result ─────────────────────────────────────────────
print(f"\nDetection result ({screen_type.value}):")
for k, v in result.items():
    if k == "sanity_issues":
        for issue in (v or []):
            print(f"  ⚠  {issue}")
    else:
        print(f"  {k:<30} {v!r}")

print(f"\nRaw OCR batch ({len(raw_hits)} hits in sections band y={y0_band}–{y1_band}):")
for bbox, text, conf in sorted(raw_hits, key=lambda h: h[0][0][1]):
    cy = int((bbox[0][1] + bbox[2][1]) / 2) + y0_band
    cx = int((bbox[0][0] + bbox[2][0]) / 2)
    print(f"  cy={cy:4d}  cx={cx:4d}  conf={conf:.2f}  {text!r}")

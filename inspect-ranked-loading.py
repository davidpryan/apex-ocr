"""
Ranked-loading screen inspector.
Usage:  python3 inspect-ranked-loading.py <image_path>

Runs ScreenClassifier, RankedLoadingDetector (map name + squad distribution)
and saves an annotated image to tests/.
"""

import os
import sys
import cv2
import easyocr
import numpy as np

from detectors import ScreenClassifier, RankedLoadingDetector
from config import (
    RANKED_LOADING_MAP_Y1, RANKED_LOADING_MAP_Y2,
    RANKED_LOADING_MAP_X1, RANKED_LOADING_MAP_X2,
    RANKED_DIST_ICON_SEARCH_PX, RANKED_SQUAD_TOTAL,
)

if len(sys.argv) < 2:
    print("Usage: python3 inspect-ranked-loading.py <image_path>")
    sys.exit(1)

path = sys.argv[1]
img  = cv2.imread(path)
if img is None:
    print(f"Cannot read: {path}")
    sys.exit(1)

h, w = img.shape[:2]
name = os.path.splitext(os.path.basename(path))[0]
print(f"Image: {name}  ({w}×{h})")

# ── Initialise ────────────────────────────────────────────────────────────────
print("Initialising EasyOCR reader…")
reader     = easyocr.Reader(["en"], gpu=False)
classifier = ScreenClassifier()
det        = RankedLoadingDetector(reader)

screen_type, _ = classifier.classify(img)
print(f"classifier → {screen_type.value}")

# ── Run detectors ─────────────────────────────────────────────────────────────
print("Running detectors…")
map_name = det.detect_map_name(img)
dist     = det.detect_squad_distribution(img)

print(f"\nMap name  : {map_name!r}")
print(f"Squad dist: {dist}")

# ── Internals for annotation ──────────────────────────────────────────────────
bar      = det._find_dist_bar(img)
segments = det._segment_dist_bar(img, *bar) if bar else []

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
# Each rank → its visual colour on screen converted to BGR annotation colour.
RANK_COLOURS: dict[str, tuple] = {
    "Apex Predator": (50,  50,  220),   # red
    "Master":        (200, 50,  190),   # purple
    "Diamond":       (220, 150,  40),   # blue
    "Platinum":      (200, 200,  50),   # teal/cyan
    "Gold":          (0,   180, 240),   # gold
    "Silver":        (190, 190, 190),   # grey
    "Bronze":        (30,  90,  180),   # bronze
}
COL_MAP    = (80,  220, 130)   # map name region
COL_BAR    = (60,  220, 60)    # bar outline
COL_ICON   = (180, 180, 60)    # icon search region

FONT  = cv2.FONT_HERSHEY_SIMPLEX
SCALE = min(w, h) / 1800
THICK = max(1, int(SCALE * 2))
THIN  = max(1, THICK - 1)


def label(vis, text, x, y, col, bg=(20, 20, 20)):
    (tw, th), base = cv2.getTextSize(text, FONT, SCALE, THICK)
    cv2.rectangle(vis, (x, y - th - base - 4), (x + tw + 6, y + base), bg, -1)
    cv2.putText(vis, text, (x + 3, y - base), FONT, SCALE, col, THICK, cv2.LINE_AA)


# ── Annotate ──────────────────────────────────────────────────────────────────
vis = img.copy()

# Map-name ROI
mx1 = int(w * RANKED_LOADING_MAP_X1)
mx2 = int(w * RANKED_LOADING_MAP_X2)
my1 = int(h * RANKED_LOADING_MAP_Y1)
my2 = int(h * RANKED_LOADING_MAP_Y2)
cv2.rectangle(vis, (mx1, my1), (mx2, my2), COL_MAP, THIN)
label(vis, "MAP NAME", mx1, my1 - 4, COL_MAP)

# Bar bounding box
if bar:
    bx1, by1, bx2, by2 = bar
    cv2.rectangle(vis, (bx1, by1), (bx2, by2), COL_BAR, THIN)
    label(vis, "SQUAD RANK DISTRIBUTION", bx1, by1 - 4, COL_BAR)

    icon_y1 = max(0, by1 - RANKED_DIST_ICON_SEARCH_PX)
    icon_y2 = by1

    for seg in segments:
        rank   = seg["rank"]
        col    = RANK_COLOURS.get(rank, (200, 200, 200))
        sx1, sx2, cx = seg["x1"], seg["x2"], seg["cx"]

        # Segment fill outline
        cv2.rectangle(vis, (sx1, by1), (sx2, by2), col, THICK)
        # Centre line
        cv2.line(vis, (cx, by1), (cx, by2), (255, 255, 255), 1)

        # Icon search box (centred at cx)
        half_w = min(icon_y2 - icon_y1, 100)
        ix1    = max(0, cx - half_w)
        ix2    = min(w,  cx + half_w)
        cv2.rectangle(vis, (ix1, icon_y1), (ix2, icon_y2), COL_ICON, THIN)

        # Label above segment: rank name + count
        count = (dist or {}).get(rank, "?")
        lbl   = f"{rank}  {count}"
        label(vis, lbl, sx1, by1 - 4, col)

# ── Legend panel ─────────────────────────────────────────────────────────────
def fmt(v):
    return str(v) if v is not None else "n/a"

dist_rows = []
if dist:
    total = sum(dist.values())
    for rank, count in dist.items():
        dist_rows.append((f"  {rank}", RANK_COLOURS.get(rank, (200,200,200)), str(count)))

legend = [
    ("SCREEN TYPE", None),
    ("  " + screen_type.value, (200, 200, 200), ""),
    ("MAP NAME", None),
    ("  " + fmt(map_name), COL_MAP, ""),
    (f"SQUAD DIST  (total: {sum(dist.values()) if dist else '—'})", None),
    *dist_rows,
]

L_SCALE = SCALE * 0.85
L_THICK = max(1, int(L_SCALE * 2))
line_h  = int(38 * L_SCALE)
pad     = int(16 * L_SCALE)
lx, ly  = int(w * 0.015), int(h * 0.30)
box_w   = int(w * 0.24)
box_h   = pad * 2 + line_h * len(legend)

overlay = vis.copy()
cv2.rectangle(overlay, (lx, ly), (lx + box_w, ly + box_h), (15, 15, 15), -1)
cv2.addWeighted(overlay, 0.72, vis, 0.28, 0, vis)
cv2.rectangle(vis, (lx, ly), (lx + box_w, ly + box_h), (90, 90, 90), 1)

for i, row in enumerate(legend):
    y = ly + pad + (i + 1) * line_h
    if row[1] is None:
        cv2.putText(vis, row[0], (lx + pad, y), FONT, L_SCALE * 0.95,
                    (255, 255, 255), L_THICK, cv2.LINE_AA)
    else:
        row_name, col, value = row
        cv2.putText(vis, row_name, (lx + pad, y), FONT, L_SCALE,
                    col, L_THICK, cv2.LINE_AA)
        if value:
            (vw, _), _ = cv2.getTextSize(value, FONT, L_SCALE, L_THICK)
            cv2.putText(vis, value, (lx + box_w - pad - vw, y),
                        FONT, L_SCALE, col, L_THICK, cv2.LINE_AA)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("tests", exist_ok=True)
out_path = os.path.join("tests", f"inspect_{name}.png")
cv2.imwrite(out_path, vis)
print(f"\nAnnotated image → {out_path}")

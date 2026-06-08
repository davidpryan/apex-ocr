"""
Full game-screen inspector.
Usage:  python3 inspect-game.py <image_path>

Runs every game-screen detector and saves an annotated image to tests/.
"""

import os
import sys
import cv2
import easyocr
import numpy as np

from detectors import (
    ScreenClassifier, ShieldDetector, WeaponDetector,
    ArmorDetector, TopRightDetector,
)
from config import (
    WEAPON_LEFT_FRAC, WEAPON_TOP_FRAC, WEAPON_WIDTH_FRAC, WEAPON_HEIGHT_FRAC,
    ARMOR_LEFT_FRAC,  ARMOR_TOP_FRAC,  ARMOR_WIDTH_FRAC,  ARMOR_HEIGHT_FRAC,
    TR_LEFT_FRAC, TR_TOP_FRAC, TR_WIDTH_FRAC, TR_HEIGHT_FRAC,
    TR_BADGE_CUTOFF_FRAC,
    TR_SQUADS_ROW_TOP, TR_SQUADS_ROW_BOT,
    TR_STATS_ROW_TOP,  TR_STATS_ROW_BOT,
    SHIELD_BAR_X1, SHIELD_BAR_X2,
    SHIELD_STRIP_Y1, SHIELD_STRIP_Y2, SHIELD_BAR_MID_Y,
)

if len(sys.argv) < 2:
    print("Usage: python3 inspect-game.py <image_path>")
    sys.exit(1)

path = sys.argv[1]
img = cv2.imread(path)
if img is None:
    print(f"Cannot read: {path}")
    sys.exit(1)

h, w = img.shape[:2]
name = os.path.splitext(os.path.basename(path))[0]
print(f"Image: {name}  ({w}×{h})")

# ── Initialise detectors ──────────────────────────────────────────────────────
print("Initialising EasyOCR reader...")
reader     = easyocr.Reader(["en"], gpu=False)
classifier = ScreenClassifier()
shield_det = ShieldDetector()
weapon_det = WeaponDetector(reader)
armor_det  = ArmorDetector(reader)
tr_det     = TopRightDetector(reader)

screen_type, bar_bottom_y = classifier.classify(img)
print(f"classifier → {screen_type.value}")

# ── Crop panels ───────────────────────────────────────────────────────────────
def crop(left_f, top_f, width_f, height_f):
    x1 = int(w * left_f)
    y1 = int(h * top_f)
    x2 = int(w * (left_f + width_f))
    y2 = int(h * (top_f + height_f))
    return img[y1:y2, x1:x2], (x1, y1)

weapon_crop, (wx, wy) = crop(WEAPON_LEFT_FRAC, WEAPON_TOP_FRAC,
                              WEAPON_WIDTH_FRAC, WEAPON_HEIGHT_FRAC)
armor_crop,  (ax, ay) = crop(ARMOR_LEFT_FRAC,  ARMOR_TOP_FRAC,
                              ARMOR_WIDTH_FRAC,  ARMOR_HEIGHT_FRAC)
tr_crop,     (tx, ty) = crop(TR_LEFT_FRAC, TR_TOP_FRAC,
                              TR_WIDTH_FRAC, TR_HEIGHT_FRAC)

# ── Run detectors ─────────────────────────────────────────────────────────────
print("Running detectors...")
shield_result = shield_det.detect(img)
weapon_result = weapon_det.detect(weapon_crop)
armor_result  = armor_det.detect(armor_crop)
tr_result     = tr_det.detect(tr_crop)

# ── Print results ─────────────────────────────────────────────────────────────
print("\nShield:")
for k, v in (shield_result or {}).items():
    print(f"  {k:<16} {v!r}")

print("\nWeapons:")
for slot in ("primary", "secondary"):
    val = weapon_result.get(slot)
    print(f"  {slot:<12} {val[0] if val else None!r}")

print(f"\nArmor:  {armor_result.get('number')!r}")

print("\nTop-right:")
for k, v in tr_result.items():
    print(f"  {k:<20} {v[0] if v else None!r}")

# ── Annotate image ────────────────────────────────────────────────────────────
vis = img.copy()

FONT     = cv2.FONT_HERSHEY_SIMPLEX
SCALE    = min(w, h) / 1800       # scale labels to screen size
THICK    = max(1, int(SCALE * 2))
THIN     = max(1, THICK - 1)


def label(text, x, y, col, bg=(20, 20, 20)):
    (tw, th), base = cv2.getTextSize(text, FONT, SCALE, THICK)
    cv2.rectangle(vis, (x, y - th - base - 4), (x + tw + 6, y + base), bg, -1)
    cv2.putText(vis, text, (x + 3, y - base), FONT, SCALE, col, THICK, cv2.LINE_AA)


def panel_rect(left_f, top_f, width_f, height_f, col, title):
    x1, y1 = int(w * left_f), int(h * top_f)
    x2, y2 = int(w * (left_f + width_f)), int(h * (top_f + height_f))
    cv2.rectangle(vis, (x1, y1), (x2, y2), col, THIN)
    label(title, x1, y1 - 4, col)


def draw_box(box, ox, oy, col):
    """Draw only the detection rectangle (no inline text — labels go in the legend)."""
    if box is None:
        return
    bx1, by1, bx2, by2 = box
    cv2.rectangle(vis, (ox + bx1, oy + by1), (ox + bx2, oy + by2), col, THICK)


# Colour map shared between boxes and the legend so they read together.
COL_WEAPON_P = (60, 200, 255)
COL_WEAPON_S = (60, 140, 255)
COL_ARMOR    = (0, 200, 180)
COL_HEALTH   = (80, 220, 80)
SHIELD_COL   = {"white": (230, 230, 230), "blue": (220, 140, 40),
                "purple": (190, 50, 200), "none": (80, 80, 80)}
TR_COLOURS = {
    "squads_remaining":  (255, 200,  80),
    "players_remaining": (255, 160,  60),
    "kills":             (100, 255, 100),
    "assists":           ( 80, 220, 100),
    "participation":     ( 60, 180, 100),
    "damage":            (255, 120, 120),
}
s_col = SHIELD_COL.get((shield_result or {}).get("shield_type", "none"), (80, 80, 80))

# ── Shield / health bars ──────────────────────────────────────────────────────
sx1, sx2 = int(w * SHIELD_BAR_X1), int(w * SHIELD_BAR_X2)
shield_y1, shield_y2 = int(h * SHIELD_STRIP_Y1), int(h * SHIELD_BAR_MID_Y)
health_y1, health_y2 = int(h * SHIELD_BAR_MID_Y), int(h * SHIELD_STRIP_Y2)
cv2.rectangle(vis, (sx1, shield_y1), (sx2, shield_y2), s_col, THICK)
cv2.rectangle(vis, (sx1, health_y1), (sx2, health_y2), COL_HEALTH, THICK)

# ── Weapon bar ────────────────────────────────────────────────────────────────
panel_rect(WEAPON_LEFT_FRAC, WEAPON_TOP_FRAC,
           WEAPON_WIDTH_FRAC, WEAPON_HEIGHT_FRAC, COL_WEAPON_P, "WEAPONS")
for slot, col in [("primary", COL_WEAPON_P), ("secondary", COL_WEAPON_S)]:
    val = weapon_result.get(slot)
    if val:
        draw_box(val[1], wx, wy, col)

# ── Armor triangle ────────────────────────────────────────────────────────────
panel_rect(ARMOR_LEFT_FRAC, ARMOR_TOP_FRAC,
           ARMOR_WIDTH_FRAC, ARMOR_HEIGHT_FRAC, COL_ARMOR, "ARMOR")
if armor_result.get("box"):
    draw_box(armor_result["box"], ax, ay, COL_ARMOR)

# ── Top-right HUD ─────────────────────────────────────────────────────────────
panel_rect(TR_LEFT_FRAC, TR_TOP_FRAC,
           TR_WIDTH_FRAC, TR_HEIGHT_FRAC, (255, 180, 60), "TOP-RIGHT")
for key, val in tr_result.items():
    if val and val[1]:
        draw_box(val[1], tx, ty, TR_COLOURS.get(key, (200, 200, 200)))

# ── Consolidated legend ───────────────────────────────────────────────────────
# Every value is listed once in a single panel so nothing overlaps on the HUD.
def fmt(v):
    return v[0] if v else "n/a"

legend = [
    ("WEAPONS", None),
    ("  primary",   COL_WEAPON_P, fmt(weapon_result.get("primary"))),
    ("  secondary", COL_WEAPON_S, fmt(weapon_result.get("secondary"))),
    ("ARMOR", None),
    ("  level", COL_ARMOR, armor_result.get("number") or "n/a"),
    ("SHIELD / HEALTH", None),
    ("  shield", s_col,      f"{shield_result['shield_type']} {shield_result['shield_hp']}HP" if shield_result else "—"),
    ("  flesh",  COL_HEALTH, f"{shield_result['flesh_hp']}HP" if shield_result else "—"),
    ("  health", (210, 210, 210), f"{shield_result['health']}HP" if shield_result else "—"),
    ("TOP-RIGHT", None),
    ("  squads",        TR_COLOURS["squads_remaining"],  fmt(tr_result.get("squads_remaining"))),
    ("  players",       TR_COLOURS["players_remaining"], fmt(tr_result.get("players_remaining"))),
    ("  kills",         TR_COLOURS["kills"],             fmt(tr_result.get("kills"))),
    ("  assists",       TR_COLOURS["assists"],           fmt(tr_result.get("assists"))),
    ("  participation", TR_COLOURS["participation"],     fmt(tr_result.get("participation"))),
    ("  damage",        TR_COLOURS["damage"],            fmt(tr_result.get("damage"))),
]

L_SCALE = SCALE * 0.85
L_THICK = max(1, int(L_SCALE * 2))
line_h  = int(38 * L_SCALE)
pad     = int(16 * L_SCALE)
lx, ly  = int(w * 0.015), int(h * 0.30)
box_w   = int(w * 0.22)
box_h   = pad * 2 + line_h * len(legend)

overlay = vis.copy()
cv2.rectangle(overlay, (lx, ly), (lx + box_w, ly + box_h), (15, 15, 15), -1)
cv2.addWeighted(overlay, 0.72, vis, 0.28, 0, vis)
cv2.rectangle(vis, (lx, ly), (lx + box_w, ly + box_h), (90, 90, 90), 1)

for i, row in enumerate(legend):
    y = ly + pad + (i + 1) * line_h
    if row[1] is None:            # section header
        cv2.putText(vis, row[0], (lx + pad, y), FONT, L_SCALE * 0.95,
                    (255, 255, 255), L_THICK, cv2.LINE_AA)
    else:
        row_name, col, value = row
        cv2.putText(vis, row_name, (lx + pad, y), FONT, L_SCALE, col, L_THICK, cv2.LINE_AA)
        (vw, _), _ = cv2.getTextSize(str(value), FONT, L_SCALE, L_THICK)
        cv2.putText(vis, str(value), (lx + box_w - pad - vw, y),
                    FONT, L_SCALE, col, L_THICK, cv2.LINE_AA)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("tests", exist_ok=True)
out_path = os.path.join("tests", f"inspect_{name}.png")
cv2.imwrite(out_path, vis)
print(f"\nAnnotated image → {out_path}")

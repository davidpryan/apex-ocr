"""
Benchmark + correctness check for ExperienceDetector.
Run with:  python3 test-experience-detection.py
"""

import time
import cv2
import easyocr
import numpy as np

from detectors import ScreenClassifier, ExperienceDetector
from config import (
    EXPERIENCE_SECTIONS_SCAN_TOP,
    EXPERIENCE_NEXT_RANK_TOP, EXPERIENCE_NEXT_RANK_BOT,
    EXPERIENCE_POINT_CHANGE_TOP, EXPERIENCE_POINT_CHANGE_BOT,
    EXPERIENCE_POINT_CHANGE_X1, EXPERIENCE_POINT_CHANGE_X2,
    EXPERIENCE_CURRENT_RP_TOP, EXPERIENCE_CURRENT_RP_BOT,
    EXPERIENCE_CURRENT_RP_X1, EXPERIENCE_CURRENT_RP_X2,
)

SCREENS = [
    "images/ui-images/experience-screen.png",
    "images/ui-images/exp-screen-3.png",
]

EXPECTED = {
    "experience-screen.png": {
        "current_rank":      "Platinum IV",
        "next_rank":         "Platinum III",
        "point_change":      "+363",
        "current_rp":        "186",
        "combat_rp_total":   "165",
        "kills":             "6",
        "kills_rp":          "120",
        "assists":           "1",
        "assists_rp":        "20",
        "base_combat_value": "20",
        "participations_rp": "30",
        "bonus_rp_total":    "286",
        "challenger_count":  "4",
        "challenger_rp":     "16",
        "top5_streak":       "3/5",
        "top5_streak_rp":    "20",
        "promotion_rp":      "250",
        "placement_rp_total":"62",
        "placement":         "#2",
        "placement_rp":      "100",
        "cost_of_entry_rp":  "-38",
    },
    "exp-screen-2.png": {
        "current_rank":      "Platinum IV",
        "next_rank":         "Platinum III",
        "point_change":      "+246",
        "current_rp":        "509",
        "combat_rp_total":   "188",
        "kills":             "12",
        "kills_rp":          "216",
        "assists":           "0",
        "assists_rp":        "0",
        "participations_formula": "2×50%−1.0",
        "participations_rp": "18",
        "placement":         "#3",
        "placement_rp":      "75",
        "cost_of_entry_tier":"Platinum IV",
        "cost_of_entry_rp":  "-48",
    },
    "exp-screen-3.png": {
        "current_rank":             "Platinum IV",
        "next_rank":                "Platinum III",
        "point_change":             "+15",
        "current_rp":               "0",
        "combat_rp_total":          "8",
        "kills":                    "0",
        "kills_rp":                 "0",
        "assists":                  "0",
        "assists_rp":               "0",
        "participations_rp":        "8",
        "kill_cap_adjustment_rp":   "0",
        "bonus_rp_total":           "0",
        "challenger_count":         "0",
        "challenger_rp":            "0",
        "top5_streak_rp":           "0",
        "promotion_rp":             "0",
        "placement_rp_total":       "7",
        "placement":                "#4",
        "placement_rp":             "55",
        "cost_of_entry_tier":       "Platinum IV",
        "cost_of_entry_rp":         "-48",
    },
}

print("Initialising EasyOCR reader...")
reader     = easyocr.Reader(["en"], gpu=False)
classifier = ScreenClassifier()
detector   = ExperienceDetector(reader)

# ─── Per-call timing probe ──────────────────────────────────────────────────

def _timed_readtext(name, img, **kwargs):
    t = time.perf_counter()
    result = reader.readtext(img, **kwargs)
    print(f"      readtext({name}): {time.perf_counter()-t:.2f}s  →  {len(result)} results")
    return result

# ─── Main loop ──────────────────────────────────────────────────────────────

totals = {"first": [], "second": []}

for path in SCREENS:
    img = cv2.imread(path)
    if img is None:
        print(f"\n[SKIP] {path}")
        continue

    name = path.split("/")[-1]
    sh, sw = img.shape[:2]
    print(f"\n{'='*62}")
    print(f"  {name}  ({sw}×{sh})")
    print(f"{'='*62}")

    detector.reset()   # simulate a fresh match / new experience screen
    screen_type, bar_bottom_y = classifier.classify(img)
    print(f"  classifier → {screen_type.value}  bar_bottom_y={bar_bottom_y}")

    # ── Timed breakdown of the 4 OCR calls ──────────────────────────────────
    print("  --- OCR call breakdown ---")

    # 1. sections band
    y0 = int(sh * 0.45); y1 = int(sh * 0.80)
    t = time.perf_counter()
    batch_raw = _timed_readtext("sections band", img[y0:y1, :])
    t_sections = time.perf_counter() - t

    # 2. rank strip
    content_h = sh - bar_bottom_y
    ry1 = bar_bottom_y + int(content_h * EXPERIENCE_NEXT_RANK_TOP)
    ry2 = bar_bottom_y + int(content_h * EXPERIENCE_NEXT_RANK_BOT)
    t = time.perf_counter()
    _timed_readtext("rank strip", img[ry1:ry2, :])
    t_rank = time.perf_counter() - t

    # 3. point change crop
    py1 = bar_bottom_y + int(content_h * EXPERIENCE_POINT_CHANGE_TOP)
    py2 = bar_bottom_y + int(content_h * EXPERIENCE_POINT_CHANGE_BOT)
    px1, px2 = int(sw * EXPERIENCE_POINT_CHANGE_X1), int(sw * EXPERIENCE_POINT_CHANGE_X2)
    import cv2
    from preprocessing import sharpen, enhance_contrast
    pc_crop = cv2.resize(img[py1:py2, px1:px2], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    t = time.perf_counter()
    _timed_readtext("point_change", pc_crop)
    t_pc = time.perf_counter() - t

    # 4. current_rp crop (approximate — just time the readtext portion)
    rp_y1 = bar_bottom_y + int(content_h * EXPERIENCE_CURRENT_RP_TOP)
    rp_y2 = bar_bottom_y + int(content_h * EXPERIENCE_CURRENT_RP_BOT)
    rp_x1, rp_x2 = int(sw * EXPERIENCE_CURRENT_RP_X1), int(sw * EXPERIENCE_CURRENT_RP_X2)
    rp_crop = img[rp_y1:rp_y2, rp_x1:rp_x2]
    t = time.perf_counter()
    _timed_readtext("current_rp", rp_crop, allowlist="0123456789RP ")
    t_rp = time.perf_counter() - t

    # ── Full detect() timing ─────────────────────────────────────────────────
    print(f"  --- full detect() ---")
    t0 = time.perf_counter()
    result = detector.detect(img, bar_bottom_y)
    elapsed = time.perf_counter() - t0
    totals["first"].append(elapsed)
    print(f"  detect()  {elapsed:.2f}s")

    # ── Results ──────────────────────────────────────────────────────────────
    exp = EXPECTED.get(name, {})
    ok = err = 0
    for key, val in result.items():
        if key == "sanity_issues":
            for issue in (val or []):
                print(f"    ⚠ {issue}")
            continue
        if key not in exp:
            continue
        tick = "✓" if str(val) == str(exp[key]) else f"✗ (expected {exp[key]!r})"
        sym  = "✓" if "✓" in tick else "✗"
        if sym == "✓": ok += 1
        else: err += 1
        print(f"    {sym} {key:<28} got={val!r}  {tick}")
    print(f"  score: {ok}/{ok+err} checked fields")

    # Show derived fields (sanity-check filled values not in EXPECTED)
    derived_keys = {"kill_cap_adjustment_rp", "promotion_rp", "placement_rp_total",
                    "bonus_rp_total", "combat_rp_total", "assists_rp"}
    for key in derived_keys:
        if key not in exp and result.get(key) is not None:
            print(f"    → derived {key} = {result[key]!r}")

# ─── Second pass: result_cache hit (same detector instance, same screen) ────
print(f"\n{'='*62}")
print("  Second pass (result cached — same ExperienceDetector instance):")
for path in SCREENS:
    img = cv2.imread(path)
    if img is None:
        continue
    _, bar_bottom_y = classifier.classify(img)
    t0 = time.perf_counter()
    detector.detect(img, bar_bottom_y)
    elapsed = time.perf_counter() - t0
    totals["second"].append(elapsed)
    print(f"    {path.split('/')[-1]:<35} {elapsed*1000:.1f}ms")

# ─── Third pass: after reset() ───────────────────────────────────────────────
print(f"\n{'='*62}")
print("  Third pass (after reset — simulates new match):")
for path in SCREENS:
    img = cv2.imread(path)
    if img is None:
        continue
    detector.reset()
    _, bar_bottom_y = classifier.classify(img)
    t0 = time.perf_counter()
    detector.detect(img, bar_bottom_y)
    elapsed = time.perf_counter() - t0
    print(f"    {path.split('/')[-1]:<35} {elapsed:.2f}s")

print(f"\n  avg first  pass (full OCR):  {sum(totals['first'])/len(totals['first']):.2f}s")
print(f"  avg second pass (cached):    {sum(totals['second'])/len(totals['second'])*1000:.1f}ms")

"""Overlay drawing for the live Apex Detector window."""

import cv2
import numpy as np

from config import (
    TR_BADGE_CUTOFF_FRAC,
    TR_SQUADS_ROW_TOP, TR_SQUADS_ROW_BOT,
    TR_STATS_ROW_TOP,  TR_STATS_ROW_BOT,
)

# Colours per detection category
_WEAPON_COLOURS = {
    "primary":   (0, 220, 255),
    "secondary": (0, 255, 120),
}
_TR_COLOURS = {
    "squads_remaining":  (255, 200, 50),
    "players_remaining": (255, 200, 50),
    "kills":        (80, 200, 255),
    "assists":      (80, 200, 255),
    "participation": (80, 200, 255),
    "damage":       (80, 200, 255),
}
_TR_LABELS = {
    "squads_remaining": "squads",
    "players_remaining": "players",
    "kills": "kills",
    "assists": "assists",
    "participation": "part.",
    "damage": "dmg",
}


def _labelled_box(
    img: np.ndarray, box: tuple, text: str, colour: tuple
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _sub_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, colour: tuple) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1)


def draw_all(
    full_bgr: np.ndarray,
    monitor: dict,
    roi_w: dict,
    roi_a: dict,
    roi_tr: dict,
    weapon_result: dict,
    armor_result: dict,
    tr_result: dict,
    fps: float,
) -> np.ndarray:
    """Draw all detections and search-region outlines onto a copy of full_bgr."""
    out = full_bgr.copy()
    ox, oy = monitor["left"], monitor["top"]

    def shifted(box: tuple, roi: dict) -> tuple:
        x1, y1, x2, y2 = box
        dx, dy = roi["left"] - ox, roi["top"] - oy
        return x1 + dx, y1 + dy, x2 + dx, y2 + dy

    # --- Weapon sub-regions (primary left / secondary right) ---
    wx   = roi_w["left"] - ox
    wy   = roi_w["top"]  - oy
    wmx  = wx + roi_w["width"] // 2
    wy2  = wy + roi_w["height"]
    _sub_box(out, wx,  wy, wmx,                wy2, (0, 140, 180))
    _sub_box(out, wmx, wy, wx + roi_w["width"], wy2, (0, 180,  80))

    # --- Weapon detections ---
    for slot, colour in _WEAPON_COLOURS.items():
        entry = weapon_result.get(slot)
        if entry:
            text, box = entry
            _labelled_box(out, shifted(box, roi_w), f"{slot}: {text}", colour)

    # --- Armor level ---
    if armor_result.get("box"):
        _labelled_box(
            out,
            shifted(armor_result["box"], roi_a),
            f"armor: {armor_result['number']}",
            (0, 220, 255),
        )

    # --- Top-right sub-regions (squads row / stats row, badge excluded) ---
    tx  = roi_tr["left"] - ox
    ty  = roi_tr["top"]  - oy
    tw  = roi_tr["width"]
    th  = roi_tr["height"]
    bx  = tx + int(tw * TR_BADGE_CUTOFF_FRAC)
    _sub_box(out, tx, ty + int(th * TR_SQUADS_ROW_TOP),
             bx, ty + int(th * TR_SQUADS_ROW_BOT), (180, 150, 30))
    _sub_box(out, tx, ty + int(th * TR_STATS_ROW_TOP),
             bx, ty + int(th * TR_STATS_ROW_BOT),  (60,  150, 200))

    # --- Top-right detections ---
    for key, entry in tr_result.items():
        if entry:
            val, box = entry
            label  = _TR_LABELS.get(key, key)
            colour = _TR_COLOURS.get(key, (200, 200, 200))
            _labelled_box(out, shifted(box, roi_tr), f"{label}: {val}", colour)

    cv2.putText(
        out,
        f"FPS: {fps:.0f}   click=pause   R=reset ROIs   Q=quit",
        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA,
    )
    return out

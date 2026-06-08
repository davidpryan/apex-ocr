"""
ROIManager — owns the three screen regions used for detection.

Responsibilities:
  - Build ROI dicts from saved pixel coordinates or fall back to reference-
    image fractions scaled to the live monitor size.
  - Persist / load ROIs to/from rois.json.
  - Run the interactive click-drag configuration UI.
"""

import json
import os

import cv2
import numpy as np

from config import (
    VERTICAL_BUFFER_PX,
    WEAPON_LEFT_FRAC, WEAPON_TOP_FRAC, WEAPON_WIDTH_FRAC, WEAPON_HEIGHT_FRAC,
    ARMOR_LEFT_FRAC,  ARMOR_TOP_FRAC,  ARMOR_WIDTH_FRAC,  ARMOR_HEIGHT_FRAC,
    TR_LEFT_FRAC,     TR_TOP_FRAC,     TR_WIDTH_FRAC,     TR_HEIGHT_FRAC,
)


class ROIManager:
    """Builds, persists, and interactively configures the three capture ROIs."""

    _CATEGORIES = [
        ("weapons",   "WEAPONS",           (0, 255, 120)),
        ("armor",     "ARMOR",             (0, 220, 255)),
        ("top_right", "TOP-RIGHT",         (255, 200,  50)),
        ("game_area", "GAME AREA (auto)",  (180, 180, 180)),
    ]

    def __init__(self, roi_file: str, vertical_buffer: int = VERTICAL_BUFFER_PX):
        self._roi_file = roi_file
        self._vertical_buffer = vertical_buffer
        self._saved: dict = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        if os.path.exists(self._roi_file):
            with open(self._roi_file) as f:
                self._saved = json.load(f)

    def save(self, rois: dict) -> None:
        with open(self._roi_file, "w") as f:
            json.dump(rois, f, indent=2)
        self._saved.update(rois)
        print(f"ROIs saved → {self._roi_file}")

    def reset(self) -> None:
        if os.path.exists(self._roi_file):
            os.remove(self._roi_file)
        self._saved.clear()
        print("ROIs reset to defaults")

    # ------------------------------------------------------------------
    # ROI builders
    # ------------------------------------------------------------------

    def _game_area(self, monitor: dict) -> tuple[int, int, int, int]:
        """(origin_x, origin_y, width, height) with vertical buffer applied."""
        return (
            monitor["left"],
            monitor["top"] + self._vertical_buffer,
            monitor["width"],
            monitor["height"] - 2 * self._vertical_buffer,
        )

    def _from_fractions(
        self, key: str, monitor: dict,
        lf: float, tf: float, wf: float, hf: float,
    ) -> dict:
        saved = self._saved.get(key)
        if saved:
            return {
                "left":   monitor["left"] + saved["left"],
                "top":    monitor["top"]  + saved["top"],
                "width":  saved["width"],
                "height": saved["height"],
            }
        ox, oy, sw, sh = self._game_area(monitor)
        return {
            "left":   ox + int(sw * lf),
            "top":    oy + int(sh * tf),
            "width":  int(sw * wf),
            "height": int(sh * hf),
        }

    def build_weapon_roi(self, monitor: dict) -> dict:
        return self._from_fractions(
            "weapons", monitor,
            WEAPON_LEFT_FRAC, WEAPON_TOP_FRAC, WEAPON_WIDTH_FRAC, WEAPON_HEIGHT_FRAC,
        )

    def build_armor_roi(self, monitor: dict) -> dict:
        return self._from_fractions(
            "armor", monitor,
            ARMOR_LEFT_FRAC, ARMOR_TOP_FRAC, ARMOR_WIDTH_FRAC, ARMOR_HEIGHT_FRAC,
        )

    def build_topright_roi(self, monitor: dict) -> dict:
        return self._from_fractions(
            "top_right", monitor,
            TR_LEFT_FRAC, TR_TOP_FRAC, TR_WIDTH_FRAC, TR_HEIGHT_FRAC,
        )

    def build_all(self, monitor: dict) -> tuple[dict, dict, dict]:
        """Return (weapon_roi, armor_roi, topright_roi) in one call."""
        return (
            self.build_weapon_roi(monitor),
            self.build_armor_roi(monitor),
            self.build_topright_roi(monitor),
        )

    # ------------------------------------------------------------------
    # Derive ROIs from a drawn game-area rectangle
    # ------------------------------------------------------------------

    @staticmethod
    def derive_from_game_area(game_area: dict) -> dict:
        """Compute all three ROIs from a hand-drawn game-area rectangle using
        the same proportional constants as the 2696×1520 reference screen."""
        gx, gy = game_area["left"], game_area["top"]
        gw, gh = game_area["width"], game_area["height"]
        return {
            "weapons": {
                "left": gx + int(gw * WEAPON_LEFT_FRAC),
                "top":  gy + int(gh * WEAPON_TOP_FRAC),
                "width":  int(gw * WEAPON_WIDTH_FRAC),
                "height": int(gh * WEAPON_HEIGHT_FRAC),
            },
            "armor": {
                "left": gx + int(gw * ARMOR_LEFT_FRAC),
                "top":  gy + int(gh * ARMOR_TOP_FRAC),
                "width":  int(gw * ARMOR_WIDTH_FRAC),
                "height": int(gh * ARMOR_HEIGHT_FRAC),
            },
            "top_right": {
                "left": gx + int(gw * TR_LEFT_FRAC),
                "top":  gy + int(gh * TR_TOP_FRAC),
                "width":  int(gw * TR_WIDTH_FRAC),
                "height": int(gh * TR_HEIGHT_FRAC),
            },
        }

    # ------------------------------------------------------------------
    # Interactive configuration UI
    # ------------------------------------------------------------------

    def configure(self, monitor: dict, sct) -> None:
        """Click-drag editor for all three ROIs (plus a GAME AREA auto-derive).
        Keys: 1/2/3/4 or TAB = select category, S = save, Q = quit."""

        full = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
        mw, mh = full.shape[1], full.shape[0]
        scale = min(1.0, 1400 / mw)

        builders = {
            "weapons":   self.build_weapon_roi,
            "armor":     self.build_armor_roi,
            "top_right": self.build_topright_roi,
        }

        # Seed from saved values or fraction-based defaults (monitor-relative coords)
        rois: dict = {}
        for key, _, _ in self._CATEGORIES:
            if key in self._saved:
                rois[key] = dict(self._saved[key])
            elif key in builders:
                r = builders[key](monitor)
                rois[key] = {
                    "left":   r["left"]   - monitor["left"],
                    "top":    r["top"]    - monitor["top"],
                    "width":  r["width"],
                    "height": r["height"],
                }

        state = {"cat": 0, "drawing": False, "sx": 0, "sy": 0}
        WIN   = "Configure ROIs"

        def render(tmp=None):
            img = full.copy()
            for key, label, color in self._CATEGORIES:
                r = rois.get(key)
                if r:
                    x1, y1 = r["left"], r["top"]
                    x2, y2 = x1 + r["width"], y1 + r["height"]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1 + 4, y1 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            _, act_label, act_color = self._CATEGORIES[state["cat"]]
            cv2.putText(
                img,
                f"Drawing: {act_label}   1=WEAPONS  2=ARMOR  3=TOP-RIGHT"
                f"  4=GAME AREA (auto)  TAB=cycle  S=save  Q=quit",
                (8, mh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, act_color, 2, cv2.LINE_AA,
            )
            disp = cv2.resize(img, None, fx=scale, fy=scale)
            if tmp:
                cv2.rectangle(disp, tmp[:2], tmp[2:], act_color, 2)
            cv2.imshow(WIN, disp)

        def on_mouse(event, x, y, *_):
            fx, fy = int(x / scale), int(y / scale)
            if event == cv2.EVENT_LBUTTONDOWN:
                state["drawing"], state["sx"], state["sy"] = True, fx, fy
            elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
                d = (
                    int(min(state["sx"], fx) * scale), int(min(state["sy"], fy) * scale),
                    int(max(state["sx"], fx) * scale), int(max(state["sy"], fy) * scale),
                )
                render(tmp=d)
            elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
                state["drawing"] = False
                x1, y1 = min(state["sx"], fx), min(state["sy"], fy)
                x2, y2 = max(state["sx"], fx), max(state["sy"], fy)
                if x2 - x1 > 5 and y2 - y1 > 5:
                    key = self._CATEGORIES[state["cat"]][0]
                    rois[key] = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
                    if key == "game_area":
                        rois.update(self.derive_from_game_area(rois["game_area"]))
                    state["cat"] = (state["cat"] + 1) % len(self._CATEGORIES)
                render()

        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, on_mouse)
        render()

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == ord("s"):
                self.save(rois)
                break
            elif k == ord("q"):
                break
            elif k == ord("\t"):
                state["cat"] = (state["cat"] + 1) % len(self._CATEGORIES)
                render()
            elif ord("1") <= k <= ord("4"):
                state["cat"] = k - ord("1")
                render()

        cv2.destroyWindow(WIN)

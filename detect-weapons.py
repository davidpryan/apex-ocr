"""
Apex Legends HUD detector — entry point.

Usage:
  python3 detect-weapons.py                       # live detection
  python3 detect-weapons.py configure=True        # draw ROIs interactively first
  python3 detect-weapons.py debug=True            # show ROI overlay then go live
  python3 detect-weapons.py --replay-interval=10  # write replay row every 10 s
"""

import sys
import threading

import cv2
import easyocr
import mss
import numpy as np

from config import (
    ROI_FILE, REPLAY_INTERVAL_SEC,
    VERTICAL_BUFFER_PX,
)
from detectors import (
    ArmorDetector, ExperienceDetector, GameAggregator, RankedLoadingDetector,
    ScreenClassifier, ShieldDetector, TopRightDetector, WeaponDetector,
)
from engine import DetectorEngine, EngineState
from map_locator import MapLocator
from overlay import OverlayWindow
from roi_manager import ROIManager

from PySide6 import QtWidgets


def _show_debug_overlay(monitor: dict, sct, roi: dict, armor_roi: dict, tr_roi: dict) -> None:
    _, _, _, eff_h = (
        monitor["left"],
        monitor["top"] + VERTICAL_BUFFER_PX,
        monitor["width"],
        monitor["height"] - 2 * VERTICAL_BUFFER_PX,
    )
    print(f"DEBUG  buffer={VERTICAL_BUFFER_PX}px  effective game height={eff_h}px")
    full    = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
    overlay = full.copy()
    ox, oy  = monitor["left"], monitor["top"]
    for r, label, color in [
        (roi,       "WEAPONS",   (0, 255, 120)),
        (armor_roi, "ARMOR",     (0, 220, 255)),
        (tr_roi,    "TOP-RIGHT", (255, 200,  50)),
    ]:
        x1, y1 = r["left"] - ox, r["top"] - oy
        x2, y2 = x1 + r["width"], y1 + r["height"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        cv2.putText(overlay, label, (x1 + 6, y1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    if VERTICAL_BUFFER_PX > 0:
        for buf_y in (VERTICAL_BUFFER_PX, full.shape[0] - VERTICAL_BUFFER_PX):
            cv2.line(overlay, (0, buf_y), (full.shape[1], buf_y), (100, 100, 255), 2)
        cv2.putText(overlay, f"buffer={VERTICAL_BUFFER_PX}px",
                    (8, VERTICAL_BUFFER_PX - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (100, 100, 255), 2, cv2.LINE_AA)
    scale = min(1.0, 1400 / overlay.shape[1])
    cv2.imshow("Debug — ROI overlay (any key to start live)",
               cv2.resize(overlay, None, fx=scale, fy=scale))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    debug     = "debug=True"     in sys.argv
    configure = "configure=True" in sys.argv
    replay_interval = float(
        next((a.split("=")[1] for a in sys.argv if a.startswith("--replay-interval=")),
             REPLAY_INTERVAL_SEC)
    )

    roi_mgr = ROIManager(ROI_FILE, VERTICAL_BUFFER_PX)
    roi_mgr.load()

    # ── Optional cv2 pre-flight (runs before Qt starts) ───────────────────────
    if configure or debug:
        with mss.MSS() as sct:
            monitor = sct.monitors[1]
            if configure:
                roi_mgr.configure(monitor, sct)
            if debug:
                roi, armor_roi, tr_roi = roi_mgr.build_all(monitor)
                _show_debug_overlay(monitor, sct, roi, armor_roi, tr_roi)

    print("Loading EasyOCR model…")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    reader.readtext(np.zeros((10, 10, 3), dtype=np.uint8))   # warm up

    weapon_det = WeaponDetector(reader)
    armor_det  = ArmorDetector(reader)
    tr_det     = TopRightDetector(reader)
    shield_det = ShieldDetector()
    classifier = ScreenClassifier()
    exp_det    = ExperienceDetector(reader)
    ranked_det = RankedLoadingDetector(reader)
    aggregator = GameAggregator()
    map_loc    = MapLocator()

    state  = EngineState()
    engine = DetectorEngine(
        state, weapon_det, armor_det, tr_det, shield_det,
        classifier, exp_det, ranked_det, aggregator, map_loc,
        roi_mgr, replay_interval,
    )

    t = threading.Thread(target=engine.run, daemon=True)
    t.start()

    print("Overlay open. Close the window to quit.\n")

    # Qt must own the main thread on macOS
    app = QtWidgets.QApplication([sys.argv[0]])
    win = OverlayWindow(state)
    win.show()
    app.exec()

    state.stop.set()
    t.join(timeout=2)


if __name__ == "__main__":
    main()

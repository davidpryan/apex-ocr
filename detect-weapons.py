"""
Apex Legends HUD detector — entry point.

Usage:
  python3 detect-weapons.py                       # live detection
  python3 detect-weapons.py configure=True        # draw ROIs interactively first
  python3 detect-weapons.py debug=True            # show ROI overlay then go live
  python3 detect-weapons.py --replay-interval=10  # write replay row every 10 s
"""

import datetime
import sys
import threading
import time

import cv2
import easyocr
import mss
import numpy as np

from config import (
    CAPTURE_PADDING, ROI_FILE, REPLAY_INTERVAL_SEC,
    SCREEN_CLASSIFY_INTERVAL, VERTICAL_BUFFER_PX,
)
from detectors import (
    ArmorDetector, ExperienceDetector, GameAggregator, RankedLoadingDetector,
    ScreenClassifier, ScreenType, ShieldDetector, TopRightDetector, WeaponDetector,
)
from display import draw_all
from map_locator import MapLocator
from preprocessing import expand_roi
from roi_manager import ROIManager
from session import GameSession, MatchPhase, PhaseDebouncer
from sinks import ExperienceWriter, MatchReplayWriter

# Map ScreenClassifier output → MatchPhase (UNKNOWN treated as MENU/lobby)
_PHASE_MAP = {
    ScreenType.GAME:           MatchPhase.GAME,
    ScreenType.EXPERIENCE:     MatchPhase.EXPERIENCE,
    ScreenType.RANKED_LOADING: MatchPhase.RANKED_LOADING,
    ScreenType.UNKNOWN:        MatchPhase.MENU,
}


def _print_on_change(key: str, new_val, last: dict) -> None:
    if new_val != last.get(key):
        last[key] = new_val
        print(f"  {key.upper():20s}: {new_val or '—'}")


def _show_debug_overlay(monitor: dict, sct, roi: dict, armor_roi: dict, tr_roi: dict) -> None:
    from config import VERTICAL_BUFFER_PX
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

    print("Loading EasyOCR model…")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    reader.readtext(np.zeros((10, 10, 3), dtype=np.uint8))   # warm up

    weapon_det = WeaponDetector(reader)
    armor_det  = ArmorDetector(reader)
    tr_det     = TopRightDetector(reader)
    shield_det = ShieldDetector()
    classifier = ScreenClassifier()
    exp_det     = ExperienceDetector(reader)
    ranked_det  = RankedLoadingDetector(reader)
    aggregator  = GameAggregator()
    map_loc     = MapLocator()

    # Session / output sinks
    debouncer    = PhaseDebouncer()
    session      = GameSession()
    exp_writer   = ExperienceWriter()
    replay_writer = MatchReplayWriter()

    print(f"Ready. Press Q to quit. Replay interval: {replay_interval:.0f} s\n")

    # ── Shared state between capture loop and OCR worker ─────────────────────
    latest_frames  = {"weapon": None, "armor": None, "tr": None, "full": None}
    latest_results = {
        "weapon": {"primary": None, "secondary": None},
        "armor":  {"number": None, "box": None},
        "shield": None,
        "tr":     {k: None for k in ("squads_remaining", "players_remaining",
                                     "kills", "assists", "participation", "damage")},
    }
    frame_locks  = {k: threading.Lock() for k in latest_frames}
    result_lock  = threading.Lock()

    def ocr_worker() -> None:
        while True:
            frames = {}
            for k, lock in frame_locks.items():
                with lock:
                    f = latest_frames[k]
                    frames[k] = f.copy() if f is not None else None

            if all(f is None for f in frames.values()):
                time.sleep(0.05)
                continue

            weapon_r = weapon_det.detect(frames["weapon"]) if frames["weapon"] is not None else None
            armor_r  = armor_det.detect(frames["armor"])   if frames["armor"]  is not None else None
            tr_r     = tr_det.detect(frames["tr"])         if frames["tr"]     is not None else None
            shield_r = shield_det.detect(frames["full"])   if frames["full"]   is not None else None

            with result_lock:
                stable = aggregator.update(weapon=weapon_r, armor=armor_r,
                                           shield=shield_r, tr=tr_r)
                latest_results.update(stable)

    threading.Thread(target=ocr_worker, daemon=True).start()

    # ── Helper: experience capture ────────────────────────────────────────────
    def _capture_experience(frame: np.ndarray, bby: int,
                            map_name: str | None = None,
                            squad_dist: dict | None = None) -> None:
        if session.game_id is None:
            return
        exp_result   = exp_det.detect(frame, bby)
        captured_at  = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        exp_writer.write(exp_result, session.game_id,
                         session.game_start_time, captured_at,
                         map_name=map_name, squad_dist=squad_dist)
        session.mark_experience_captured()
        print(f"  [EXP] Captured — game_id={session.game_id}  map={map_name or '?'}")

    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        roi, armor_roi, tr_roi = roi_mgr.build_all(monitor)
        print(f"Weapon ROI:    {roi['width']}x{roi['height']} at ({roi['left']}, {roi['top']})")
        print(f"Armor ROI:     {armor_roi['width']}x{armor_roi['height']} at ({armor_roi['left']}, {armor_roi['top']})")
        print(f"Top-right ROI: {tr_roi['width']}x{tr_roi['height']} at ({tr_roi['left']}, {tr_roi['top']})\n")

        if configure:
            roi_mgr.configure(monitor, sct)
            roi, armor_roi, tr_roi = roi_mgr.build_all(monitor)

        if debug:
            _show_debug_overlay(monitor, sct, roi, armor_roi, tr_roi)

        display_scale    = min(1.0, 1400 / monitor["width"])
        WIN              = "Apex Detector"
        paused           = {"value": False}
        frozen           = [None]

        def on_click(event, *_):
            if event == cv2.EVENT_LBUTTONDOWN:
                paused["value"] = not paused["value"]

        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, on_click)

        last_printed    = {"primary": None, "secondary": None, "armor-level": None,
                           "current_rank": None, "next_rank": None,
                           "point_change": None, "current_rp": None,
                           "shield": None, "health": None}
        last_printed_tr = {k: None for k in ("squads_remaining", "players_remaining",
                                              "kills", "assists", "participation", "damage")}
        fps_time         = time.time()
        frame_count      = 0
        screen_type      = ScreenType.UNKNOWN
        bar_bottom_y     = 0
        classify_counter = 0
        last_replay_mono              = time.monotonic()
        last_exp_frame                = None   # most-recent full frame seen during EXPERIENCE
        _ranked_loading_map_captured  = False  # reset each time RANKED_LOADING is confirmed

        while True:
            if paused["value"]:
                if frozen[0] is not None:
                    frame = frozen[0].copy()
                    h, w  = frame.shape[:2]
                    cv2.rectangle(frame, (0, 0), (w, 38), (0, 0, 0), -1)
                    cv2.putText(frame, "PAUSED — click to resume",
                                (8, 27), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 200, 255), 2, cv2.LINE_AA)
                    cv2.imshow(WIN, frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue

            full_bgr = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
            ox, oy   = monitor["left"], monitor["top"]

            # ── Classify + debounce + session update ─────────────────────────
            classify_counter += 1
            if classify_counter >= SCREEN_CLASSIFY_INTERVAL:
                classify_counter = 0
                new_type, bar_bottom_y = classifier.classify(full_bgr)
                raw_phase      = _PHASE_MAP[new_type]
                confirmed      = debouncer.push(raw_phase)

                if confirmed is not None:
                    if confirmed == MatchPhase.RANKED_LOADING:
                        _ranked_loading_map_captured = False
                    # Save game context BEFORE update(): if a new game is minted
                    # inside update() (EXPERIENCE→GAME direct transition), the
                    # EXPERIENCE_EXIT_UNCAPTURED event must still tag the row with
                    # the old game_id, not the new one.
                    _pre_game_id = session.game_id
                    _pre_game_st = session.game_start_time
                    events = session.update(confirmed)
                    for ev in events:
                        if ev == GameSession.Event.GAME_STARTED:
                            with result_lock:
                                aggregator.reset()
                            exp_det.reset()
                            last_replay_mono = time.monotonic()
                            print(f"  [SESSION] New game — id={session.game_id}  map={session.map_name or '?'}")
                        elif ev == GameSession.Event.EXPERIENCE_EXIT_UNCAPTURED:
                            # Use _pre_game_id — session.game_id may have already
                            # advanced if _start_game() was called in this update().
                            if last_exp_frame is not None and _pre_game_id is not None:
                                print("  [EXP] Best-effort capture on exit")
                                exp_result  = exp_det.detect(last_exp_frame, bar_bottom_y)
                                captured_at = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                                exp_writer.write(exp_result, _pre_game_id,
                                                 _pre_game_st, captured_at,
                                                 map_name=session.map_name,
                                                 squad_dist=session.squad_dist)
                                if session.game_id == _pre_game_id:
                                    # Same game (EXPERIENCE→MENU, no new game started).
                                    session.mark_experience_captured()

                    screen_type = new_type
                    if confirmed != MatchPhase.EXPERIENCE:
                        last_exp_frame = None   # stale after leaving EXPERIENCE

            # ── FPS bookkeeping ───────────────────────────────────────────────
            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps_time, frame_count = time.time(), 0
            fps = frame_count / max(elapsed, 0.001)

            # ── Branch: EXPERIENCE ────────────────────────────────────────────
            if screen_type == ScreenType.EXPERIENCE:
                last_exp_frame = full_bgr   # always keep the latest frame

                # Timed capture: once, 5 s after EXPERIENCE was first seen
                if session.should_capture_experience():
                    _capture_experience(full_bgr, bar_bottom_y,
                                        session.map_name, session.squad_dist)

                # Display — only show rank overlay; don't OCR every frame
                display   = full_bgr.copy()
                rank_text = session.game_id or "…"
                cv2.putText(display,
                            f"EXPERIENCE   game_id={rank_text}   FPS: {fps:.0f}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (200, 200, 255), 2, cv2.LINE_AA)

            # ── Branch: GAME (HUD detectors + replay) ────────────────────────
            elif screen_type == ScreenType.GAME:
                p_roi   = expand_roi(roi,       monitor, CAPTURE_PADDING)
                p_armor = expand_roi(armor_roi, monitor, CAPTURE_PADDING)
                p_tr    = expand_roi(tr_roi,    monitor, CAPTURE_PADDING)

                def crop(r: dict) -> np.ndarray:
                    return full_bgr[
                        r["top"] - oy : r["top"] - oy + r["height"],
                        r["left"] - ox : r["left"] - ox + r["width"],
                    ]

                for key, r in (("weapon", p_roi), ("armor", p_armor), ("tr", p_tr)):
                    with frame_locks[key]:
                        latest_frames[key] = crop(r)
                with frame_locks["full"]:
                    latest_frames["full"] = full_bgr

                with result_lock:
                    weapon_res = dict(latest_results["weapon"])
                    armor_res  = dict(latest_results["armor"])
                    tr_res     = dict(latest_results["tr"])
                    shield_res = latest_results["shield"]
                    agg_state  = dict(latest_results)

                for slot in ("primary", "secondary"):
                    entry = weapon_res.get(slot)
                    _print_on_change(slot, entry[0] if entry else None, last_printed)
                _print_on_change("armor-level", armor_res.get("number"), last_printed)
                if shield_res:
                    _print_on_change("shield",
                                     f"{shield_res['shield_type']} {shield_res['shield_hp']}HP",
                                     last_printed)
                    _print_on_change("health", shield_res["health"], last_printed)
                for key, entry in tr_res.items():
                    _print_on_change(key, entry[0] if entry else None, last_printed_tr)

                # Periodic replay row
                now_mono = time.monotonic()
                if (session.game_id is not None
                        and now_mono - last_replay_mono >= replay_interval):
                    last_replay_mono = now_mono
                    row_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                    map_info = map_loc.locate(full_bgr)
                    replay_writer.write(
                        agg_state,
                        session.game_id,
                        session.game_start_time,
                        row_time,
                        session.elapsed_s,
                        map_name=session.map_name,
                        map_info=map_info,
                    )

                display = draw_all(full_bgr, monitor, p_roi, p_armor, p_tr,
                                   weapon_res, armor_res, tr_res, fps)

            # ── Branch: RANKED_LOADING — OCR map name + squad dist once ────────
            elif screen_type == ScreenType.RANKED_LOADING:
                if not _ranked_loading_map_captured:
                    map_name_ocr = ranked_det.detect_map_name(full_bgr)
                    dist_ocr     = ranked_det.detect_squad_distribution(full_bgr)
                    if map_name_ocr:
                        session.set_map_name(map_name_ocr)
                    if dist_ocr:
                        session.set_squad_distribution(dist_ocr)
                    if map_name_ocr:
                        _ranked_loading_map_captured = True
                        print(f"  [MAP] {map_name_ocr}  dist={dist_ocr}")

                display = full_bgr.copy()
                cv2.putText(display,
                            f"RANKED LOADING  map={session.map_name or '?'}  FPS: {fps:.0f}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (160, 255, 160), 2, cv2.LINE_AA)

            # ── Branch: UNKNOWN / MENU — show plain frame ─────────────────────
            else:
                display = full_bgr.copy()
                cv2.putText(display, f"MENU / UNKNOWN   FPS: {fps:.0f}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (160, 160, 160), 2, cv2.LINE_AA)

            frozen[0] = cv2.resize(display, None, fx=display_scale, fy=display_scale)
            cv2.imshow(WIN, frozen[0])

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("r"):
                roi_mgr.reset()
                roi, armor_roi, tr_roi = roi_mgr.build_all(monitor)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

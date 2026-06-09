"""
DetectorEngine — screen-capture / OCR loop that runs on a background thread.

All cv2 display calls are removed; results are published to EngineState so the
PySide6 overlay can read them.  The ocr_worker sub-thread is started inside run().
"""

import datetime
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import mss
import numpy as np

from config import (
    CAPTURE_PADDING, OUTPUT_DIR, REPLAY_INTERVAL_SEC,
    SCREEN_CLASSIFY_INTERVAL,
    SHIELD_BAR_X1, SHIELD_BAR_X2, SHIELD_STRIP_Y1, SHIELD_STRIP_Y2,
    TR_LEFT_FRAC, TR_TOP_FRAC, TR_WIDTH_FRAC, TR_HEIGHT_FRAC,
    WEAPON_LEFT_FRAC, WEAPON_TOP_FRAC,
)
from detectors import (
    ArmorDetector, ExperienceDetector, GameAggregator, RankedLoadingDetector,
    ScreenClassifier, ScreenType, ShieldDetector, TopRightDetector, WeaponDetector,
)
from map_locator import MapLocator
from preprocessing import expand_roi
from roi_manager import ROIManager
from session import GameSession, MatchPhase, PhaseDebouncer
from sinks import ExperienceWriter, MatchReplayWriter


# ---------------------------------------------------------------------------
# Shared state (GUI reads, engine writes)
# ---------------------------------------------------------------------------

@dataclass
class EngineState:
    lock:             threading.Lock  = field(default_factory=threading.Lock)
    recording:        threading.Event = field(default_factory=threading.Event)
    stop:             threading.Event = field(default_factory=threading.Event)
    # written by engine, read by GUI (hold lock while reading):
    fps:              float       = 0.0
    mode:             str         = "—"
    game_id:          str | None  = None
    rows_written:     int         = 0
    # set/cleared by GUI when record is toggled (hold lock):
    rec_started_mono: float | None = None

    def snapshot(self) -> dict:
        with self.lock:
            rec_elapsed: float | None = None
            if self.recording.is_set() and self.rec_started_mono is not None:
                rec_elapsed = time.monotonic() - self.rec_started_mono
            return {
                "fps":         self.fps,
                "mode":        self.mode,
                "game_id":     self.game_id,
                "rows":        self.rows_written,
                "recording":   self.recording.is_set(),
                "rec_elapsed": rec_elapsed,
            }


# ---------------------------------------------------------------------------
# Phase map
# ---------------------------------------------------------------------------

_PHASE_MAP = {
    ScreenType.GAME:           MatchPhase.GAME,
    ScreenType.EXPERIENCE:     MatchPhase.EXPERIENCE,
    ScreenType.RANKED_LOADING: MatchPhase.RANKED_LOADING,
    ScreenType.UNKNOWN:        MatchPhase.MENU,
}


# ---------------------------------------------------------------------------
# HUD presence check
# ---------------------------------------------------------------------------

def _hud_present(frame: np.ndarray) -> bool:
    """
    Fast pixel-level check for in-game HUD elements.

    Tests three regions independently and requires at least 2/3 to pass.
    This prevents entering GAME mode during non-gameplay screens (menus,
    spectate intro, kill cam, etc.) that the template classifier may
    misidentify.

    Thresholds are conservative — tune HUD_*_THRESH in config.py if needed.
    """
    h, w = frame.shape[:2]
    passed = 0

    # 1. Top-right panel: box borders + stat text create bright pixels
    tx1 = int(w * TR_LEFT_FRAC)
    ty1 = int(h * TR_TOP_FRAC)
    tx2 = int(w * (TR_LEFT_FRAC + TR_WIDTH_FRAC))
    ty2 = int(h * (TR_TOP_FRAC + TR_HEIGHT_FRAC))
    tr_crop = frame[ty1:ty2, tx1:tx2]
    if tr_crop.size > 0:
        gray = cv2.cvtColor(tr_crop, cv2.COLOR_BGR2GRAY)
        bright = np.count_nonzero(gray > 150)
        if bright > gray.size * 0.04:
            passed += 1

    # 2. Weapon ROI: bright text on dark HUD strip = high contrast
    wx1 = int(w * WEAPON_LEFT_FRAC)
    wy1 = int(h * WEAPON_TOP_FRAC)
    wp_crop = frame[wy1:h, wx1:w]
    if wp_crop.size > 0:
        gray = cv2.cvtColor(wp_crop, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) > 25.0:
            passed += 1

    # 3. Shield/health strip: colored (non-gray) bar fills
    sx1 = int(w * SHIELD_BAR_X1)
    sx2 = int(w * SHIELD_BAR_X2)
    sy1 = int(h * SHIELD_STRIP_Y1)
    sy2 = int(h * SHIELD_STRIP_Y2)
    sh_crop = frame[sy1:sy2, sx1:sx2]
    if sh_crop.size > 0:
        hsv = cv2.cvtColor(sh_crop, cv2.COLOR_BGR2HSV)
        colored = np.count_nonzero(
            (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 60)
        )
        if colored > hsv[:, :, 0].size * 0.03:
            passed += 1

    return passed >= 2


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DetectorEngine:
    def __init__(
        self,
        state:           EngineState,
        weapon_det:      WeaponDetector,
        armor_det:       ArmorDetector,
        tr_det:          TopRightDetector,
        shield_det:      ShieldDetector,
        classifier:      ScreenClassifier,
        exp_det:         ExperienceDetector,
        ranked_det:      RankedLoadingDetector,
        aggregator:      GameAggregator,
        map_loc:         MapLocator,
        roi_mgr:         ROIManager,
        replay_interval: float = REPLAY_INTERVAL_SEC,
    ):
        self.state           = state
        self.weapon_det      = weapon_det
        self.armor_det       = armor_det
        self.tr_det          = tr_det
        self.shield_det      = shield_det
        self.classifier      = classifier
        self.exp_det         = exp_det
        self.ranked_det      = ranked_det
        self.aggregator      = aggregator
        self.map_loc         = map_loc
        self.roi_mgr         = roi_mgr
        self.replay_interval = replay_interval

    def run(self) -> None:
        state = self.state

        debouncer = PhaseDebouncer()
        session   = GameSession()

        # Per-game sinks — created on GAME_STARTED into output/<game_id>/
        sinks: dict = {"exp": None, "replay": None}

        def _new_sinks_for_game(game_id: str) -> None:
            game_dir = os.path.join(OUTPUT_DIR, game_id)
            os.makedirs(game_dir, exist_ok=True)
            sinks["exp"]    = ExperienceWriter(os.path.join(game_dir, "match_history.csv"))
            sinks["replay"] = MatchReplayWriter(os.path.join(game_dir, "match_replay.csv"))

        # ── Shared frames/results between capture loop and OCR worker ────────
        latest_frames  = {"weapon": None, "armor": None, "tr": None, "full": None}
        latest_results = {
            "weapon": {"primary": None, "secondary": None},
            "armor":  {"number": None, "box": None},
            "shield": None,
            "tr":     {k: None for k in ("squads_remaining", "players_remaining",
                                         "kills", "assists", "participation", "damage")},
        }
        frame_locks = {k: threading.Lock() for k in latest_frames}
        result_lock = threading.Lock()

        def ocr_worker() -> None:
            while not state.stop.is_set():
                frames = {}
                for k, lock in frame_locks.items():
                    with lock:
                        f = latest_frames[k]
                        frames[k] = f.copy() if f is not None else None

                if all(f is None for f in frames.values()):
                    time.sleep(0.05)
                    continue

                weapon_r = self.weapon_det.detect(frames["weapon"]) if frames["weapon"] is not None else None
                armor_r  = self.armor_det.detect(frames["armor"])   if frames["armor"]  is not None else None
                tr_r     = self.tr_det.detect(frames["tr"])         if frames["tr"]     is not None else None
                shield_r = self.shield_det.detect(frames["full"])   if frames["full"]   is not None else None

                with result_lock:
                    stable = self.aggregator.update(
                        weapon=weapon_r, armor=armor_r,
                        shield=shield_r, tr=tr_r,
                    )
                    latest_results.update(stable)

        threading.Thread(target=ocr_worker, daemon=True).start()

        # ── Experience capture helper ─────────────────────────────────────────
        def _capture_experience(frame: np.ndarray, bby: int,
                                 map_name: str | None = None,
                                 squad_dist: dict | None = None) -> None:
            if session.game_id is None or sinks["exp"] is None:
                return
            exp_result  = self.exp_det.detect(frame, bby)
            captured_at = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            if state.recording.is_set():
                sinks["exp"].write(exp_result, session.game_id,
                                   session.game_start_time, captured_at,
                                   map_name=map_name, squad_dist=squad_dist)
                with state.lock:
                    state.rows_written += 1
            session.mark_experience_captured()
            suffix = "" if state.recording.is_set() else "  (not recording — skipped write)"
            print(f"  [EXP] Captured — game_id={session.game_id}  map={map_name or '?'}{suffix}")

        with mss.MSS() as sct:
            monitor = sct.monitors[1]
            roi, armor_roi, tr_roi = self.roi_mgr.build_all(monitor)
            print(f"Weapon ROI:    {roi['width']}x{roi['height']} at ({roi['left']}, {roi['top']})")
            print(f"Armor ROI:     {armor_roi['width']}x{armor_roi['height']} at ({armor_roi['left']}, {armor_roi['top']})")
            print(f"Top-right ROI: {tr_roi['width']}x{tr_roi['height']} at ({tr_roi['left']}, {tr_roi['top']})\n")

            screen_type                   = ScreenType.UNKNOWN
            _hud_was_absent               = True   # True → show READY, False → show type name
            bar_bottom_y                  = 0
            classify_counter              = 0
            last_replay_mono              = time.monotonic()
            last_exp_frame: np.ndarray | None = None
            _ranked_loading_map_captured  = False
            _fps_ts: deque[float]         = deque(maxlen=30)

            last_printed    = {"primary": None, "secondary": None, "armor-level": None}
            last_printed_tr = {k: None for k in ("squads_remaining", "players_remaining",
                                                   "kills", "assists", "participation", "damage")}

            def _print_on_change(key: str, new_val, last: dict) -> None:
                if new_val != last.get(key):
                    last[key] = new_val
                    print(f"  {key.upper():20s}: {new_val or '—'}")

            while not state.stop.is_set():
                full_bgr = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
                ox, oy   = monitor["left"], monitor["top"]

                # ── Classify + HUD check + debounce + session ─────────────────
                classify_counter += 1
                if classify_counter >= SCREEN_CLASSIFY_INTERVAL:
                    classify_counter = 0
                    new_type, bar_bottom_y = self.classifier.classify(full_bgr)

                    # Require HUD elements before entering GAME — prevents
                    # menus/cutscenes/kill-cam from triggering recording.
                    if new_type == ScreenType.GAME and not _hud_present(full_bgr):
                        _hud_was_absent = True
                        new_type = ScreenType.UNKNOWN  # treat as MENU for session
                    else:
                        _hud_was_absent = False

                    raw_phase = _PHASE_MAP[new_type]
                    confirmed = debouncer.push(raw_phase)

                    if confirmed is not None:
                        if confirmed == MatchPhase.RANKED_LOADING:
                            _ranked_loading_map_captured = False
                        _pre_game_id = session.game_id
                        _pre_game_st = session.game_start_time
                        events = session.update(confirmed)
                        for ev in events:
                            if ev == GameSession.Event.GAME_STARTED:
                                with result_lock:
                                    self.aggregator.reset()
                                self.exp_det.reset()
                                last_replay_mono = time.monotonic()
                                _new_sinks_for_game(session.game_id)
                                print(f"  [SESSION] New game — id={session.game_id}  map={session.map_name or '?'}")
                            elif ev == GameSession.Event.EXPERIENCE_EXIT_UNCAPTURED:
                                if last_exp_frame is not None and _pre_game_id is not None and sinks["exp"] is not None:
                                    print("  [EXP] Best-effort capture on exit")
                                    exp_result  = self.exp_det.detect(last_exp_frame, bar_bottom_y)
                                    captured_at = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                                    if state.recording.is_set():
                                        sinks["exp"].write(exp_result, _pre_game_id,
                                                           _pre_game_st, captured_at,
                                                           map_name=session.map_name,
                                                           squad_dist=session.squad_dist)
                                        with state.lock:
                                            state.rows_written += 1
                                    if session.game_id == _pre_game_id:
                                        session.mark_experience_captured()

                        screen_type = new_type
                        if confirmed != MatchPhase.EXPERIENCE:
                            last_exp_frame = None

                # READY = classifier saw a game-like screen but HUD elements absent.
                # Genuine UNKNOWN = classifier couldn't match any template.
                if screen_type == ScreenType.UNKNOWN and _hud_was_absent:
                    display_mode = "READY"
                else:
                    display_mode = screen_type.name

                # ── Rolling FPS ───────────────────────────────────────────────
                _fps_ts.append(time.monotonic())
                fps = (len(_fps_ts) - 1) / (_fps_ts[-1] - _fps_ts[0]) if len(_fps_ts) > 1 else 0.0

                # ── Publish to overlay ────────────────────────────────────────
                with state.lock:
                    state.fps     = fps
                    state.mode    = display_mode
                    state.game_id = session.game_id

                # ── Branch: EXPERIENCE ────────────────────────────────────────
                if screen_type == ScreenType.EXPERIENCE:
                    last_exp_frame = full_bgr
                    if session.should_capture_experience():
                        _capture_experience(full_bgr, bar_bottom_y,
                                            session.map_name, session.squad_dist)

                # ── Branch: GAME ──────────────────────────────────────────────
                elif screen_type == ScreenType.GAME:
                    p_roi   = expand_roi(roi,       monitor, CAPTURE_PADDING)
                    p_armor = expand_roi(armor_roi, monitor, CAPTURE_PADDING)
                    p_tr    = expand_roi(tr_roi,    monitor, CAPTURE_PADDING)

                    def crop(r: dict) -> np.ndarray:
                        return full_bgr[
                            r["top"]  - oy : r["top"]  - oy + r["height"],
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
                        agg_state  = dict(latest_results)

                    for slot in ("primary", "secondary"):
                        entry = weapon_res.get(slot)
                        _print_on_change(slot, entry[0] if entry else None, last_printed)
                    _print_on_change("armor-level", armor_res.get("number"), last_printed)
                    for key, entry in tr_res.items():
                        _print_on_change(key, entry[0] if entry else None, last_printed_tr)

                    now_mono = time.monotonic()
                    if (session.game_id is not None
                            and now_mono - last_replay_mono >= self.replay_interval):
                        last_replay_mono = now_mono
                        if state.recording.is_set() and sinks["replay"] is not None:
                            row_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                            map_info = self.map_loc.locate(full_bgr, session.map_name)
                            sinks["replay"].write(
                                agg_state,
                                session.game_id,
                                session.game_start_time,
                                row_time,
                                session.elapsed_s,
                                map_name=session.map_name,
                                map_info=map_info,
                            )
                            with state.lock:
                                state.rows_written += 1

                # ── Branch: RANKED_LOADING ────────────────────────────────────
                elif screen_type == ScreenType.RANKED_LOADING:
                    if not _ranked_loading_map_captured:
                        map_name_ocr = self.ranked_det.detect_map_name(full_bgr)
                        dist_ocr     = self.ranked_det.detect_squad_distribution(full_bgr)
                        if map_name_ocr:
                            session.set_map_name(map_name_ocr)
                        if dist_ocr:
                            session.set_squad_distribution(dist_ocr)
                        if map_name_ocr:
                            _ranked_loading_map_captured = True
                            print(f"  [MAP] {map_name_ocr}  dist={dist_ocr}")

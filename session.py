"""
Match session state machine for the Apex Legends detector.

MatchPhase      — MENU / GAME / EXPERIENCE
PhaseDebouncer  — requires K consecutive same classifications before committing
GameSession     — owns game_id, game_start_time, and all match-boundary logic

All timers use time.monotonic() for interval arithmetic (immune to NTP jumps).
Wall-clock timestamps (ISO 8601 strings) are produced by time.time() / datetime.

Clocks are injected so unit tests can supply a fake clock without sleeping.
"""

import datetime
import enum
import random
import string
import time as _time_module

from config import (
    EXPERIENCE_CAPTURE_DELAY_SEC,
    NEW_GAME_GAP_SEC,
    PHASE_DEBOUNCE_FRAMES,
)


class MatchPhase(enum.Enum):
    MENU           = "menu"           # lobby / unknown / not in match
    RANKED_LOADING = "ranked_loading" # ranked lobby / dropzone selection
    GAME           = "game"           # in-match HUD
    EXPERIENCE     = "experience"     # post-match summary screen


# ---------------------------------------------------------------------------
# PhaseDebouncer
# ---------------------------------------------------------------------------

class PhaseDebouncer:
    """Requires ``window`` consecutive identical raw classifications before
    committing a phase change.  Prevents single-frame classifier flicker from
    triggering match-boundary events.

    Usage::

        debouncer = PhaseDebouncer(window=PHASE_DEBOUNCE_FRAMES)
        confirmed = debouncer.push(raw_phase)   # None until stable
    """

    def __init__(self, window: int = PHASE_DEBOUNCE_FRAMES):
        self._window    = window
        self._candidate = None
        self._count     = 0
        self._confirmed = None

    def push(self, raw: MatchPhase) -> MatchPhase | None:
        """Feed one raw observation; return the newly confirmed phase or None."""
        if raw == self._candidate:
            self._count += 1
        else:
            self._candidate = raw
            self._count     = 1

        if self._count >= self._window and raw != self._confirmed:
            self._confirmed = raw
            return raw
        return None

    @property
    def confirmed(self) -> MatchPhase | None:
        return self._confirmed


# ---------------------------------------------------------------------------
# GameSession
# ---------------------------------------------------------------------------

def _make_game_id(now_fn) -> str:
    ts    = datetime.datetime.fromtimestamp(now_fn()).strftime("%Y%m%d-%H%M%S")
    rand  = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{ts}-{rand}"


def _iso(now_fn) -> str:
    return datetime.datetime.fromtimestamp(now_fn()).strftime("%Y-%m-%dT%H:%M:%S")


class GameSession:
    """Owns ``game_id``, ``game_start_time``, and the match-boundary logic.

    Drive by calling ``update(confirmed_phase)`` each time ``PhaseDebouncer``
    commits a new phase.  The caller inspects the returned event list and acts
    on ``Events.GAME_STARTED``, ``Events.EXPERIENCE_READY`` etc.

    New-game rule
    -------------
    A new game_id is minted when entering GAME if either:
      (a) an EXPERIENCE screen was seen since the last game started, **or**
      (b) we have been in a non-GAME phase for ≥ ``new_game_gap_s`` seconds.
    Otherwise (brief UNKNOWN flicker) the same match continues.

    Experience capture
    ------------------
    ``should_capture_experience()`` returns True once:
      - We are in EXPERIENCE phase, AND
      - ``experience_delay_s`` seconds have elapsed since EXPERIENCE was first
        entered, AND
      - ``mark_experience_captured()`` has not yet been called for this game.

    Best-effort capture: on the EXPERIENCE→non-EXPERIENCE transition, if the
    delay fired but ``mark_experience_captured()`` was never called,
    ``EXPERIENCE_EXIT_UNCAPTURED`` is emitted so the caller can OCR and write
    immediately before moving on.
    """

    class Event(enum.Enum):
        GAME_STARTED              = "game_started"
        EXPERIENCE_READY          = "experience_ready"          # delay elapsed
        EXPERIENCE_EXIT_UNCAPTURED = "experience_exit_uncaptured"  # best-effort

    def __init__(
        self,
        clock=None,   # monotonic, for intervals
        now=None,     # wall clock, for timestamps
        experience_delay_s: float = EXPERIENCE_CAPTURE_DELAY_SEC,
        new_game_gap_s: float     = NEW_GAME_GAP_SEC,
    ):
        self._clock              = clock or _time_module.monotonic
        self._now                = now   or _time_module.time
        self._exp_delay          = experience_delay_s
        self._new_game_gap       = new_game_gap_s

        self._phase              = MatchPhase.MENU
        self._game_id: str | None            = None
        self._game_start_time: str | None    = None
        self._game_start_mono: float | None  = None

        self._had_experience     = False   # EXPERIENCE seen in this match
        self._last_nongame_mono: float | None = None  # when non-GAME began

        self._exp_first_mono: float | None = None     # when EXPERIENCE started
        self._exp_captured   = False

        self._map_name: str | None   = None  # set from RANKED_LOADING screen OCR
        self._squad_dist: dict | None = None  # set from RANKED_LOADING squad bar

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, phase: MatchPhase) -> list["GameSession.Event"]:
        """Advance the session with a newly confirmed phase; return events."""
        now_mono = self._clock()
        events   = []

        if phase == self._phase:
            # Same phase — check timers only.
            events.extend(self._check_timers(now_mono))
            return events

        prev = self._phase
        self._phase = phase

        # ── Leaving a phase ──────────────────────────────────────────────
        if prev == MatchPhase.GAME:
            self._last_nongame_mono = now_mono

        if prev == MatchPhase.EXPERIENCE:
            if not self._exp_captured and self._exp_first_mono is not None:
                events.append(self.Event.EXPERIENCE_EXIT_UNCAPTURED)
            self._exp_first_mono = None

        # ── Entering a phase ─────────────────────────────────────────────
        if phase == MatchPhase.GAME:
            if self._should_start_new_game(now_mono):
                self._start_game(now_mono)
                events.append(self.Event.GAME_STARTED)
            # (else: resume same game after brief flicker)

        elif phase == MatchPhase.EXPERIENCE:
            self._had_experience = True
            self._exp_first_mono = now_mono
            # _exp_captured is NOT reset here — it stays True if already captured
            # for this game_id.  It is only reset in _start_game() (new match).
            # Orphan case: summary screen with no preceding game this session.
            if self._game_id is None:
                self._start_game(now_mono)
                events.append(self.Event.GAME_STARTED)

        elif phase in (MatchPhase.MENU, MatchPhase.RANKED_LOADING):
            pass   # nothing to do on entering MENU or RANKED_LOADING

        events.extend(self._check_timers(now_mono))
        return events

    def should_capture_experience(self) -> bool:
        """True when the 5 s delay has elapsed and we haven't captured yet."""
        if self._exp_captured or self._exp_first_mono is None:
            return False
        return self._clock() - self._exp_first_mono >= self._exp_delay

    def mark_experience_captured(self) -> None:
        self._exp_captured = True

    def set_map_name(self, name: str | None) -> None:
        """Record the map from the ranked-loading screen OCR."""
        self._map_name = name

    def set_squad_distribution(self, dist: dict | None) -> None:
        """Record the squad rank distribution from the ranked-loading bar."""
        self._squad_dist = dist

    @property
    def map_name(self) -> str | None:
        return self._map_name

    @property
    def squad_dist(self) -> dict | None:
        return self._squad_dist

    @property
    def game_id(self) -> str | None:
        return self._game_id

    @property
    def game_start_time(self) -> str | None:
        return self._game_start_time

    @property
    def elapsed_s(self) -> float:
        """Seconds since game start (monotonic); 0 if no game active."""
        if self._game_start_mono is None:
            return 0.0
        return self._clock() - self._game_start_mono

    @property
    def phase(self) -> MatchPhase:
        return self._phase

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_start_new_game(self, now_mono: float) -> bool:
        if self._game_id is None:
            return True
        if self._had_experience:
            return True
        if (self._last_nongame_mono is not None
                and now_mono - self._last_nongame_mono >= self._new_game_gap):
            return True
        return False

    def _start_game(self, now_mono: float) -> None:
        self._game_id         = _make_game_id(self._now)
        self._game_start_time = _iso(self._now)
        self._game_start_mono = now_mono
        self._had_experience  = False
        self._last_nongame_mono = None
        self._exp_captured    = False   # reset for the new match

    def _check_timers(self, now_mono: float) -> list["GameSession.Event"]:
        events = []
        if (self._phase == MatchPhase.EXPERIENCE
                and not self._exp_captured
                and self._exp_first_mono is not None
                and now_mono - self._exp_first_mono >= self._exp_delay):
            events.append(self.Event.EXPERIENCE_READY)
        return events

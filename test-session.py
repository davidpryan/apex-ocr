"""
Unit tests for session.py (PhaseDebouncer + GameSession).
No OpenCV, no capture hardware — entirely driven by a fake clock.

Run with:  python3 test-session.py

Covers every edge case from next_steps.md §8.
"""

import os
import sys
import tempfile

from session import GameSession, MatchPhase, PhaseDebouncer
from sinks import ExperienceWriter, MatchReplayWriter

E = GameSession.Event


# ---------------------------------------------------------------------------
# Fake clock helpers
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, t=0.0):
        self.t = t
    def __call__(self):
        return self.t
    def advance(self, dt):
        self.t += dt


# ---------------------------------------------------------------------------
# PhaseDebouncer tests
# ---------------------------------------------------------------------------

def test_debouncer_requires_window():
    d = PhaseDebouncer(window=3)
    assert d.push(MatchPhase.GAME) is None
    assert d.push(MatchPhase.GAME) is None
    assert d.push(MatchPhase.GAME) == MatchPhase.GAME
    assert d.confirmed == MatchPhase.GAME


def test_debouncer_resets_on_change():
    d = PhaseDebouncer(window=3)
    d.push(MatchPhase.GAME)
    d.push(MatchPhase.GAME)
    d.push(MatchPhase.MENU)   # flicker — resets count
    assert d.push(MatchPhase.GAME) is None  # must start again


def test_debouncer_no_repeat_event():
    d = PhaseDebouncer(window=2)
    d.push(MatchPhase.GAME); d.push(MatchPhase.GAME)
    assert d.push(MatchPhase.GAME) is None  # already confirmed, no re-emit


# ---------------------------------------------------------------------------
# GameSession tests
# ---------------------------------------------------------------------------

def _session(exp_delay=5.0, gap=20.0):
    mono = FakeClock()
    wall = FakeClock(1_700_000_000.0)
    sess = GameSession(clock=mono, now=wall,
                       experience_delay_s=exp_delay, new_game_gap_s=gap)
    return sess, mono, wall


def test_first_game_mints_id():
    sess, mono, _ = _session()
    events = sess.update(MatchPhase.GAME)
    assert E.GAME_STARTED in events
    assert sess.game_id is not None


def test_two_consecutive_games_get_different_ids():
    """Two consecutive games → two distinct game_ids, replays correctly partitioned."""
    sess, mono, _ = _session()
    sess.update(MatchPhase.GAME)
    id1 = sess.game_id

    # Experience screen between the two games
    sess.update(MatchPhase.EXPERIENCE)
    mono.advance(6)
    sess.update(MatchPhase.EXPERIENCE)   # tick timers

    sess.update(MatchPhase.GAME)
    id2 = sess.game_id

    assert id1 != id2, "Second game must have a fresh game_id"


def test_unknown_flicker_keeps_same_id():
    """Mid-game UNKNOWN flicker → no new game_id, no split."""
    sess, mono, _ = _session(gap=20.0)
    sess.update(MatchPhase.GAME)
    id1 = sess.game_id

    mono.advance(2)
    sess.update(MatchPhase.MENU)   # brief gap (2 s < 20 s gap)
    mono.advance(2)
    events = sess.update(MatchPhase.GAME)

    assert E.GAME_STARTED not in events, "Brief flicker must not start a new game"
    assert sess.game_id == id1


def test_sustained_gap_starts_new_game():
    """Game with no experience screen → next game after gap still gets new id."""
    sess, mono, _ = _session(gap=20.0)
    sess.update(MatchPhase.GAME)
    id1 = sess.game_id

    sess.update(MatchPhase.MENU)        # leave game; gap timer starts here
    mono.advance(25)                    # 25 s pass in non-GAME (> 20 s gap)
    events = sess.update(MatchPhase.GAME)

    assert E.GAME_STARTED in events
    assert sess.game_id != id1


def test_experience_ready_after_delay():
    """EXPERIENCE + delay elapsed → EXPERIENCE_READY fired, capture allowed."""
    sess, mono, _ = _session(exp_delay=5.0)
    sess.update(MatchPhase.GAME)
    sess.update(MatchPhase.EXPERIENCE)

    mono.advance(4)
    events = sess.update(MatchPhase.EXPERIENCE)
    assert E.EXPERIENCE_READY not in events
    assert not sess.should_capture_experience()

    mono.advance(2)
    events = sess.update(MatchPhase.EXPERIENCE)
    assert E.EXPERIENCE_READY in events
    assert sess.should_capture_experience()


def test_experience_captured_only_once():
    """Re-enter EXPERIENCE / flicker → still only one experience row per game."""
    sess, mono, _ = _session(exp_delay=5.0)
    sess.update(MatchPhase.GAME)
    sess.update(MatchPhase.EXPERIENCE)
    mono.advance(6)
    sess.update(MatchPhase.EXPERIENCE)

    assert sess.should_capture_experience()
    sess.mark_experience_captured()
    assert not sess.should_capture_experience()

    # Flicker back and forth
    sess.update(MatchPhase.MENU)
    sess.update(MatchPhase.EXPERIENCE)
    mono.advance(6)
    sess.update(MatchPhase.EXPERIENCE)
    assert not sess.should_capture_experience(), "Already captured — must not fire again"


def test_short_experience_best_effort():
    """Experience screen < 5 s → best-effort EXPERIENCE_EXIT_UNCAPTURED fired."""
    sess, mono, _ = _session(exp_delay=5.0)
    sess.update(MatchPhase.GAME)
    sess.update(MatchPhase.EXPERIENCE)
    mono.advance(2)   # only 2 s — delay not elapsed

    events = sess.update(MatchPhase.GAME)   # player dismissed summary early
    assert E.EXPERIENCE_EXIT_UNCAPTURED in events


def test_program_starts_on_experience_orphan():
    """Program starts on EXPERIENCE → orphan game_id minted, row can be written."""
    sess, _, _ = _session()
    assert sess.game_id is None
    events = sess.update(MatchPhase.EXPERIENCE)
    assert E.GAME_STARTED in events, "Orphan game_id must be minted"
    assert sess.game_id is not None


def test_game_start_time_consistent():
    """game_start_time identical across a game's replay rows and experience row."""
    sess, mono, _ = _session(exp_delay=5.0)
    sess.update(MatchPhase.GAME)
    start_time = sess.game_start_time

    mono.advance(30)
    sess.update(MatchPhase.EXPERIENCE)
    mono.advance(6)
    sess.update(MatchPhase.EXPERIENCE)

    assert sess.game_start_time == start_time, "game_start_time must not change mid-match"


def test_replay_cadence_monotonic():
    """elapsed_s is monotonic-driven, not frame-count-driven."""
    sess, mono, _ = _session()
    sess.update(MatchPhase.GAME)
    mono.advance(5)
    assert abs(sess.elapsed_s - 5.0) < 0.01
    mono.advance(10)
    assert abs(sess.elapsed_s - 15.0) < 0.01


# ---------------------------------------------------------------------------
# Sinks integration tests (write/read round-trip, schema guard)
# ---------------------------------------------------------------------------

def test_experience_writer_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        w = ExperienceWriter(path)
        values = {"current_rank": "Platinum IV", "point_change": "+363",
                  "placement": "#2", "kills": "6", "assists": "1",
                  "participations_rp": "30", "base_combat_value": "20",
                  "combat_rp_total": "165", "bonus_rp_total": "286",
                  "challenger_count": "4", "promotion_rp": "250",
                  "placement_rp_total": "62", "cost_of_entry_rp": "-38"}
        w.write(values, "test-id-001", "2026-06-07T10:00:00", "2026-06-07T10:05:05")
        # Read back
        import csv as _csv
        with open(path) as fh:
            rows = list(_csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["game_id"] == "test-id-001"
        assert rows[0]["current_rank"] == "Platinum IV"
        assert rows[0]["placement"] == "#2"
    finally:
        os.unlink(path)


def test_replay_writer_suppresses_warmup():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        w = MatchReplayWriter(path)
        empty_state = {"weapon": {}, "armor": {}, "shield": None, "tr": {}}
        written = w.write(empty_state, "gid", "2026-06-07T10:00:00",
                          "2026-06-07T10:00:05", 5.0)
        assert not written, "Warm-up row with all-None stats must be suppressed"
    finally:
        os.unlink(path)


def test_replay_writer_writes_when_populated():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        w = MatchReplayWriter(path)
        state = {
            "weapon": {"primary": ("WINGMAN", None), "secondary": None},
            "armor":  {"number": "3", "box": None},
            "shield": {"shield_type": "blue", "shield_hp": 67,
                       "flesh_hp": 100, "health": 167},
            "tr":     {"squads_remaining": ("16", None), "players_remaining": ("44", None),
                       "kills": ("1", None), "assists": ("0", None),
                       "participation": ("2", None), "damage": ("69", None)},
        }
        written = w.write(state, "gid", "2026-06-07T10:00:00",
                          "2026-06-07T10:00:10", 10.0)
        assert written
        import csv as _csv
        with open(path) as fh:
            rows = list(_csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["primary_weapon"] == "WINGMAN"
        assert rows[0]["shield_type"] == "blue"
        assert rows[0]["kills"] == "1"
        assert rows[0]["map_x"] == ""   # reserved, null
    finally:
        os.unlink(path)


def test_sink_schema_guard():
    """Stale-schema CSV is archived; a fresh file is started."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
        f.write("old_col1,old_col2\n1,2\n")
    try:
        from sinks import CsvSink
        sink = CsvSink(path, ["new_col1", "new_col2", "new_col3"])
        sink.write_row({"new_col1": "a", "new_col2": "b", "new_col3": "c"})
        import csv as _csv
        with open(path) as fh:
            rows = list(_csv.DictReader(fh))
        assert len(rows) == 1
        assert "new_col1" in rows[0]
        # Backup file should exist
        bak_files = [f for f in os.listdir(os.path.dirname(path) or ".")
                     if f.startswith(os.path.basename(path) + ".bak.")]
        assert bak_files, "Stale file must be backed up"
    finally:
        os.unlink(path)
        for bak in bak_files:
            os.unlink(os.path.join(os.path.dirname(path) or ".", bak))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_debouncer_requires_window,
        test_debouncer_resets_on_change,
        test_debouncer_no_repeat_event,
        test_first_game_mints_id,
        test_two_consecutive_games_get_different_ids,
        test_unknown_flicker_keeps_same_id,
        test_sustained_gap_starts_new_game,
        test_experience_ready_after_delay,
        test_experience_captured_only_once,
        test_short_experience_best_effort,
        test_program_starts_on_experience_orphan,
        test_game_start_time_consistent,
        test_replay_cadence_monotonic,
        test_experience_writer_roundtrip,
        test_replay_writer_suppresses_warmup,
        test_replay_writer_writes_when_populated,
        test_sink_schema_guard,
    ]
    ok = fail = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            ok += 1
        except Exception as exc:
            print(f"  ✗ {t.__name__}: {exc}")
            fail += 1
    print(f"\n  {ok}/{ok+fail} tests passed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)

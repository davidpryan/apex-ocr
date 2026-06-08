"""
Integration tests: session + sink pipeline, no OpenCV required.

Exercises the §8 edge cases that require CSV-level verification — i.e., that
both outputs (experience.csv / match_replay.csv) are written correctly and
join on game_id across multi-game scenarios.

Run:  python3 test-integration.py
"""

import csv
import datetime
import os
import sys
import tempfile

from session import GameSession, MatchPhase
from sinks import ExperienceWriter, MatchReplayWriter

E = GameSession.Event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, t=0.0):
        self.t = t
    def __call__(self): return self.t
    def advance(self, dt): self.t += dt


def _iso(wall):
    return datetime.datetime.fromtimestamp(wall()).strftime("%Y-%m-%dT%H:%M:%S")


def _minimal_agg():
    """Non-null aggregator state that passes warm-up suppression."""
    return {
        "weapon": {"primary": ("WINGMAN", None), "secondary": None},
        "armor":  {"number": "3"},
        "shield": None,
        "tr":     {},
    }


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _setup(exp_path, replay_path, exp_delay=5.0, gap=20.0):
    mono = FakeClock()
    wall = FakeClock(1_700_000_000.0)
    sess = GameSession(clock=mono, now=wall,
                       experience_delay_s=exp_delay, new_game_gap_s=gap)
    exp_w  = ExperienceWriter(exp_path)
    rep_w  = MatchReplayWriter(replay_path)
    return sess, mono, wall, exp_w, rep_w


def _write_replay_if_due(sess, mono, wall, rep_w, last_replay, interval=5.0):
    """Write a replay row if the cadence interval has elapsed. Returns updated last_replay."""
    if sess.game_id is not None and mono() - last_replay >= interval:
        rep_w.write(_minimal_agg(), sess.game_id, sess.game_start_time,
                    _iso(wall), sess.elapsed_s)
        return mono()
    return last_replay


def _try_write_experience(sess, wall, exp_w):
    """Write experience row if the 5 s delay has fired. Returns True if written."""
    if sess.should_capture_experience():
        exp_w.write({}, sess.game_id, sess.game_start_time, _iso(wall))
        sess.mark_experience_captured()
        return True
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_two_games_joined_correctly():
    """Two consecutive games → 2 exp rows, 2 distinct game_ids, replay rows partitioned."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as ef, \
         tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as rf:
        exp_path, replay_path = ef.name, rf.name
    try:
        sess, mono, wall, exp_w, rep_w = _setup(exp_path, replay_path)
        last_replay = mono()

        # ── Game 1: 3 replay rows ─────────────────────────────────────────────
        sess.update(MatchPhase.GAME)
        id1, gst1 = sess.game_id, sess.game_start_time
        for _ in range(3):
            mono.advance(5); wall.advance(5)
            last_replay = _write_replay_if_due(sess, mono, wall, rep_w, last_replay)

        # ── Experience 1 ──────────────────────────────────────────────────────
        sess.update(MatchPhase.EXPERIENCE)
        mono.advance(6); wall.advance(6)
        _try_write_experience(sess, wall, exp_w)

        # ── Game 2: 2 replay rows ─────────────────────────────────────────────
        events = sess.update(MatchPhase.GAME)
        assert E.GAME_STARTED in events
        id2, gst2 = sess.game_id, sess.game_start_time
        assert id1 != id2
        last_replay = mono()
        for _ in range(2):
            mono.advance(5); wall.advance(5)
            last_replay = _write_replay_if_due(sess, mono, wall, rep_w, last_replay)

        # ── Experience 2 ──────────────────────────────────────────────────────
        sess.update(MatchPhase.EXPERIENCE)
        mono.advance(6); wall.advance(6)
        _try_write_experience(sess, wall, exp_w)

        # ── Assert ────────────────────────────────────────────────────────────
        exp_rows = _read_csv(exp_path)
        rep_rows = _read_csv(replay_path)

        assert len(exp_rows) == 2, f"Expected 2 exp rows, got {len(exp_rows)}"
        assert exp_rows[0]["game_id"] == id1
        assert exp_rows[1]["game_id"] == id2
        assert exp_rows[0]["game_start_time"] == gst1
        assert exp_rows[1]["game_start_time"] == gst2

        by_game = {}
        for r in rep_rows:
            by_game.setdefault(r["game_id"], []).append(r)
        assert set(by_game) == {id1, id2}, f"Unexpected game_ids in replay: {set(by_game)}"
        assert len(by_game[id1]) == 3, f"Game 1 should have 3 replay rows, got {len(by_game[id1])}"
        assert len(by_game[id2]) == 2, f"Game 2 should have 2 replay rows, got {len(by_game[id2])}"

        # game_start_time must be consistent within each game across both CSVs
        for r in by_game[id1]:
            assert r["game_start_time"] == gst1
        for r in by_game[id2]:
            assert r["game_start_time"] == gst2
    finally:
        os.unlink(exp_path)
        os.unlink(replay_path)


def test_game_no_experience_screen():
    """Game with no experience → replay rows only; next game after gap gets new id."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as ef, \
         tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as rf:
        exp_path, replay_path = ef.name, rf.name
    try:
        sess, mono, wall, exp_w, rep_w = _setup(exp_path, replay_path, gap=20.0)
        last_replay = mono()

        sess.update(MatchPhase.GAME)
        id1 = sess.game_id
        for _ in range(2):
            mono.advance(5); wall.advance(5)
            last_replay = _write_replay_if_due(sess, mono, wall, rep_w, last_replay)

        # Leave without hitting experience screen (disconnect / quit)
        sess.update(MatchPhase.MENU)
        mono.advance(25); wall.advance(25)

        events = sess.update(MatchPhase.GAME)
        assert E.GAME_STARTED in events
        id2 = sess.game_id
        assert id1 != id2
        last_replay = mono()
        mono.advance(5); wall.advance(5)
        _write_replay_if_due(sess, mono, wall, rep_w, last_replay)

        exp_rows = _read_csv(exp_path)
        rep_rows = _read_csv(replay_path)

        assert len(exp_rows) == 0, f"Expected no exp rows, got {len(exp_rows)}"

        by_game = {}
        for r in rep_rows:
            by_game.setdefault(r["game_id"], []).append(r)
        assert set(by_game) == {id1, id2}
        assert len(by_game[id1]) == 2
        assert len(by_game[id2]) == 1
    finally:
        os.unlink(exp_path)
        os.unlink(replay_path)


def test_best_effort_capture_writes_row():
    """Experience screen dismissed before 5 s delay → best-effort path writes one row."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as ef, \
         tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as rf:
        exp_path, replay_path = ef.name, rf.name
    try:
        sess, mono, wall, exp_w, rep_w = _setup(exp_path, replay_path, exp_delay=5.0)

        sess.update(MatchPhase.GAME)
        game_id = sess.game_id

        sess.update(MatchPhase.EXPERIENCE)
        mono.advance(2); wall.advance(2)   # only 2 s — delay NOT elapsed

        # Simulate detect-weapons.py: save context BEFORE update
        pre_game_id = sess.game_id
        pre_game_st = sess.game_start_time
        events = sess.update(MatchPhase.MENU)

        assert E.EXPERIENCE_EXIT_UNCAPTURED in events
        assert E.GAME_STARTED not in events   # MENU doesn't start a new game

        # Best-effort write using pre-saved game context
        exp_w.write({}, pre_game_id, pre_game_st, _iso(wall))
        if sess.game_id == pre_game_id:
            sess.mark_experience_captured()

        exp_rows = _read_csv(exp_path)
        assert len(exp_rows) == 1, f"Expected 1 exp row, got {len(exp_rows)}"
        assert exp_rows[0]["game_id"] == game_id
    finally:
        os.unlink(exp_path)
        os.unlink(replay_path)


def test_best_effort_uses_old_game_id_on_direct_exp_to_game_transition():
    """EXPERIENCE→GAME direct: best-effort row carries the OLD game's id, not the new one."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as ef, \
         tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as rf:
        exp_path, replay_path = ef.name, rf.name
    try:
        sess, mono, wall, exp_w, rep_w = _setup(exp_path, replay_path, exp_delay=5.0)

        sess.update(MatchPhase.GAME)
        old_game_id = sess.game_id
        old_game_st = sess.game_start_time

        sess.update(MatchPhase.EXPERIENCE)
        mono.advance(2); wall.advance(2)

        # Save BEFORE update (mirrors the fix in detect-weapons.py)
        pre_game_id = sess.game_id
        pre_game_st = sess.game_start_time
        events = sess.update(MatchPhase.GAME)   # direct EXPERIENCE→GAME transition

        assert E.EXPERIENCE_EXIT_UNCAPTURED in events
        assert E.GAME_STARTED in events          # new game was minted
        new_game_id = sess.game_id
        assert new_game_id != old_game_id        # sanity check

        # Best-effort: use pre_game_id (old game); don't mark_experience_captured
        # because _start_game() already reset _exp_captured for the new game.
        exp_w.write({}, pre_game_id, pre_game_st, _iso(wall))
        if sess.game_id == pre_game_id:
            sess.mark_experience_captured()

        exp_rows = _read_csv(exp_path)
        assert len(exp_rows) == 1
        assert exp_rows[0]["game_id"] == old_game_id, (
            f"Row must carry old_game_id={old_game_id!r}, "
            f"got {exp_rows[0]['game_id']!r} (new_game_id={new_game_id!r})"
        )
    finally:
        os.unlink(exp_path)
        os.unlink(replay_path)


def test_orphan_experience_writes_row():
    """Program starts on EXPERIENCE → orphan game_id minted, one exp row written."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as ef, \
         tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as rf:
        exp_path, replay_path = ef.name, rf.name
    try:
        sess, mono, wall, exp_w, rep_w = _setup(exp_path, replay_path)
        assert sess.game_id is None

        events = sess.update(MatchPhase.EXPERIENCE)
        assert E.GAME_STARTED in events   # orphan game_id minted
        assert sess.game_id is not None

        mono.advance(6); wall.advance(6)
        _try_write_experience(sess, wall, exp_w)

        exp_rows = _read_csv(exp_path)
        assert len(exp_rows) == 1
        assert exp_rows[0]["game_id"]   # non-empty
    finally:
        os.unlink(exp_path)
        os.unlink(replay_path)


def test_no_duplicate_experience_on_reentry():
    """Re-entering EXPERIENCE after capture does not write a second row."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as ef, \
         tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as rf:
        exp_path, replay_path = ef.name, rf.name
    try:
        sess, mono, wall, exp_w, rep_w = _setup(exp_path, replay_path)

        sess.update(MatchPhase.GAME)
        sess.update(MatchPhase.EXPERIENCE)
        mono.advance(6); wall.advance(6)
        _try_write_experience(sess, wall, exp_w)

        # Flicker: brief menu, then re-enter experience
        sess.update(MatchPhase.MENU)
        sess.update(MatchPhase.EXPERIENCE)
        mono.advance(6); wall.advance(6)

        wrote = _try_write_experience(sess, wall, exp_w)
        assert not wrote, "should_capture_experience() must be False after mark_experience_captured()"

        exp_rows = _read_csv(exp_path)
        assert len(exp_rows) == 1, f"Expected 1 row, got {len(exp_rows)}"
    finally:
        os.unlink(exp_path)
        os.unlink(replay_path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_two_games_joined_correctly,
        test_game_no_experience_screen,
        test_best_effort_capture_writes_row,
        test_best_effort_uses_old_game_id_on_direct_exp_to_game_transition,
        test_orphan_experience_writes_row,
        test_no_duplicate_experience_on_reentry,
    ]
    ok = fail = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            ok += 1
        except Exception as exc:
            import traceback
            print(f"  ✗ {t.__name__}: {exc}")
            traceback.print_exc()
            fail += 1
    print(f"\n  {ok}/{ok+fail} tests passed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)

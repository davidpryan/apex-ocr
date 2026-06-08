# Next Steps — Per-Game Experience Capture + Live Match Replay

Plan to evolve the current single-CSV detector into two associated outputs:

1. **Experience CSV** — one row per game, captured **once, 5 s after** the
   experience screen is first detected (so all values have populated).
2. **`match_replay.csv`** — one row every **X seconds (≈5)** during live play, each
   tagged with a `game_id` and `game_start_time`, plus (future) the player's
   position on the map.

The two outputs must be **correctly associated**: the experience row for a game and
all that game's replay rows share the same `game_id`.

---

## 1. Target State

```
                 ┌─────────────────────────── ScreenClassifier (every N frames)
                 │
   MENU/UNKNOWN ─┤
                 │   on first GAME after a match boundary →
                 ▼        mint game_id + game_start_time (GameSession.start)
   ┌──────────────────────────┐
   │ GAME                      │  every X s → MatchReplayWriter.write(row, game_id)
   │  run HUD detectors        │
   │  → GameAggregator         │
   └──────────────────────────┘
                 │  on first EXPERIENCE → record experience_seen_at = now
                 ▼  (game_id is retained from the game that just ended)
   ┌──────────────────────────┐
   │ EXPERIENCE                │  at experience_seen_at + 5 s, once →
   │  ExperienceDetector       │     ExperienceWriter.write(values, game_id)
   └──────────────────────────┘
                 │  on MENU/UNKNOWN → match boundary closed
                 ▼
   (next GAME mints a NEW game_id)
```

`game_id` links the two CSVs. A `match_replay` row and the experience row for the
same match carry identical `game_id` + `game_start_time`.

---

## 2. Current State (what exists today)

| Piece | File | Notes |
|---|---|---|
| Screen classification (GAME / EXPERIENCE / UNKNOWN) | `detectors.py` `ScreenClassifier` | runs every `SCREEN_CLASSIFY_INTERVAL` frames |
| Experience values | `detectors.py` `ExperienceDetector.detect()` | static-screen result cache; **writes CSV itself** |
| Experience CSV | `results.csv` via `_append_csv()` | 14 columns, **no game_id**, written on first detect |
| Live HUD detectors | `WeaponDetector`, `ArmorDetector`, `TopRightDetector`, `ShieldDetector` | |
| Temporal stabilization | `GameAggregator` | already wired into `detect-weapons.py` |
| Live loop | `detect-weapons.py` `main()` | capture → classify → branch GAME/EXPERIENCE |

### Critical issues in the current code that block the target state
1. **`ExperienceDetector.detect()` writes CSV internally** (`_append_csv` at the end
   of `detect()`). Capture policy (when/whether to persist, which `game_id`) is the
   loop's job, not the detector's. This must be decoupled.
2. **No 5 s delay** — the experience row is written on the *first* detect, before
   values have settled.
3. **`exp_det.reset()` is never called** in `detect-weapons.py`. Across a multi-game
   session the cached result from the previous match would be re-emitted for the next
   match's experience screen. (Confirmed: only `aggregator.reset()` is wired.)
4. **No session / `game_id` concept** anywhere.
5. **No periodic (every-X-s) capture** for live play.
6. **`results.csv` schema has no `game_id`/timestamp**, so association is impossible.

---

## 3. Gap Analysis — shortcomings, inefficiencies, missing systems

### Missing systems
- **GameSession state machine** — owns `game_id`, `game_start_time`, the
  match-boundary logic, and the "experience captured?" flag. Does not exist.
- **Wall-clock/monotonic scheduler** — timers for the 5 s experience delay and the
  X-second replay cadence. Use `time.monotonic()` for intervals (immune to NTP
  jumps) and `time.time()`/`datetime` only for the *recorded* timestamps.
- **Two output sinks** — `ExperienceWriter` and `MatchReplayWriter` classes that own
  their CSV schema, header creation, and `game_id` injection. Replaces the
  detector-embedded `_append_csv`.
- **Map locator (future)** — `MapLocator.locate(full_bgr) -> {map_x, map_y,
  location}`; reserve nullable columns now so the schema is stable.

### Shortcomings / correctness risks
- **Match-boundary debouncing.** Classifier flicker (a stray UNKNOWN mid-fight, or a
  single GAME frame in a menu) must not mint a new `game_id` or split a match. Need
  hysteresis: require *K consecutive* classifications before committing a transition,
  **and** define a new game as "GAME seen after an EXPERIENCE (or a sustained
  non-GAME gap)", not merely "GAME after any non-GAME".
- **Experience screen shorter than 5 s.** If the player advances past the summary
  quickly, the 5 s capture never fires. Mitigation: also fire a best-effort capture
  on the EXPERIENCE→(non-EXPERIENCE) transition if not yet captured.
- **Match with no experience screen** (player dashboards/leaves). The replay rows
  exist but no experience row — fine, but the *next* GAME must still mint a new
  `game_id`. The "new game after sustained non-GAME gap" rule covers this.
- **Program starts on the summary screen** (no preceding GAME this session). No
  `game_id` exists at capture time → mint an "orphan" `game_id` so the row is still
  keyed.
- **Duplicate experience rows.** Guard with a per-`game_id` `experience_captured`
  flag so re-entering EXPERIENCE (or flicker) writes at most one row.
- **Warm-up nulls in replay.** Early GAME frames read mostly `None`. Rows are
  **suppressed** until ≥1 stat field is non-null.
- **Game-start-time accuracy.** First GAME-confirmed frame lags real match start
  (drop ship, classify interval). Acceptable v1; later refine with a match-start
  banner detector.
- **Restart mid-match.** In-memory `game_id` is lost on restart; the post-restart
  GAME mints a fresh id and the experience screen associates to it (acceptable).
  Optional: persist the active session to a tiny JSON for crash-recovery.

### Inefficiencies (fixable as part of this work)
- **Experience OCR runs every EXPERIENCE frame** today (cached after first). New
  design runs it **once** at the 5 s mark → less CPU, and removes reliance on the
  internal cache for correctness.
- **HUD detectors run during UNKNOWN** as well as GAME. Replay only needs GAME; we
  can skip detector work (or at least replay writes) outside GAME.
- **Classifier cadence vs capture cadence** are independent; ensure the 5 s timers
  are driven by wall/monotonic time, not frame counts (frame rate varies).

### Data / schema concerns
- **Schema versioning.** Appending to an existing CSV whose header differs from the
  current code silently misaligns columns. Add a header-compatibility check (or a
  schema version column) and fail loudly / roll the file.
- **`results.csv` rename.** Rename to **`match_history.csv`** and add
  `game_id`, `game_start_time`, `captured_at`. Migrating old rows is out of scope.

---

## 4. Proposed Architecture

### New module: `session.py`
```python
class MatchPhase(enum.Enum):
    MENU = "menu"          # UNKNOWN / lobby
    GAME = "game"
    EXPERIENCE = "experience"

class GameSession:
    """Owns game_id, game_start_time, and match-boundary logic.

    Driven by confirmed screen types (post-debounce). Emits events the loop acts on:
      - on_game_start(game_id, game_start_time)
      - on_experience_ready()           # 5 s after experience first seen
      - exposes current game_id for writers
    """
    def __init__(self, clock=time.monotonic, now=time.time,
                 experience_delay_s=5.0, new_game_gap_s=20.0): ...
    def update(self, confirmed_phase) -> list[Event]: ...
    def should_capture_experience(self) -> bool: ...   # delay elapsed & not captured
    def mark_experience_captured(self) -> None: ...
    @property
    def game_id(self) -> str | None: ...
    @property
    def game_start_time(self) -> str | None: ...
```
- `game_id`: `f"{YYYYMMDD-HHMMSS}-{short_random}"` (sortable + unique).
- **Debounce** lives just above `GameSession`: a small `PhaseDebouncer` that only
  reports a phase after K consecutive identical classifications.
- **New-game rule:** mint a new `game_id` when entering GAME and either (a) an
  EXPERIENCE was seen since the last game started, or (b) we've been in non-GAME for
  ≥ `new_game_gap_s`. Otherwise treat GAME re-entry as the same match.
- Inject `clock`/`now` so tests use a fake clock.

### New module: `sinks.py` (CSV writers)
```python
class CsvSink:
    def __init__(self, path, headers): ...   # writes header if file new/empty
    def write_row(self, mapping: dict): ...  # validates keys == headers

class ExperienceWriter(CsvSink):  # experience.csv schema (see §5)
    def write(self, values: dict, game_id, game_start_time, captured_at): ...

class MatchReplayWriter(CsvSink):  # match_replay.csv schema (see §5)
    def write(self, agg_state: dict, game_id, game_start_time,
              row_time, elapsed_s, map_info=None): ...
```
- Remove `_append_csv`, `_CSV_HEADERS`, `_CSV_FIELDS`, and the `RESULTS_CSV`
  call-site from `ExperienceDetector`; the detector returns values only.

### Changes to `detect-weapons.py`
- Instantiate `GameSession`, `PhaseDebouncer`, `ExperienceWriter`,
  `MatchReplayWriter`.
- Map `ScreenType` → `MatchPhase`; feed the **debounced** phase to `session.update()`.
- On `on_game_start`: `aggregator.reset()`, `exp_det.reset()`, reset the replay timer.
- **GAME branch:** when `monotonic() - last_replay >= REPLAY_INTERVAL_SEC`, snapshot
  the aggregated state under `result_lock`, write a `match_replay` row, update
  `last_replay`.
- **EXPERIENCE branch:** don't OCR until `session.should_capture_experience()` is
  true; then call `exp_det.detect()` **once**, `ExperienceWriter.write(..., game_id)`,
  `session.mark_experience_captured()`. Also do the best-effort capture on the
  EXPERIENCE→non-EXPERIENCE edge if still uncaptured.

### Future hook: `map_locator.py`
- `MapLocator.locate(full_bgr) -> {"map_x", "map_y", "location"} | Nones`.
- v1 candidates: template-match the player chevron on the always-on **minimap**
  (top-left ROI) and/or detect when the **full map** (TAB) is open for a more precise
  fix. Reserve the columns now; implement later.

---

## 5. Data Schemas

### `experience.csv` (one row per game)
Existing 14 columns **plus**:
| column | source |
|---|---|
| `game_id` | `GameSession.game_id` |
| `game_start_time` | `GameSession.game_start_time` (ISO 8601) |
| `captured_at` | wall-clock at capture (ISO 8601) |
| *(existing)* | current_rank, current_rp, point_change, placement, kills, assists, participation, base_combat_value, combat_rp_total, bonus_rp_total, challenger, promotion, placement_rp_total, cost_of_entry |

### `match_replay.csv` (one row every X s during GAME)
| column | source |
|---|---|
| `game_id` | session |
| `game_start_time` | session (ISO 8601) |
| `row_time` | wall-clock at row (ISO 8601) |
| `elapsed_s` | `monotonic() - game_start_monotonic` |
| `primary_weapon`, `secondary_weapon` | aggregator |
| `armor_level` | aggregator |
| `shield_type`, `shield_hp`, `flesh_hp`, `health` | aggregator |
| `squads_remaining`, `players_remaining` | aggregator |
| `kills`, `assists`, `participation`, `damage` | aggregator |
| `map_x`, `map_y`, `location` | **reserved** (null v1) → MapLocator later |

Join key for analysis: `experience.csv.game_id == match_replay.csv.game_id`.

---

## 6. Association Design (game_id lifecycle)

| Event (debounced) | Action |
|---|---|
| First GAME of session | mint `game_id`, set `game_start_*`, reset aggregator+exp_det+replay timer |
| GAME after EXPERIENCE | **new** match → mint new `game_id` (+ resets) |
| GAME after sustained non-GAME (≥ gap) | **new** match → mint new `game_id` |
| GAME after brief UNKNOWN flicker (< gap, no EXPERIENCE) | same match, keep `game_id` |
| each X s in GAME | write `match_replay` row with current `game_id` |
| First EXPERIENCE | start 5 s timer; **retain** the just-ended game's `game_id` |
| EXPERIENCE + 5 s (once) | write `experience` row with that `game_id` |
| EXPERIENCE start with no `game_id` (orphan) | mint an orphan `game_id`, still write |

This guarantees the experience row inherits the `game_id` of the game whose summary
it is, so the two CSVs join cleanly.

---

## 7. Implementation Phases

**Phase 1 — Decouple I/O from detection**
- Remove `_append_csv`/`_CSV_*`/`RESULTS_CSV` usage from `ExperienceDetector`;
  `detect()` returns values only. Update `inspect-experience.py` / tests accordingly.
- Add `sinks.py` with `CsvSink`, `ExperienceWriter`, `MatchReplayWriter`.
- Add config: `MATCH_HISTORY_CSV` (was `RESULTS_CSV`), `MATCH_REPLAY_CSV`,
  `REPLAY_INTERVAL_SEC=5` (CLI-overridable), `EXPERIENCE_CAPTURE_DELAY_SEC=5`,
  `PHASE_DEBOUNCE_FRAMES`, `NEW_GAME_GAP_SEC`.

**Phase 2 — Session + debounce**
- Add `session.py` (`MatchPhase`, `PhaseDebouncer`, `GameSession`) with injectable
  clock. Unit-test the state machine in isolation (no OpenCV).

**Phase 3 — Wire into the live loop**
- Map `ScreenType`→`MatchPhase`, debounce, drive `GameSession`.
- GAME branch: periodic `match_replay` writes from aggregated state.
- EXPERIENCE branch: 5 s-delayed single capture via `ExperienceWriter`; best-effort
  capture on exit edge; call `exp_det.reset()` on new game.

**Phase 4 — Map locator stub**
- Add `map_locator.py` returning Nones; reserve `map_x`, `map_y`, `location` columns.
  v1 implementation: minimap player-chevron template match (always available, no TAB needed).

**Phase 5 — Hardening**
- Schema-version/header check in `CsvSink`. Optional session persistence JSON.
- Optional: skip HUD detection entirely during UNKNOWN to save CPU.

---

## 8. Edge Cases Checklist (must be covered by tests)
- [ ] Two consecutive games → two experience rows, two distinct `game_id`s, replay
      rows correctly partitioned.
- [ ] Mid-game UNKNOWN flicker → no new `game_id`, no split.
- [ ] Experience screen < 5 s → best-effort capture still writes one row.
- [ ] Game with no experience screen → replay rows only; next game still new id.
- [ ] Program starts on EXPERIENCE → orphan `game_id`, one row.
- [ ] Re-enter EXPERIENCE / flicker → still only one experience row per game.
- [ ] Replay cadence holds at X s regardless of frame rate (monotonic-driven).
- [ ] `game_start_time` identical across a game's replay rows and its experience row.

## 9. Testing Plan
- **Unit (no capture):** drive `GameSession` with a scripted phase sequence and a
  fake clock; assert events, `game_id` lifecycle, capture timing, dedupe.
- **Sink tests:** write/read round-trip; header created once; key validation; append
  across "sessions"; schema-version guard.
- **Integration (offline):** synthesize a phase timeline + canned aggregator/exp
  values through the loop's session logic (factored into a testable function) and
  assert the two CSVs join on `game_id` with the expected row counts.

## 10. Decisions (resolved)

| Question | Decision |
|---|---|
| Replay interval | Default **5 s**, CLI-configurable (`--replay-interval N`) |
| Experience CSV name | Rename `results.csv` → **`match_history.csv`** |
| Warm-up replay rows | **Suppress** rows until ≥1 stat field is non-null |
| Map position v1 | **Minimap player-chevron** template match (always available) |
| Session persistence across restart | Acceptable to lose active game on crash — no persistence needed |

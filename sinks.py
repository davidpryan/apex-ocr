"""
CSV output sinks for the two match outputs.

CsvSink         — base class: header guard, schema-version check, write_row
ExperienceWriter — match_history.csv, one row per game (written 5 s after summary)
MatchReplayWriter — match_replay.csv, one row every X s during GAME
"""

import csv
import datetime
import os

from config import MATCH_HISTORY_CSV, MATCH_REPLAY_CSV

# Bump this string whenever a column is added or renamed.  On open, CsvSink
# compares it against the first row of the file and renames the stale file
# rather than silently mis-aligning columns.
_SCHEMA_VERSION = "v2"


class CsvSink:
    """Append-only CSV writer with header creation and schema-version guard.

    On first write to a new/empty file the header row is written automatically.
    If the file exists but its header doesn't match (stale schema), the old file
    is renamed to <name>.bak.<timestamp> and a fresh file is started.
    """

    def __init__(self, path: str, headers: list[str]):
        self._path    = path
        self._headers = headers
        self._checked = False   # deferred: check on first actual write

    def _ensure_ready(self) -> None:
        if self._checked:
            return
        self._checked = True

        if not os.path.exists(self._path) or os.path.getsize(self._path) == 0:
            # New file — write header now so it's present even if no rows follow.
            with open(self._path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self._headers).writeheader()
            return

        # Existing file — read first row and validate schema.
        with open(self._path, newline="") as fh:
            reader = csv.reader(fh)
            try:
                existing_headers = next(reader)
            except StopIteration:
                existing_headers = []

        if existing_headers != self._headers:
            ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bak = f"{self._path}.bak.{ts}"
            os.rename(self._path, bak)
            print(f"[sinks] Schema changed — old file archived to {bak}")
            with open(self._path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self._headers).writeheader()

    def write_row(self, mapping: dict) -> None:
        self._ensure_ready()
        # Fill missing keys with None rather than raising.
        row = {h: mapping.get(h) for h in self._headers}
        with open(self._path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self._headers).writerow(row)


# ---------------------------------------------------------------------------
# Field mappings: internal result-dict key → CSV column name
# ---------------------------------------------------------------------------

# Keys inside ExperienceDetector.detect() result dict → column names
_EXP_FIELD_MAP = {
    "current_rank":       "current_rank",
    "current_rp":         "current_rp",
    "point_change":       "point_change",
    "placement":          "placement",
    "kills":              "kills",
    "assists":            "assists",
    "participations_rp":  "participation",
    "base_combat_value":  "base_combat_value",
    "combat_rp_total":    "combat_rp_total",
    "bonus_rp_total":     "bonus_rp_total",
    "challenger_count":   "challenger",
    "promotion_rp":       "promotion",
    "placement_rp_total": "placement_rp_total",
    "cost_of_entry_rp":   "cost_of_entry",
}

# Canonical base-rank name → CSV column name for squad distribution
_SQUAD_DIST_COLS: dict[str, str] = {
    "Rookie":        "squads_rookie",
    "Bronze":        "squads_bronze",
    "Silver":        "squads_silver",
    "Gold":          "squads_gold",
    "Platinum":      "squads_platinum",
    "Diamond":       "squads_diamond",
    "Master":        "squads_master",
    "Apex Predator": "squads_predator",
}

_EXP_HEADERS = [
    "game_id", "game_start_time", "map_name",
    "squads_rookie", "squads_bronze", "squads_silver", "squads_gold",
    "squads_platinum", "squads_diamond", "squads_master", "squads_predator",
    "captured_at",
    "current_rank", "current_rp", "point_change", "placement",
    "kills", "assists", "participation", "base_combat_value",
    "combat_rp_total", "bonus_rp_total", "challenger",
    "promotion", "placement_rp_total", "cost_of_entry",
]

_REPLAY_HEADERS = [
    "game_id", "game_start_time", "map_name", "row_time", "elapsed_s",
    "primary_weapon", "secondary_weapon", "armor_level",
    "shield_type", "shield_hp", "flesh_hp", "health",
    "squads_remaining", "players_remaining",
    "kills", "assists", "participation", "damage",
    "map_x", "map_y", "location",
]

# Stat fields that count as "populated" for warm-up suppression.
_REPLAY_STAT_FIELDS = {
    "primary_weapon", "secondary_weapon", "armor_level",
    "shield_type", "shield_hp", "squads_remaining", "kills", "damage",
}


class ExperienceWriter(CsvSink):
    """Writes one row to match_history.csv per completed game."""

    def __init__(self, path: str = MATCH_HISTORY_CSV):
        super().__init__(path, _EXP_HEADERS)

    def write(self, values: dict, game_id: str,
              game_start_time: str, captured_at: str,
              map_name: str | None = None,
              squad_dist: dict | None = None) -> None:
        row = {"game_id": game_id,
               "game_start_time": game_start_time,
               "map_name": map_name,
               "captured_at": captured_at}
        for src_key, col in _EXP_FIELD_MAP.items():
            row[col] = values.get(src_key)
        for rank, col in _SQUAD_DIST_COLS.items():
            row[col] = (squad_dist or {}).get(rank)
        self.write_row(row)


class MatchReplayWriter(CsvSink):
    """Writes one row to match_replay.csv every X seconds during GAME.

    Rows are suppressed until at least one stat field is non-null (warm-up).
    """

    def __init__(self, path: str = MATCH_REPLAY_CSV):
        super().__init__(path, _REPLAY_HEADERS)

    def write(self, agg_state: dict, game_id: str, game_start_time: str,
              row_time: str, elapsed_s: float,
              map_name: str | None = None,
              map_info: dict | None = None) -> bool:
        """Build and write a replay row.  Returns True if the row was written,
        False if it was suppressed (all stats still None / warm-up)."""

        weapon  = agg_state.get("weapon") or {}
        armor   = agg_state.get("armor")  or {}
        shield  = agg_state.get("shield") or {}
        tr      = agg_state.get("tr")     or {}
        map_inf = map_info or {}

        def _val(entry):
            return entry[0] if entry else None

        row = {
            "game_id":            game_id,
            "game_start_time":    game_start_time,
            "map_name":           map_name,
            "row_time":           row_time,
            "elapsed_s":          round(elapsed_s, 1),
            "primary_weapon":     _val(weapon.get("primary")),
            "secondary_weapon":   _val(weapon.get("secondary")),
            "armor_level":        armor.get("number"),
            "shield_type":        shield.get("shield_type"),
            "shield_hp":          shield.get("shield_hp"),
            "flesh_hp":           shield.get("flesh_hp"),
            "health":             shield.get("health"),
            "squads_remaining":   _val(tr.get("squads_remaining")),
            "players_remaining":  _val(tr.get("players_remaining")),
            "kills":              _val(tr.get("kills")),
            "assists":            _val(tr.get("assists")),
            "participation":      _val(tr.get("participation")),
            "damage":             _val(tr.get("damage")),
            "map_x":              map_inf.get("map_x"),
            "map_y":              map_inf.get("map_y"),
            "location":           map_inf.get("location"),
        }

        # Suppress warm-up rows where no meaningful stat has been read yet.
        if not any(row.get(f) is not None for f in _REPLAY_STAT_FIELDS):
            return False

        self.write_row(row)
        return True

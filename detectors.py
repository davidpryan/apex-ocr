"""
Detector classes, one per HUD region or screen type:

  ScreenClassifier  — decides whether the current frame is GAME or EXPERIENCE
  WeaponDetector    — reads primary / secondary weapon names from the weapon bar
  ArmorDetector     — reads the armor-level digit (1/2/3) from the triangle
  TopRightDetector  — reads squads, players, kills, assists, participation, damage
  ExperienceDetector — stub for the post-match summary screen
"""

import enum
import os
import re
from collections import Counter, deque

import cv2
import easyocr
import numpy as np

import difflib

from config import (
    EXPERIENCE_TEMPLATE, SCREEN_CLASSIFY_THRESHOLD, SCREEN_CLASSIFY_SEARCH_FRAC,
    RANKED_LOADING_TEMPLATE, RANKED_LOADING_THRESHOLD, RANKED_LOADING_SEARCH_FRAC,
    RANKED_LOADING_REF_WIDTH,
    RANKED_LOADING_MAP_Y1, RANKED_LOADING_MAP_Y2,
    RANKED_LOADING_MAP_X1, RANKED_LOADING_MAP_X2,
    APEX_MAP_NAMES,
    EXPERIENCE_NEXT_RANK_TOP,    EXPERIENCE_NEXT_RANK_BOT,
    EXPERIENCE_POINT_CHANGE_TOP, EXPERIENCE_POINT_CHANGE_BOT,
    EXPERIENCE_POINT_CHANGE_X1,  EXPERIENCE_POINT_CHANGE_X2,
    EXPERIENCE_CURRENT_RP_TOP,   EXPERIENCE_CURRENT_RP_BOT,
    EXPERIENCE_CURRENT_RP_X1,    EXPERIENCE_CURRENT_RP_X2,
    EXPERIENCE_SECTIONS_SCAN_TOP, EXPERIENCE_SECTIONS_SCAN_BOT,
    EXPERIENCE_BIG_ROW_GAP, EXPERIENCE_SMALL_ROW_GAP, EXPERIENCE_ROW_HALF_H,
    EXPERIENCE_COMBAT_COUNT_X, EXPERIENCE_COMBAT_RP_X,
    EXPERIENCE_BONUSES_COUNT_X, EXPERIENCE_BONUSES_RP_X,
    EXPERIENCE_PLACEMENT_TEXT_X, EXPERIENCE_PLACEMENT_RP_X,
    RANK_PROGRESSION, RANK_LOOKUP,
    SHIELD_BAR_X1, SHIELD_BAR_X2,
    SHIELD_STRIP_Y1, SHIELD_STRIP_Y2, SHIELD_BAR_MID_Y,
    WEAPON_UPSCALE, MIN_CONFIDENCE, WEAPON_NAMES, WEAPON_TOKENS,
    ARMOR_UPSCALE, ARMOR_MIN_CONF,
    TR_UPSCALE, TR_MIN_CONF,
    TR_BADGE_CUTOFF_FRAC,
    TR_SQUADS_ROW_TOP, TR_SQUADS_ROW_BOT,
    TR_STATS_ROW_TOP,  TR_STATS_ROW_BOT,
    TR_ROW_Y_OFFSET,
    ICON_DIR, ICON_REF_PANEL_H, ICON_MATCH_THRESH, ICON_NUMBER_WINDOW,
    RANKED_DIST_ICON_DIR,
    RANKED_DIST_SEARCH_Y1, RANKED_DIST_SEARCH_Y2,
    RANKED_DIST_SEARCH_X1, RANKED_DIST_SEARCH_X2,
    RANKED_DIST_FILL_V_MIN, RANKED_DIST_FILL_S_MIN,
    RANKED_DIST_COL_THRESH, RANKED_DIST_SEG_MIN_PX,
    RANKED_DIST_ICON_SEARCH_PX, RANKED_DIST_ICON_THRESH,
    RANKED_SQUAD_TOTAL,
)
from preprocessing import enhance_contrast, sharpen


# ---------------------------------------------------------------------------
# Weapon detector
# ---------------------------------------------------------------------------

class WeaponDetector:
    """Detects primary and secondary weapon names from the weapon-bar crop."""

    def __init__(self, reader: easyocr.Reader):
        self._reader = reader

    # ------------------------------------------------------------------

    @staticmethod
    def _matches_weapon(text: str) -> bool:
        """True if text plausibly names a weapon.
        Exact token match first (fast path); then substring match to catch
        partial reads caused by ROI clipping (e.g. 'WINGMA' ⊂ 'wingman')."""
        normalized = text.lower().replace("-", "").replace(".", "").replace(" ", "")
        if len(normalized) < 3:
            return False
        tokens = text.lower().replace("-", " ").replace(".", "").split()
        if any(t in WEAPON_TOKENS for t in tokens):
            return True
        return any(
            len(normalized) >= 4
            and normalized in name.lower().replace("-", "").replace(".", "").replace(" ", "")
            for name in WEAPON_NAMES
        )

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """Returns {"primary": (text, box) | None, "secondary": (text, box) | None}."""
        prepped    = sharpen(enhance_contrast(frame_bgr))
        big        = cv2.resize(prepped, None, fx=WEAPON_UPSCALE, fy=WEAPON_UPSCALE,
                                interpolation=cv2.INTER_CUBIC)
        midpoint_x = frame_bgr.shape[1] // 2

        primary = secondary = None

        for bbox, text, conf in self._reader.readtext(big):
            if conf < MIN_CONFIDENCE:
                continue
            tl, _, br, _ = bbox
            x1 = int(tl[0] / WEAPON_UPSCALE)
            y1 = int(tl[1] / WEAPON_UPSCALE)
            x2 = int(br[0] / WEAPON_UPSCALE)
            y2 = int(br[1] / WEAPON_UPSCALE)
            text = text.strip()
            if text.replace(" ", "").isnumeric() or not self._matches_weapon(text):
                continue
            text = text.upper()
            cx = (x1 + x2) // 2
            if cx < midpoint_x:
                if primary is None or conf > primary[2]:
                    primary = (text, (x1, y1, x2, y2), conf)
            else:
                if secondary is None or conf > secondary[2]:
                    secondary = (text, (x1, y1, x2, y2), conf)

        return {
            "primary":   (primary[0],   primary[1])   if primary   else None,
            "secondary": (secondary[0], secondary[1]) if secondary else None,
        }


# ---------------------------------------------------------------------------
# Shield detector
# ---------------------------------------------------------------------------

class ShieldDetector:
    """Detects the player's armor shield type and current HP from the bottom-left HUD.

    The player card shows two horizontal bars to the right of the portrait:
      - Shield bar (upper): white=2 segs/50HP, blue=3 segs/75HP, purple=4 segs/100HP
      - Health bar (lower): always white, used as full-width reference

    Strategy:
      1. Split the search band at SHIELD_BAR_MID_Y — shield strip above, health below.
      2. Find the health bar x-extent (white, usually full) → total bar width.
      3. Score each shield colour in the shield strip within that x range.
      4. Measure the rightmost filled column to get fill percentage.
      5. shield_hp = round(fill_pct * max_hp).
    """

    _SPECS = {
        "purple": (100, 4),
        "blue":   ( 75, 3),
        "white":  ( 50, 2),
    }

    # HSV bounds: (lo, hi) each a 3-tuple
    _COLOUR_RANGES = [
        ("purple", (125, 50,  50), (165, 255, 255)),
        ("blue",   ( 90, 80,  60), (130, 255, 255)),
        ("white",  (  0,  0, 155), (180,  55, 255)),
    ]

    def detect(self, full_bgr: np.ndarray) -> dict | None:
        """Return shield/health readings, or None if no bars found.

        Keys: shield_type, shield_hp, flesh_hp, health.
        """
        h, w = full_bgr.shape[:2]

        x1 = int(w * SHIELD_BAR_X1)
        x2 = int(w * SHIELD_BAR_X2)
        mid_y = int(h * SHIELD_BAR_MID_Y)
        top_y = int(h * SHIELD_STRIP_Y1)
        bot_y = int(h * SHIELD_STRIP_Y2)

        shield_roi = full_bgr[top_y:mid_y, x1:x2]
        health_roi = full_bgr[mid_y:bot_y, x1:x2]
        rw = x2 - x1  # roi width

        # ── Health bar ───────────────────────────────────────────────────────
        # bar_left: where the bar starts (right of portrait icon) — consistent
        #           regardless of fill level.
        # bar_right_full: the rightmost pixel when health is 100 % — observed
        #           to be roi_width - 1 across all reference screens (gap = 1 px).
        # Using this fixed right anchor means bar_width never shrinks with damage,
        # so shield fill fractions stay correct even when flesh is not full.
        h_hsv   = cv2.cvtColor(health_roi, cv2.COLOR_BGR2HSV)
        h_white = cv2.inRange(h_hsv,
                              np.array([0, 0, 155]), np.array([180, 55, 255]))
        h_thresh = health_roi.shape[0] * 255 * 0.25
        h_cols   = np.where(h_white.sum(axis=0) >= h_thresh)[0]

        if not h_cols.size:
            return None

        bar_left       = int(h_cols.min())
        bar_right_full = rw - 1          # fixed full-bar right edge
        bar_width      = bar_right_full - bar_left
        if bar_width < 20:
            return None

        # Flesh fill: rightmost white column that clears the row threshold
        h_fill_right = int(h_cols.max())
        flesh_fill   = min(1.0, (h_fill_right - bar_left + 1) / bar_width)
        flesh_hp     = round(flesh_fill * 100)

        # ── Shield bar: score each colour in the bar x-range ────────────────
        s_hsv = cv2.cvtColor(shield_roi, cv2.COLOR_BGR2HSV)
        sh    = shield_roi.shape[0]

        masks  = {}
        scores = {}
        for name, lo, hi in self._COLOUR_RANGES:
            m = cv2.inRange(s_hsv, np.array(lo), np.array(hi))
            masks[name]  = m
            scores[name] = int(m[:, bar_left:bar_right_full].sum())

        shield_type = max(scores, key=scores.get)

        # Require at least 8 % of what a fully-filled bar would score
        min_score = bar_width * sh * 255 * 0.08
        if scores[shield_type] < min_score:
            return {
                "shield_type": "none",
                "shield_hp":   0,
                "flesh_hp":    flesh_hp,
                "health":      flesh_hp,
            }

        # ── Shield fill: rightmost consistently coloured column ──────────────
        mask       = masks[shield_type]
        col_scores = mask[:, bar_left:bar_right_full + 1].sum(axis=0)
        col_thresh = sh * 255 * 0.15
        filled     = np.where(col_scores >= col_thresh)[0]

        if not filled.size:
            return {
                "shield_type": shield_type,
                "shield_hp":   0,
                "flesh_hp":    flesh_hp,
                "health":      flesh_hp,
            }

        max_hp   = self._SPECS[shield_type][0]
        fill_pct = min(1.0, (filled.max() + 1) / bar_width)
        shield_hp = round(fill_pct * max_hp)

        # Sanity check: shields absorb damage before health in Apex.
        # If health is already depleted (>15%), shields must be 0.
        # A sub-15% shield reading alongside depleted health is detection noise.
        if flesh_fill < 0.85 and fill_pct < 0.15:
            shield_hp   = 0
            shield_type = "none"

        health = shield_hp + flesh_hp

        return {
            "shield_type": shield_type,
            "shield_hp":   shield_hp,
            "flesh_hp":    flesh_hp,
            "health":      health,
        }


# ---------------------------------------------------------------------------
# Armor detector
# ---------------------------------------------------------------------------

class ArmorDetector:
    """Detects the armor level (1, 2, or 3) from the bottom-left triangle."""

    def __init__(self, reader: easyocr.Reader):
        self._reader = reader

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """Returns {"number": str | None, "box": tuple | None}."""
        big  = cv2.resize(
            sharpen(enhance_contrast(frame_bgr)),
            None, fx=ARMOR_UPSCALE, fy=ARMOR_UPSCALE, interpolation=cv2.INTER_CUBIC,
        )
        best = None
        for bbox, text, conf in self._reader.readtext(big, allowlist="123", paragraph=False):
            text = text.strip()
            if len(text) != 1 or text not in "123" or conf < ARMOR_MIN_CONF:
                continue
            if best is None or conf > best[2]:
                tl, _, br, _ = bbox
                x1, y1 = int(tl[0] / ARMOR_UPSCALE), int(tl[1] / ARMOR_UPSCALE)
                x2, y2 = int(br[0] / ARMOR_UPSCALE), int(br[1] / ARMOR_UPSCALE)
                best = (text, (x1, y1, x2, y2), conf)

        return {"number": best[0], "box": best[1]} if best else {"number": None, "box": None}


# ---------------------------------------------------------------------------
# Top-right stats detector
# ---------------------------------------------------------------------------

class TopRightDetector:
    """Detects squads remaining, players remaining, kills, assists,
    participation, and damage from the top-right HUD panel."""

    _STAT_KEYS = ("kills", "assists", "participation", "damage")

    def __init__(self, reader: easyocr.Reader, icon_dir: str = ICON_DIR):
        self._reader = reader
        self._icons  = self._load_icons(icon_dir)

    # ------------------------------------------------------------------
    # Icon loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_icons(icon_dir: str) -> dict[str, np.ndarray]:
        icons = {}
        for stat in ("kills", "assists", "participation", "damage"):
            path = os.path.join(icon_dir, f"{stat}_raw.png")
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                icons[stat] = img
                print(f"  icon loaded: {stat} ({img.shape[1]}x{img.shape[0]}px)")
            else:
                print(f"  icon missing: {path}")
        return icons

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prep(crop: np.ndarray) -> np.ndarray:
        """Sharpen then upscale.  No CLAHE — high-contrast text degrades with it.
        Used for icon template matching (keeps full tonal detail)."""
        return cv2.resize(sharpen(crop), None, fx=TR_UPSCALE, fy=TR_UPSCALE,
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _prep_ocr(crop: np.ndarray) -> np.ndarray:
        """Binarize then upscale for digit OCR.

        HUD text is near-white; a brightness threshold isolates it from the
        background so OCR works whether the panel sits on bright sky or dark
        terrain.  This is the key fix for dark-background game screens."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        return cv2.resize(binary, None, fx=TR_UPSCALE, fy=TR_UPSCALE,
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _tall_enough(bbox, img_h: int, min_frac: float = 0.30) -> bool:
        """True if an OCR box is at least min_frac of the row height.
        Filters out the small FPS/ping debug text that bleeds into the band."""
        box_h = bbox[2][1] - bbox[0][1]
        return box_h >= img_h * min_frac

    def _match_icons(self, st_big: np.ndarray, ph: int) -> dict[str, float]:
        """Return {stat: icon_right_x} for icons matched in the upscaled stats row."""
        if not self._icons:
            return {}
        st_gray    = cv2.cvtColor(st_big, cv2.COLOR_BGR2GRAY)
        icon_scale = (ph / ICON_REF_PANEL_H) * TR_UPSCALE
        matched    = {}
        for stat, icon_gray in self._icons.items():
            tmpl = cv2.resize(icon_gray, None, fx=icon_scale, fy=icon_scale,
                              interpolation=cv2.INTER_CUBIC)
            if tmpl.shape[0] >= st_gray.shape[0] or tmpl.shape[1] >= st_gray.shape[1]:
                continue
            _, score, _, loc = cv2.minMaxLoc(
                cv2.matchTemplate(st_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            )
            if score >= ICON_MATCH_THRESH:
                matched[stat] = loc[0] + tmpl.shape[1]
        return matched

    @staticmethod
    def _parse_box(tl, br, upscale: int, y_offset: int) -> tuple[int, int, int, int]:
        return (
            int(tl[0] / upscale),
            int(tl[1] / upscale) + y_offset,
            int(br[0] / upscale),
            int(br[1] / upscale) + y_offset,
        )

    def _ocr_digits(self, band_bgr: np.ndarray, y_offset: int) -> list[tuple]:
        """OCR the stats band; return [(cx_upscaled, digits, box_panel, conf)].

        Runs OCR on both the tonal (sharpened) and binarized renders and merges
        the results: the tonal pass keeps bright-background screens accurate, the
        binarized pass recovers digits on dark backgrounds.  Near-duplicate hits
        (same position) are collapsed, keeping the higher-confidence read.
        """
        merged: list[tuple] = []
        for big in (self._prep(band_bgr), self._prep_ocr(band_bgr)):
            img_h = big.shape[0]
            # No allowlist: lets the white stat icons read as letters (then stripped
            # by re.sub) instead of being forced into spurious digits like the
            # airplane→"4" that would corrupt the damage value.
            for bbox, text, conf in self._reader.readtext(big):
                if conf < TR_MIN_CONF or not self._tall_enough(bbox, img_h, 0.20):
                    continue
                digits = re.sub(r"\D", "", text)
                if not digits:
                    continue
                tl, _, br, _ = bbox
                cx  = (tl[0] + br[0]) / 2
                box = self._parse_box(tl, br, TR_UPSCALE, y_offset)
                # The tonal pass runs first and is trusted on bright screens; the
                # binary pass only *fills* positions the tonal pass missed (dark
                # screens).  So add a hit only when no existing hit is nearby.
                if not any(abs(c[0] - cx) < 40 for c in merged):
                    merged.append((cx, digits, box, conf))
        return [(cx, digits, box) for cx, digits, box, _ in merged]

    def _assign_stats(
        self,
        sorted_candidates: list[tuple],
        icon_right_x: dict[str, float],
    ) -> dict:
        result = {k: None for k in self._STAT_KEYS}

        if len(icon_right_x) >= 2:
            used: set = set()
            for stat in self._STAT_KEYS:
                if stat not in icon_right_x:
                    continue
                lo, hi = icon_right_x[stat], icon_right_x[stat] + ICON_NUMBER_WINDOW
                for cand in sorted_candidates:
                    if id(cand) in used:
                        continue
                    cx, digits, box = cand
                    if lo <= cx <= hi:
                        result[stat] = (digits, box)
                        used.add(id(cand))
                        break
            # Positional fallback for stats whose icon wasn't matched
            remaining = [c for c in sorted_candidates if id(c) not in used]
            for stat in self._STAT_KEYS:
                if stat not in icon_right_x and remaining:
                    _, digits, box = remaining.pop(0)
                    result[stat] = (digits, box)
        else:
            for i, (_, val, box) in enumerate(sorted_candidates):
                if i < len(self._STAT_KEYS):
                    result[self._STAT_KEYS[i]] = (val, box)

        return result

    # ------------------------------------------------------------------
    # Public detection
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """Returns a dict keyed by squads_remaining, players_remaining, kills,
        assists, participation, damage.  Each value is (text, box) or None."""
        ph, pw   = frame_bgr.shape[:2]
        badge_x  = int(pw * TR_BADGE_CUTOFF_FRAC)
        sq_y1    = int(ph * TR_SQUADS_ROW_TOP)
        sq_y2    = int(ph * TR_SQUADS_ROW_BOT)
        st_y1    = int(ph * TR_STATS_ROW_TOP)
        st_y2    = int(ph * TR_STATS_ROW_BOT)

        result = {k: None for k in
                  ("squads_remaining", "players_remaining",
                   *self._STAT_KEYS)}

        # --- Row 1: squads + players ---
        # Binarized OCR *without* a digit allowlist so the "SQUADS LEFT" letters
        # stay attached to the count as one token ("16 SQUADS LEFT") instead of
        # merging into a digit blob.  squads = number before SQUADS (OCR-tolerant
        # regex); players = the rightmost pure-digit token (before the badge).
        sq_band = frame_bgr[sq_y1:sq_y2, :badge_x]
        sq_big  = self._prep_ocr(sq_band)
        sq_h    = sq_big.shape[0]
        squads_box = players_box = None

        sq_hits = []
        for bbox, text, conf in self._reader.readtext(sq_big):
            if conf < TR_MIN_CONF or not self._tall_enough(bbox, sq_h):
                continue
            tl, _, br, _ = bbox
            sq_hits.append(((tl[0] + br[0]) / 2, text.strip(),
                            self._parse_box(tl, br, TR_UPSCALE, sq_y1)))
        sq_hits.sort(key=lambda c: c[0])

        # squads: tolerant "<n> SQUADS" match (O↔0, common OCR letter swaps)
        for _, text, box in sq_hits:
            m = re.search(r"(\d+)\s*S?QUA[D0O]S", text.upper())
            if m:
                result["squads_remaining"] = (m.group(1), box)
                squads_box = box
                break

        # players: rightmost token that is purely digits (the count after the icon)
        for cx, text, box in reversed(sq_hits):
            if re.fullmatch(r"\d+", text):
                result["players_remaining"] = (text, box)
                players_box = box
                break

        # --- Row 2: stats (Y-anchored from squads/players if available) ---
        row_half_h  = int(ph * (TR_STATS_ROW_BOT - TR_STATS_ROW_TOP) / 2)
        anchor_box  = squads_box or players_box
        if anchor_box:
            sq_cy  = (anchor_box[1] + anchor_box[3]) / 2
            cy     = int(sq_cy + ph * TR_ROW_Y_OFFSET)
            st_y1  = max(0,  cy - row_half_h)
            st_y2  = min(ph, cy + row_half_h)

        st_band         = frame_bgr[st_y1:st_y2, :badge_x]
        candidates      = self._ocr_digits(st_band, st_y1)            # tonal + binarized merge
        sorted_cands    = sorted(candidates, key=lambda c: c[0])
        icon_right_x    = self._match_icons(self._prep(st_band), ph)  # tonal for icons

        result.update(self._assign_stats(sorted_cands, icon_right_x))

        # Damage cleanup: when the airplane icon can't be windowed out (dark
        # screens, no icon match) it reads as a leading digit, e.g. "41144".
        # Damage is ≤4 digits, so drop leading digits until it fits.
        dmg = result.get("damage")
        if dmg and len(dmg[0]) > 4:
            result["damage"] = (dmg[0][-4:], dmg[1])

        # kills / assists / participation cannot exceed 59 (60-player lobby).
        for key in ("kills", "assists", "participation"):
            entry = result.get(key)
            if entry and int(entry[0]) >= 60:
                result[key] = None

        # Sanity: kills + assists + participation must total < 60.
        # A higher sum means at least one field has an OCR error; discard all
        # three rather than silently propagating a bad number.
        kap = [result.get(k) for k in ("kills", "assists", "participation")]
        if sum(int(v[0]) for v in kap if v is not None) >= 60:
            result["kills"] = result["assists"] = result["participation"] = None

        return result


# ---------------------------------------------------------------------------
# Temporal aggregation for the game HUD
# ---------------------------------------------------------------------------

class _FieldVoter:
    """Windowed majority vote with persistence for a single field.

    Each frame pushes one observation (value may be None for a miss). The emitted
    value is the most frequent non-None observation in the rolling window, provided
    it has at least ``min_votes`` support; otherwise the last emitted value persists.

    This rejects single-frame OCR noise (a wrong read seen once never wins) and
    recovers fields that only read correctly on some frames (a value seen on a few
    good frames out of the window is emitted), while holding steady through dropouts.
    """

    __slots__ = ("_obs", "_min_votes", "current")

    def __init__(self, window: int, min_votes: int):
        self._obs = deque(maxlen=window)
        self._min_votes = min_votes
        self.current: tuple | None = None      # (value, box) or None

    def push(self, value, box) -> tuple | None:
        self._obs.append((value, box))
        counts = Counter(v for v, _ in self._obs if v is not None)
        if counts:
            val, n = counts.most_common(1)[0]
            if n >= self._min_votes:
                box_for = next((b for v, b in reversed(self._obs) if v == val), None)
                self.current = (val, box_for)
        return self.current


class GameAggregator:
    """Temporally stabilizes every game-HUD field across video frames.

    Wraps the per-frame output of the game-section detectors (weapons, armor,
    shield, top-right) and returns consensus-stabilized values in the same dict
    shapes the detectors produce, so it is a drop-in replacement for the old
    "keep last non-None" logic.

    Usage::

        agg = GameAggregator()
        stable = agg.update(weapon=weapon_res, armor=armor_res,
                            shield=shield_res, tr=tr_res)
        # stable == {"weapon": {...}, "armor": {...}, "shield": {...}, "tr": {...}}

    A single-frame wrong read needs ``min_votes`` repetitions before it is emitted,
    so transient noise is filtered; genuinely changing values (damage, HP, squads)
    are followed with a lag of at most ``min_votes`` frames.  Call :meth:`reset`
    at the start of each match.
    """

    _WEAPON_SLOTS = ("primary", "secondary")
    _SHIELD_KEYS  = ("shield_type", "shield_hp", "flesh_hp", "health")
    _TR_KEYS      = ("squads_remaining", "players_remaining", "kills",
                     "assists", "participation", "damage")

    def __init__(self, window: int = 10, min_votes: int = 2):
        self._window    = window
        self._min_votes = min_votes
        self._voters    = self._new_voters()

    def _new_voters(self) -> dict[str, _FieldVoter]:
        fields = (*self._WEAPON_SLOTS, "armor_number",
                  *self._SHIELD_KEYS, *self._TR_KEYS)
        return {f: _FieldVoter(self._window, self._min_votes) for f in fields}

    def reset(self) -> None:
        """Clear all history — call when a new match starts."""
        self._voters = self._new_voters()

    # ------------------------------------------------------------------

    def _push_pair(self, field: str, entry) -> None:
        """Push a detector (text, box) pair (or None) onto a field's voter."""
        if entry is None:
            self._voters[field].push(None, None)
        else:
            self._voters[field].push(entry[0], entry[1])

    def update(self, *, weapon=None, armor=None, shield=None, tr=None) -> dict:
        """Fold one frame of detector results in; return stabilized results.

        Any group omitted (None) counts as "not detected this frame" — every field
        in it observes None, so its stored value persists but ages out of the window.
        """
        weapon = weapon or {}
        for slot in self._WEAPON_SLOTS:
            self._push_pair(slot, weapon.get(slot))

        armor = armor or {}
        self._voters["armor_number"].push(armor.get("number"), armor.get("box"))

        shield = shield or {}
        for key in self._SHIELD_KEYS:
            self._voters[key].push(shield.get(key), None)

        tr = tr or {}
        for key in self._TR_KEYS:
            self._push_pair(key, tr.get(key))

        return self._build()

    def _build(self) -> dict:
        def cur(field):
            return self._voters[field].current        # (value, box) or None

        weapon = {slot: cur(slot) for slot in self._WEAPON_SLOTS}

        armor_cur = cur("armor_number")
        armor = {"number": armor_cur[0] if armor_cur else None,
                 "box":    armor_cur[1] if armor_cur else None}

        shield_vals = {k: (cur(k)[0] if cur(k) else None) for k in self._SHIELD_KEYS}
        shield = shield_vals if shield_vals["shield_type"] is not None else None

        tr = {key: cur(key) for key in self._TR_KEYS}

        return {"weapon": weapon, "armor": armor, "shield": shield, "tr": tr}


# ---------------------------------------------------------------------------
# Screen type
# ---------------------------------------------------------------------------

class ScreenType(enum.Enum):
    GAME           = "game"           # in-match HUD — run weapon/armor/stats detectors
    EXPERIENCE     = "experience"     # post-match summary — run ExperienceDetector
    RANKED_LOADING = "ranked_loading" # ranked lobby / dropzone — run RankedLoadingDetector
    UNKNOWN        = "unknown"        # neither matched confidently; hold last state


# ---------------------------------------------------------------------------
# Ranked loading screen detector
# ---------------------------------------------------------------------------

# Normalised map-name lookup: "STORM POINT" → "Storm Point", etc.
_MAP_NAME_NORM = {
    re.sub(r"[^A-Z0-9 ]", "", n.upper()): n
    for n in APEX_MAP_NAMES
}

# Icon filename stem → canonical base-rank name (no sub-tier)
_RANK_ICON_FILES: dict[str, str] = {
    "bronze":        "Bronze",
    "silver":        "Silver",
    "gold":          "Gold",
    "platinum":      "Platinum",
    "diamond":       "Diamond",
    "master":        "Master",
    "apex_predator": "Apex Predator",
}


def _hue_to_base_rank(h: int, s: int) -> str | None:
    """Map median HSV hue + saturation of a bar segment to a base rank name."""
    if h <= 8 or h >= 172:          # red (wraps at 0) → Predator
        return "Apex Predator"
    if 125 <= h <= 142:             # purple → Master
        return "Master"
    if 95 <= h <= 115:              # blue → Diamond
        return "Diamond"
    if 84 <= h < 95:                # teal/cyan → Platinum (estimate, no reference)
        return "Platinum"
    if 18 <= h < 35:                # gold/yellow → Gold (estimate, no reference)
        return "Gold"
    if 8 < h < 18:                  # brown/orange → Bronze (estimate, no reference)
        return "Bronze"
    return None


class RankedLoadingDetector:
    """OCRs the map-name sub-banner on the ranked loading screen.

    Call ``detect_map_name(full_bgr)`` once per loading screen (after
    ``ScreenClassifier`` has confirmed ``RANKED_LOADING``).  Returns the
    canonical map name (e.g. "Storm Point") or None if the sub-banner is not
    readable.

    Map names are validated against ``APEX_MAP_NAMES`` using fuzzy matching so
    minor OCR errors ("ST0RM POINT", "BROKENMOON") are tolerated.
    """

    def __init__(self, reader: easyocr.Reader):
        self._reader = reader
        self._rank_icons = self._load_rank_icons()

    # ------------------------------------------------------------------
    # Icon loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rank_icons() -> dict[str, np.ndarray]:
        """Load rank icon PNGs (RGBA) as greyscale, composited over mid-grey."""
        icons: dict[str, np.ndarray] = {}
        for stem, canon in _RANK_ICON_FILES.items():
            path = os.path.join(RANKED_DIST_ICON_DIR, f"{stem}.png")
            img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  rank icon missing: {path}")
                continue
            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3:4].astype(np.float32) / 255.0
                bg    = np.full((*img.shape[:2], 3), 90, np.uint8)
                rgb   = (img[:, :, :3].astype(np.float32) * alpha
                         + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            else:
                rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            icons[canon] = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            print(f"  rank icon loaded: {canon} ({icons[canon].shape[1]}×{icons[canon].shape[0]}px)")
        return icons

    def detect_map_name(self, full_bgr: np.ndarray) -> str | None:
        h, w = full_bgr.shape[:2]
        y1 = int(h * RANKED_LOADING_MAP_Y1)
        y2 = int(h * RANKED_LOADING_MAP_Y2)
        x1 = int(w * RANKED_LOADING_MAP_X1)
        x2 = int(w * RANKED_LOADING_MAP_X2)
        crop = full_bgr[y1:y2, x1:x2]

        # Threshold to white text on dark background, upscale for OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        big = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        for _, text, conf in self._reader.readtext(big):
            if conf < 0.30:
                continue
            candidate = self._match(text)
            if candidate:
                return candidate
        return None

    @staticmethod
    def _match(ocr_text: str) -> str | None:
        norm = re.sub(r"[^A-Z0-9 ]", "", ocr_text.upper()).strip()
        if norm in _MAP_NAME_NORM:
            return _MAP_NAME_NORM[norm]
        close = difflib.get_close_matches(norm, _MAP_NAME_NORM.keys(), n=1, cutoff=0.72)
        return _MAP_NAME_NORM[close[0]] if close else None

    # ------------------------------------------------------------------
    # Squad rank distribution detection
    # ------------------------------------------------------------------

    def detect_squad_distribution(self, full_bgr: np.ndarray) -> dict[str, int] | None:
        """Detect the squad rank distribution bar and return {rank: count}.

        Returns None if the bar cannot be located.  Total across all ranks is
        always RANKED_SQUAD_TOTAL (20).  Rank names match RANK_PROGRESSION base
        names ("Diamond", "Master", "Apex Predator", …).
        """
        bar = self._find_dist_bar(full_bgr)
        if bar is None:
            return None
        bx1, by1, bx2, by2 = bar

        segments = self._segment_dist_bar(full_bgr, bx1, by1, bx2, by2)
        if not segments:
            return None

        icon_y1 = max(0, by1 - RANKED_DIST_ICON_SEARCH_PX)
        icon_y2 = by1
        for seg in segments:
            icon_rank = self._match_dist_icon(full_bgr, seg["cx"], icon_y1, icon_y2)
            if icon_rank:
                seg["rank"] = icon_rank
            seg["count_ocr"] = self._ocr_dist_count(full_bgr,
                                                     seg["x1"], seg["x2"], by1, by2)

        return self._reconcile_dist_counts(segments)

    def _find_dist_bar(self, full_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        h, w = full_bgr.shape[:2]
        sy1 = int(h * RANKED_DIST_SEARCH_Y1)
        sy2 = int(h * RANKED_DIST_SEARCH_Y2)
        sx1 = int(w * RANKED_DIST_SEARCH_X1)
        sx2 = int(w * RANKED_DIST_SEARCH_X2)
        region = full_bgr[sy1:sy2, sx1:sx2]
        hsv    = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        V, S   = hsv[:, :, 2], hsv[:, :, 1]

        # Bar fill: coloured (high S + V) OR silver (high V, low S)
        fill = ((V > RANKED_DIST_FILL_V_MIN) & (S > RANKED_DIST_FILL_S_MIN)) | \
               ((V > 190) & (S < 50))
        row_frac = fill.mean(axis=1)
        if row_frac.max() < 0.05:
            return None

        peak = int(np.argmax(row_frac))
        top = peak
        while top > 0 and row_frac[top - 1] > 0.08:
            top -= 1
        bot = peak
        while bot < len(row_frac) - 2 and row_frac[bot + 1] > 0.08:
            bot += 1

        col_frac = fill[top:bot + 1].mean(axis=0)
        bar_cols = np.where(col_frac > RANKED_DIST_COL_THRESH)[0]
        if not bar_cols.size:
            return None

        bx1 = sx1 + int(bar_cols.min())
        bx2 = sx1 + int(bar_cols.max())
        if bx2 - bx1 < 50:
            return None
        return bx1, sy1 + top, bx2, sy1 + bot

    def _segment_dist_bar(
        self, full_bgr: np.ndarray, bx1: int, by1: int, bx2: int, by2: int
    ) -> list[dict]:
        bar_hsv = cv2.cvtColor(full_bgr[by1:by2, bx1:bx2], cv2.COLOR_BGR2HSV)
        H, S, V = bar_hsv[:, :, 0], bar_hsv[:, :, 1], bar_hsv[:, :, 2]

        colored_fill = (V > RANKED_DIST_FILL_V_MIN) & (S > RANKED_DIST_FILL_S_MIN)
        silver_fill  = (V > 190) & (S < 50)
        fill_mask    = colored_fill | silver_fill

        # Column fill-fraction: drop below threshold → divider / outer border
        col_score = fill_mask.mean(axis=0)
        is_fill   = col_score > RANKED_DIST_COL_THRESH

        segments: list[dict] = []
        in_seg, seg_start = False, 0
        for i, f in enumerate(np.append(is_fill, False)):
            if f and not in_seg:
                in_seg, seg_start = True, i
            elif not f and in_seg:
                in_seg = False
                width = i - seg_start
                if width < RANKED_DIST_SEG_MIN_PX:
                    continue

                # Classify by median hue of coloured-fill pixels (white text excluded)
                seg_col = colored_fill[:, seg_start:i]
                seg_sil = silver_fill[:, seg_start:i]
                hues    = H[:, seg_start:i][seg_col]

                if hues.size >= 10:
                    med_h = int(np.median(hues))
                    med_s = int(np.median(S[:, seg_start:i][seg_col]))
                    rank  = _hue_to_base_rank(med_h, med_s)
                elif seg_sil.sum() > 10:
                    rank = "Silver"
                else:
                    rank = None

                if rank is None:
                    continue

                cx = bx1 + seg_start + width // 2
                segments.append({
                    "rank_colour": rank,
                    "rank":        rank,
                    "x1":          bx1 + seg_start,
                    "x2":          bx1 + i,
                    "cx":          cx,
                    "width":       width,
                })
        return segments

    def _match_dist_icon(
        self, full_bgr: np.ndarray, seg_cx: int, icon_y1: int, icon_y2: int
    ) -> str | None:
        if not self._rank_icons:
            return None
        region_h = icon_y2 - icon_y1
        if region_h < 30:
            return None
        half_w    = min(region_h, 100)
        x1        = max(0, seg_cx - half_w)
        x2        = min(full_bgr.shape[1], seg_cx + half_w)
        crop_gray = cv2.cvtColor(full_bgr[icon_y1:icon_y2, x1:x2], cv2.COLOR_BGR2GRAY)

        best_rank: str | None = None
        best_score            = RANKED_DIST_ICON_THRESH

        # Try several target heights covering the observed on-screen icon range
        for target_h in (70, 90, 110, 130, 150):
            if target_h >= region_h:
                continue
            for rank, tmpl in self._rank_icons.items():
                scale     = target_h / tmpl.shape[0]
                resized   = cv2.resize(tmpl, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
                th, tw    = resized.shape[:2]
                ch, cw    = crop_gray.shape[:2]
                if th >= ch or tw >= cw:
                    continue
                result    = cv2.matchTemplate(crop_gray, resized, cv2.TM_CCOEFF_NORMED)
                score     = float(result.max())
                if score > best_score:
                    best_score = score
                    best_rank  = rank

        return best_rank

    def _ocr_dist_count(
        self, full_bgr: np.ndarray, x1: int, x2: int, by1: int, by2: int
    ) -> int | None:
        crop    = full_bgr[by1:by2, x1:x2]
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, bw   = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        big     = cv2.resize(bw, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        best: tuple[int, float] | None = None
        for _, text, conf in self._reader.readtext(big, allowlist="0123456789"):
            digits = text.strip()
            if digits.isdigit() and conf > 0.30:
                if best is None or conf > best[1]:
                    best = (int(digits), conf)
        return best[0] if best else None

    @staticmethod
    def _reconcile_dist_counts(segments: list[dict]) -> dict[str, int]:
        total_w    = sum(s["width"] for s in segments)
        raw        = [RANKED_SQUAD_TOTAL * s["width"] / total_w for s in segments]
        rounded    = [round(r) for r in raw]

        # Largest-remainder fix for rounding drift
        diff = RANKED_SQUAD_TOTAL - sum(rounded)
        if diff != 0:
            errors = [r - rnd for r, rnd in zip(raw, rounded)]
            for _ in range(abs(diff)):
                if diff > 0:
                    idx = max(range(len(errors)), key=lambda i: errors[i])
                    rounded[idx] += 1
                    errors[idx]  -= 1
                else:
                    idx = min(range(len(errors)), key=lambda i: errors[i])
                    rounded[idx] -= 1
                    errors[idx]  += 1

        width_counts = rounded

        # Prefer OCR; fall back to width proportion for missing reads
        ocrs  = [s.get("count_ocr") for s in segments]
        final = [ocr if ocr is not None else wc
                 for ocr, wc in zip(ocrs, width_counts)]

        # Force total to RANKED_SQUAD_TOTAL if still off
        diff = RANKED_SQUAD_TOTAL - sum(final)
        if diff != 0:
            target = next((i for i, o in enumerate(ocrs) if o is None), None)
            if target is None:
                target = min(range(len(final)), key=lambda i: final[i])
            final[target] += diff

        return {s["rank"]: c for s, c in zip(segments, final) if c > 0}


class ScreenClassifier:
    """Classifies a full-screen frame as GAME, EXPERIENCE, or RANKED_LOADING.

    EXPERIENCE: template-matches the purple SUMMARY tab (existing logic).
    RANKED_LOADING: template-matches the Apex logo at the top-left of the
      ranked lobby / dropzone banner.
    GAME: default when neither template matches.

    Templates and thresholds are calibrated against reference screenshots:
      experience: score 1.00 on exp screens, 0.47 on game/loading screens
      ranked_loading: score 0.93–1.00 on ranked screens, 0.48–0.56 on game screens
    """

    def __init__(self, template_path: str = EXPERIENCE_TEMPLATE,
                 ranked_template_path: str = RANKED_LOADING_TEMPLATE):
        tmpl = cv2.imread(template_path)
        if tmpl is None:
            raise FileNotFoundError(f"Screen classifier template not found: {template_path}")
        self._template   = tmpl
        self._ref_width  = 2726   # pixel width of the experience template source

        rtmpl = cv2.imread(ranked_template_path)
        if rtmpl is None:
            raise FileNotFoundError(f"Ranked-loading template not found: {ranked_template_path}")
        self._ranked_template  = rtmpl
        self._ranked_ref_width = RANKED_LOADING_REF_WIDTH

    def classify(self, full_bgr: np.ndarray) -> tuple["ScreenType", int]:
        """Classify the frame and return (ScreenType, bar_bottom_y).

        bar_bottom_y is the y-coordinate of the bottom edge of the purple
        LIVE/SUMMARY bar when EXPERIENCE is detected, or 0 otherwise.
        All experience-screen crop ratios should be anchored to this value.
        """
        sh, sw = full_bgr.shape[:2]

        # ── 1. Check EXPERIENCE first (purple tab bar) ────────────────────────
        scale = sw / self._ref_width
        tmpl  = cv2.resize(self._template, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
        search_h = max(tmpl.shape[0] + 1, int(sh * SCREEN_CLASSIFY_SEARCH_FRAC))
        region   = full_bgr[:search_h, :]
        if tmpl.shape[0] < region.shape[0] and tmpl.shape[1] < region.shape[1]:
            _, score, _, loc = cv2.minMaxLoc(
                cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
            )
            if score >= SCREEN_CLASSIFY_THRESHOLD:
                bar_bottom_y = loc[1] + tmpl.shape[0]
                return ScreenType.EXPERIENCE, bar_bottom_y

        # ── 2. Check RANKED_LOADING (Apex logo banner) ────────────────────────
        rscale = sw / self._ranked_ref_width
        rtmpl  = cv2.resize(self._ranked_template, None, fx=rscale, fy=rscale,
                             interpolation=cv2.INTER_AREA)
        rsearch_h = max(rtmpl.shape[0] + 1, int(sh * RANKED_LOADING_SEARCH_FRAC))
        rregion   = full_bgr[:rsearch_h, :]
        if rtmpl.shape[0] < rregion.shape[0] and rtmpl.shape[1] < rregion.shape[1]:
            _, rscore, _, _ = cv2.minMaxLoc(
                cv2.matchTemplate(rregion, rtmpl, cv2.TM_CCOEFF_NORMED)
            )
            if rscore >= RANKED_LOADING_THRESHOLD:
                return ScreenType.RANKED_LOADING, 0

        return ScreenType.GAME, 0


# ---------------------------------------------------------------------------
# Experience screen detector (stub)
# ---------------------------------------------------------------------------

class ExperienceDetector:
    """Reads the post-match EXPERIENCE summary screen.

    Implemented:
      current_rank  — derived from the "NEXT RANK: ..." text (e.g. "Platinum IV")
      next_rank     — as shown on screen (e.g. "Platinum III")

    Stubbed (TODO):
      ranked_rp, combat_rp, kills, assists, participations,
      bonus_rp, placement_rp, placement, cost_of_entry
    """

    # Matches "NEXT RANK" optionally followed by any non-alpha separator, then the rank
    # name and optional Roman-numeral tier.  The separator class covers colon, space,
    # apostrophe, and quote — OCR sometimes misreads ":" as "'" on downscaled images.
    _NEXT_RANK_RE = re.compile(
        r"NEXT\s*RANK[:\s'\"]*([A-Z]+(?:\s+[A-Z]+)?)\s*(IV|III|II|I)?",
        re.IGNORECASE,
    )

    def __init__(self, reader: easyocr.Reader):
        self._reader = reader
        # Store as a screen-height fraction so the cache stays valid across
        # screens with different resolutions (e.g. 1526px vs 1558px).
        self._cached_combat_header_frac: float | None = None
        # Experience screens are static — cache the full result after the first
        # detect() call so subsequent frames are free.  Call reset() when the
        # screen transitions away from EXPERIENCE.
        self._cached_result: dict | None = None

    # ------------------------------------------------------------------
    # Rank detection
    # ------------------------------------------------------------------

    @staticmethod
    def _bar_crop(
        full_bgr: np.ndarray,
        bar_bottom_y: int,
        top_rel: float, bot_rel: float,
        x1_abs: float, x2_abs: float,
        upscale: float = 2,
    ) -> np.ndarray:
        """Crop a region using bar-relative y fractions and absolute x fractions,
        then resize by upscale (default 2×; was 3× but game text is large enough)."""
        h, w      = full_bgr.shape[:2]
        content_h = h - bar_bottom_y
        y1 = bar_bottom_y + int(content_h * top_rel)
        y2 = bar_bottom_y + int(content_h * bot_rel)
        x1, x2    = int(w * x1_abs), int(w * x2_abs)
        crop      = full_bgr[y1:y2, x1:x2]
        return cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    def _detect_point_change(self, full_bgr: np.ndarray, bar_bottom_y: int) -> str | None:
        """Detect the RP change for this match (e.g. '+363' or '-20')."""
        big = self._bar_crop(
            full_bgr, bar_bottom_y,
            EXPERIENCE_POINT_CHANGE_TOP, EXPERIENCE_POINT_CHANGE_BOT,
            EXPERIENCE_POINT_CHANGE_X1,  EXPERIENCE_POINT_CHANGE_X2,
        )
        for _, text, conf in self._reader.readtext(big):
            if conf < 0.4:
                continue
            m = re.search(r'[+\-]?\s*\d+', text)
            if m:
                return m.group(0).replace(" ", "")
        return None

    def _detect_current_rp(self, full_bgr: np.ndarray, bar_bottom_y: int) -> str | None:
        """Detect the current RP shown on the yellow progress bar.

        Works in two stages:
        1. Finds where the yellow bar ends (the current-position marker) using HSV
           colour detection so the crop is anchored to the actual RP value regardless
           of how full the bar is.
        2. White-pixel thresholding + connected-component filtering removes the rank
           badge chevrons that flank the number, which were reading as spurious digits.
           Badges are eliminated by aspect ratio: they are wider than they are tall
           (w/h > 1.1), whereas digit strokes are portrait (w/h < 1.1).
        """
        h, w = full_bgr.shape[:2]
        content_h = h - bar_bottom_y
        y1 = bar_bottom_y + int(content_h * EXPERIENCE_CURRENT_RP_TOP)
        y2 = bar_bottom_y + int(content_h * EXPERIENCE_CURRENT_RP_BOT)
        x1, x2 = int(w * EXPERIENCE_CURRENT_RP_X1), int(w * EXPERIENCE_CURRENT_RP_X2)
        bar_strip = full_bgr[y1:y2, x1:x2]

        # Find rightmost yellow column (current-position marker on the bar).
        # The right end of the bar carries narrow chevron decorations (~25px wide) that
        # are always yellow regardless of RP; the real bar marker is ≥40px wide.
        # Split yellow columns into contiguous groups and discard narrow ones.
        hsv        = cv2.cvtColor(bar_strip, cv2.COLOR_BGR2HSV)
        yellow     = cv2.inRange(hsv, np.array([15, 80, 120]), np.array([40, 255, 255]))
        col_yellow = np.any(yellow > 0, axis=0)
        yellow_cols = np.where(col_yellow)[0]
        if not yellow_cols.size:
            return "0"   # bar is at the start of the tier (0 RP earned)
        gaps   = np.diff(yellow_cols)
        groups = np.split(yellow_cols, np.where(gaps > 5)[0] + 1)
        wide   = [g for g in groups if len(g) >= 40]
        if not wide:
            return "0"   # only decorative chevrons found — bar is at zero
        bar_end_x = int(wide[-1].max())

        # Crop a window left-heavy around the marker (text may sit on yellow or dark side)
        left_margin  = int(w * 0.15)
        right_margin = int(w * 0.04)
        wx1 = max(0, bar_end_x - left_margin)
        wx2 = min(bar_strip.shape[1], bar_end_x + right_margin)
        rp_crop = bar_strip[:, wx1:wx2]

        # White-pixel isolation then component filtering
        gray = cv2.cvtColor(rp_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        big    = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        big_h  = big.shape[0]
        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(big, connectivity=8)
        clean  = np.zeros_like(big)
        for i in range(1, n_comp):
            ch   = stats[i, cv2.CC_STAT_HEIGHT]
            cw   = stats[i, cv2.CC_STAT_WIDTH]
            area = stats[i, cv2.CC_STAT_AREA]
            if (40 <= ch <= big_h * 0.85
                    and area > 300
                    and (cw / ch) < 1.2):
                clean[labels == i] = 255

        hits = []
        for bbox, text, conf in self._reader.readtext(clean, allowlist="0123456789RP "):
            if conf < 0.25:
                continue
            cx     = int((bbox[0][0] + bbox[2][0]) / 2)
            digits = re.sub(r'\D', '', text)
            if digits:
                hits.append((cx, int(digits), conf))

        valid = [(cx, n, c) for cx, n, c in sorted(hits) if 1 <= n <= 749]
        return str(valid[0][1]) if valid else None

    def _detect_rank(self, full_bgr: np.ndarray, bar_bottom_y: int) -> tuple[str | None, str | None]:
        """Return (current_rank, next_rank) by finding "NEXT RANK ..." on screen.

        Crop coordinates are expressed as fractions of the content height that
        lies below bar_bottom_y (the bottom of the purple LIVE/SUMMARY bar),
        so they stay accurate whether or not the game is truly full-screen.

        next_rank    — e.g. "Platinum III"
        current_rank — the preceding step in RANK_PROGRESSION, e.g. "Platinum IV"
        """
        h = full_bgr.shape[0]
        content_h = h - bar_bottom_y
        y1 = bar_bottom_y + int(content_h * EXPERIENCE_NEXT_RANK_TOP)
        y2 = bar_bottom_y + int(content_h * EXPERIENCE_NEXT_RANK_BOT)
        crop = full_bgr[y1:y2, :]
        # Rank text is 40-60px tall on a 1500+px screen; 0.75x gives 30-45px, enough for accuracy
        crop = cv2.resize(crop, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)

        for _, text, conf in self._reader.readtext(crop):
            if conf < 0.35:
                continue
            m = self._NEXT_RANK_RE.search(text.upper())
            if not m:
                continue

            # Reconstruct the canonical rank string
            rank_word = m.group(1).strip().title()       # e.g. "Platinum" or "Apex Predator"
            tier_word = (m.group(2) or "").strip()       # e.g. "III" or ""
            next_str  = f"{rank_word} {tier_word}".strip()

            # Normalise via RANK_LOOKUP (handles "APEXPREDATOR" → "Apex Predator" etc.)
            next_canonical = RANK_LOOKUP.get(next_str.upper())
            if next_canonical is None:
                # Partial match fallback: find the first progression entry containing rank_word
                next_canonical = next(
                    (r for r in RANK_PROGRESSION if r.upper().startswith(rank_word.upper())),
                    None,
                )
            if next_canonical is None:
                continue

            idx = RANK_PROGRESSION.index(next_canonical)
            current_canonical = RANK_PROGRESSION[idx - 1] if idx > 0 else None
            return current_canonical, next_canonical

        return None, None

    # ------------------------------------------------------------------
    # Section helpers
    # ------------------------------------------------------------------

    def _find_combat_header_y(self, full_bgr: np.ndarray) -> int:
        """Locate the COMBAT section header by OCR; cached as a screen-height fraction."""
        sh, sw = full_bgr.shape[:2]
        if self._cached_combat_header_frac is not None:
            return int(sh * self._cached_combat_header_frac)
        y_off  = int(sh * EXPERIENCE_SECTIONS_SCAN_TOP)
        crop   = full_bgr[y_off : int(sh * EXPERIENCE_SECTIONS_SCAN_BOT), :]
        for bbox, text, conf in self._reader.readtext(crop):
            if conf > 0.50 and text.upper().strip() == "COMBAT":
                header_y = y_off + int(bbox[0][1])
                self._cached_combat_header_frac = header_y / sh
                return header_y
        self._cached_combat_header_frac = 0.540
        return int(sh * 0.540)

    def _sections_band_ocr(
        self, full_bgr: np.ndarray
    ) -> list[tuple[float, float, str, float]]:
        """Run ONE readtext call over the sections area (y 45–80% of screen).

        Replaces 19 individual _ocr_cell calls: 1 header scan + 18 value cells
        across COMBAT / BONUSES / PLACEMENT. Returns a list of
        (cx, cy, text, conf) where cx/cy are absolute full-screen pixel coords.
        """
        sh, sw = full_bgr.shape[:2]
        y0   = int(sh * 0.45)
        y1   = int(sh * 0.80)
        crop = full_bgr[y0:y1, :]
        # Native scale: the experience screen is static, so this 4-5s call is
        # paid only ONCE per match (result cache in detect() handles all later
        # frames).  Downscaling to save time here costs OCR accuracy on small
        # single-digit count fields (kills, challenger_count, etc.).
        hits = []
        for bbox, text, conf in self._reader.readtext(crop):
            tl, _, br, _ = bbox
            cx = (tl[0] + br[0]) / 2
            cy = (tl[1] + br[1]) / 2 + y0
            hits.append((cx, cy, text.strip(), conf))
        return hits

    def _find_combat_header_y_from_batch(
        self,
        batch: list[tuple[float, float, str, float]],
        sh: int,
        sw: int,
    ) -> int:
        """Derive combat header y from pre-computed batch results (no extra OCR call).

        Cache is stored as a screen-height fraction so it stays valid across
        screens at different resolutions.
        """
        if self._cached_combat_header_frac is not None:
            return int(sh * self._cached_combat_header_frac)
        x_cutoff = sw * 0.40
        for cx, cy, text, conf in batch:
            if conf > 0.50 and text.upper().strip() == "COMBAT" and cx < x_cutoff:
                self._cached_combat_header_frac = cy / sh
                return int(cy)
        self._cached_combat_header_frac = 0.540
        return int(sh * 0.540)

    @staticmethod
    def _lookup_batch(
        batch: list[tuple[float, float, str, float]],
        y_center: int,
        x1_frac: float,
        x2_frac: float,
        sw: int,
        y_tol: int,
        min_conf: float = 0.35,
    ) -> list[tuple[str, float]]:
        """Filter batch results to those in cell (y_center±y_tol, x1_frac..x2_frac).

        Drop-in replacement for _ocr_cell output format: list of (text, conf).
        """
        x1_abs = sw * x1_frac
        x2_abs = sw * x2_frac
        return [
            (text, conf)
            for cx, cy, text, conf in batch
            if x1_abs <= cx <= x2_abs and abs(cy - y_center) <= y_tol and conf >= min_conf
        ]

    def _ocr_cell(
        self,
        full_bgr: np.ndarray,
        y_center: int,
        x1_frac: float,
        x2_frac: float,
        upscale: int = 3,
        allowlist: str | None = None,
    ) -> list[tuple[str, float]]:
        """Crop one row-cell, upscale, and return (text, conf) pairs ≥ 0.35."""
        sh, sw    = full_bgr.shape[:2]
        half_h    = int(sh * EXPERIENCE_ROW_HALF_H)
        y1, y2    = max(0, y_center - half_h), min(sh, y_center + half_h)
        x1, x2    = int(sw * x1_frac), int(sw * x2_frac)
        crop      = full_bgr[y1:y2, x1:x2]
        big       = cv2.resize(crop, None, fx=upscale, fy=upscale,
                               interpolation=cv2.INTER_CUBIC)
        kwargs    = {"allowlist": allowlist} if allowlist else {}
        return [(t, c) for _, t, c in self._reader.readtext(big, **kwargs) if c > 0.35]

    @staticmethod
    def _parse_rp(texts: list[tuple[str, float]], negate: bool = False) -> str | None:
        """Extract an RP number from OCR hits.

        Prefers explicit 'NNN RP' patterns (sorted by confidence) over bare digit
        strings, which may be OCR noise from neighbouring cells.  Strips leading
        noise digits for values like '7189 RP' → 188.
        """
        def _to_int(text: str) -> int | None:
            digits = re.sub(r"\D", "", text)
            if not digits:
                return None
            n = int(digits)
            if n > 9999:
                for i in range(1, len(digits)):
                    n2 = int(digits[i:])
                    if n2 <= 9999:
                        return n2
                return None
            return n

        # First pass: look for explicit "NNN RP" pattern (most reliable)
        for text, _ in sorted(texts, key=lambda x: -x[1]):
            if re.search(r"\d+\s*RP", text, re.IGNORECASE):
                n = _to_int(text)
                if n is not None and 0 <= n <= 9999:
                    return str(-n) if negate else str(n)
        # Fallback: any digit string in range
        for text, _ in sorted(texts, key=lambda x: -x[1]):
            n = _to_int(text)
            if n is not None and 0 <= n <= 9999:
                return str(-n) if negate else str(n)
        return None

    @staticmethod
    def _parse_count(texts: list[tuple[str, float]]) -> str | None:
        """Extract a small integer count (0–999) from OCR hits."""
        for text, _ in texts:
            digits = re.sub(r"\D", "", text)
            if digits and int(digits) <= 999:
                return digits
        return None

    @staticmethod
    def _parse_placement(texts: list[tuple[str, float]]) -> str | None:
        """Extract '#N' placement text."""
        for text, _ in texts:
            m = re.search(r"#\s*(\d+)", text)
            if m:
                return f"#{m.group(1)}"
        return None

    @staticmethod
    def _parse_streak(texts: list[tuple[str, float]]) -> str | None:
        """Extract 'N/5' top-5 streak fraction."""
        for text, _ in texts:
            m = re.search(r"(\d)\s*/\s*5", text)
            if m:
                return f"{m.group(1)}/5"
        return None

    @staticmethod
    def _parse_tier_name(texts: list[tuple[str, float]]) -> str | None:
        """Extract a rank tier name (e.g. 'Platinum IV') from OCR hits."""
        for text, conf in texts:
            candidate = RANK_LOOKUP.get(text.upper().strip())
            if candidate:
                return candidate
            # Partial match: two-word rank names
            for rank in RANK_PROGRESSION:
                if rank.upper() in text.upper():
                    return rank
        return None

    @staticmethod
    def _parse_participations(texts: list[tuple[str, float]]) -> str | None:
        """Extract participations formula like '3×50%−15'."""
        for text, _ in texts:
            # Normalise OCR variants: 'I'→'1', 'x'→'×', '-'→'−'
            t = text.replace("I", "1").replace("l", "1").replace("x", "×").replace("X", "×")
            m = re.search(r"(\d+)[×*](\d+)%[-−](\d+\.?\d*)", t)
            if m:
                return f"{m.group(1)}×{m.group(2)}%−{m.group(3)}"
        return None

    # ------------------------------------------------------------------
    # Section detection
    # ------------------------------------------------------------------

    _BASE_COMBAT_VALUE_RE = re.compile(
        r"(?:BASE\s+)?COMBAT\s+VALUE.*?(\d+)", re.IGNORECASE
    )

    def _detect_combat(
        self,
        full_bgr: np.ndarray,
        header_y: int,
        batch: list[tuple[float, float, str, float]] | None = None,
    ) -> dict:
        sh, sw = full_bgr.shape[:2]
        big_gap   = int(sh * EXPERIENCE_BIG_ROW_GAP)
        small_gap = int(sh * EXPERIENCE_SMALL_ROW_GAP)
        y_tol     = int(sh * EXPERIENCE_ROW_HALF_H)
        totals_y  = header_y + big_gap
        row_a_y   = totals_y  + big_gap
        row_b_y   = row_a_y   + small_gap
        row_c_y   = row_b_y   + small_gap
        row_d_y   = row_c_y   + small_gap

        x1r, x2r = EXPERIENCE_COMBAT_RP_X
        x1c, x2c = EXPERIENCE_COMBAT_COUNT_X

        def _cell(y, x1, x2, fallback_upscale=None, **kwargs):
            if batch is not None:
                hits = self._lookup_batch(batch, y, x1, x2, sw, y_tol)
                if not hits and fallback_upscale is not None:
                    return self._ocr_cell(full_bgr, y, x1, x2,
                                          upscale=fallback_upscale, **kwargs)
                return hits
            return self._ocr_cell(full_bgr, y, x1, x2, **kwargs)

        # BASE COMBAT VALUE label sits just below the combat section in the batch.
        base_combat_value = None
        if batch is not None:
            for _cx, _cy, text, conf in batch:
                m = self._BASE_COMBAT_VALUE_RE.search(text)
                if m and conf > 0.5:
                    base_combat_value = str(int(m.group(1)))
                    break

        result = {
            "combat_rp_total":        self._parse_rp(_cell(totals_y, x1r, x2r,
                                                           allowlist="0123456789RP ")),
            "kills":                  self._parse_count(_cell(row_a_y, x1c, x2c,
                                                              fallback_upscale=4,
                                                              allowlist="0123456789")),
            "kills_rp":               self._parse_rp(_cell(row_a_y, x1r, x2r,
                                                           allowlist="0123456789RP ")),
            "assists":                self._parse_count(_cell(row_b_y, x1c, x2c,
                                                              fallback_upscale=4,
                                                              allowlist="0123456789")),
            "assists_rp":             self._parse_rp(_cell(row_b_y, x1r, x2r,
                                                           allowlist="0123456789RP ")),
            "participations_formula": self._parse_participations(_cell(row_c_y, x1c, x2c)),
            "participations_rp":      self._parse_rp(_cell(row_c_y, x1r, x2r,
                                                           allowlist="0123456789RP ")),
            "kill_cap_adjustment_rp": self._parse_rp(_cell(row_d_y, x1r, x2r,
                                                           allowlist="0123456789RP "),
                                                     negate=True),
        }
        if base_combat_value is not None:
            result["base_combat_value"] = base_combat_value
        return result

    def _detect_bonuses(
        self,
        full_bgr: np.ndarray,
        header_y: int,
        batch: list[tuple[float, float, str, float]] | None = None,
    ) -> dict:
        sh, sw = full_bgr.shape[:2]
        big_gap   = int(sh * EXPERIENCE_BIG_ROW_GAP)
        small_gap = int(sh * EXPERIENCE_SMALL_ROW_GAP)
        y_tol     = int(sh * EXPERIENCE_ROW_HALF_H)
        totals_y  = header_y + big_gap
        row_a_y   = totals_y  + big_gap
        row_b_y   = row_a_y   + small_gap
        row_c_y   = row_b_y   + small_gap

        x1r, x2r = EXPERIENCE_BONUSES_RP_X
        x1c, x2c = EXPERIENCE_BONUSES_COUNT_X

        def _cell(y, x1, x2, **kwargs):
            if batch is not None:
                return self._lookup_batch(batch, y, x1, x2, sw, y_tol)
            return self._ocr_cell(full_bgr, y, x1, x2, **kwargs)

        return {
            "bonus_rp_total":   self._parse_rp(_cell(totals_y, x1r, x2r,
                                                     allowlist="0123456789RP ")),
            "challenger_count": self._parse_count(_cell(row_a_y, x1c, x2c, upscale=4)),
            "challenger_rp":    self._parse_rp(_cell(row_a_y, x1r, x2r,
                                                     allowlist="0123456789RP ")),
            "top5_streak":      self._parse_streak(_cell(row_b_y, x1c, x2c, upscale=4)),
            "top5_streak_rp":   self._parse_rp(_cell(row_b_y, x1r, x2r,
                                                     allowlist="0123456789RP ")),
            "promotion_rp":     self._parse_rp(_cell(row_c_y, x1r, x2r,
                                                     allowlist="0123456789RP ")),
        }

    def _detect_placement(
        self,
        full_bgr: np.ndarray,
        header_y: int,
        batch: list[tuple[float, float, str, float]] | None = None,
    ) -> dict:
        sh, sw = full_bgr.shape[:2]
        big_gap   = int(sh * EXPERIENCE_BIG_ROW_GAP)
        small_gap = int(sh * EXPERIENCE_SMALL_ROW_GAP)
        y_tol     = int(sh * EXPERIENCE_ROW_HALF_H)
        totals_y  = header_y + big_gap
        row_a_y   = totals_y  + big_gap
        row_b_y   = row_a_y   + small_gap

        x1t, x2t = EXPERIENCE_PLACEMENT_TEXT_X
        x1r, x2r = EXPERIENCE_PLACEMENT_RP_X

        def _cell(y, x1, x2, **kwargs):
            if batch is not None:
                return self._lookup_batch(batch, y, x1, x2, sw, y_tol)
            return self._ocr_cell(full_bgr, y, x1, x2, **kwargs)

        # Placement row: scan the full section width sorted left-to-right.
        # placement     = first token whose text contains '#' immediately left of a number.
        # placement_rp  = first token whose text has a number immediately left of 'RP'.
        if batch is not None:
            row_a_hits = sorted(
                [(cx, text, conf)
                 for cx, cy, text, conf in batch
                 if sw * 0.80 <= cx <= sw and abs(cy - row_a_y) <= y_tol and conf >= 0.35],
                key=lambda h: h[0],
            )
            placement = placement_rp = None
            for _, text, conf in row_a_hits:
                if placement is None and re.search(r"#\s*\d+", text):
                    placement = self._parse_placement([(text, conf)])
                if placement_rp is None and re.search(r"\d+\s*RP", text, re.IGNORECASE):
                    placement_rp = self._parse_rp([(text, conf)])
        else:
            placement    = self._parse_placement(_cell(row_a_y, x1t, x2t))
            placement_rp = self._parse_rp(_cell(row_a_y, x1r, x2r,
                                                allowlist="0123456789RP "))

        return {
            "placement_rp_total": self._parse_rp(_cell(totals_y, x1r, x2r,
                                                        allowlist="0123456789RP ")),
            "placement":          placement,
            "placement_rp":       placement_rp,
            "cost_of_entry_tier": self._parse_tier_name(_cell(row_b_y, x1t, x2t)),
            "cost_of_entry_rp":   self._parse_rp(_cell(row_b_y, x1r, x2r,
                                                        allowlist="0123456789RP "),
                                                  negate=True),
        }

    # ------------------------------------------------------------------
    # Sanity checks and derived values
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_sanity_checks(result: dict) -> dict:
        """Validate and fill in missing RP values using the four known constraints.

        Constraints (each applied as a 3-or-more-variable system):
          placement_rp_total = placement_rp + cost_of_entry_rp
          combat_rp_total    = kills_rp + assists_rp + participations_rp
                               + (kill_cap_adjustment_rp or 0)    [kill_cap optional]
          bonus_rp_total     = challenger_rp + top5_streak_rp
                               + (promotion_rp or 0)              [promotion optional]
          point_change       = combat_rp_total + bonus_rp_total + placement_rp_total

        For each constraint, if exactly one value is absent, it is derived.
        If all are present and disagree, the conflict is noted in "sanity_issues".
        The one exception: when combat_rp_total conflicts with the component sum
        (e.g. rank-badge digit merge producing "7189" vs "189"), the total is
        overridden by the component sum because individual cells are more reliable.
        """
        r = dict(result)
        issues: list[str] = []
        TOLERANCE = 2  # allow ±2 RP rounding before flagging a conflict

        def get(key: str) -> int | None:
            val = r.get(key)
            if val is None:
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        def put(key: str, value: int) -> None:
            """Set a field only when it is currently None (never overwrites)."""
            if r.get(key) is None:
                r[key] = (f"+{value}" if key == "point_change" and value >= 0
                           else str(value))

        def constrain(
            total_key: str,
            part_keys: list[str],
            optional: frozenset[str] = frozenset(),
        ) -> bool:
            """Apply total = sum(parts), treating optional parts as 0 when absent.

            Returns True if a conflict is detected (all values known but total ≠ sum).
            """
            total = get(total_key)
            parts = {k: get(k) for k in part_keys}
            miss_req = [k for k in part_keys if k not in optional and parts[k] is None]
            miss_opt = [k for k in part_keys if k in optional and parts[k] is None]
            known_sum = sum(v for v in parts.values() if v is not None)

            if total is None and not miss_req:
                # Derive total from all known parts (missing optional = 0)
                put(total_key, known_sum)
            elif total is not None and len(miss_req) == 1 and not miss_opt:
                # Derive the single missing required part
                put(miss_req[0], total - known_sum)
            elif total is not None and not miss_req and len(miss_opt) == 1:
                # Derive the single missing optional part
                put(miss_opt[0], total - known_sum)
            elif total is not None and not miss_req and not miss_opt:
                if abs(total - known_sum) > TOLERANCE:
                    return True  # conflict
            return False

        # ── 1. Placement ─────────────────────────────────────────────────────
        # placement_rp_total = placement_rp + cost_of_entry_rp
        # (cost_of_entry_rp is already stored as a negative value)
        constrain("placement_rp_total", ["placement_rp", "cost_of_entry_rp"])

        # ── 2. Combat ────────────────────────────────────────────────────────
        # kill_cap_adjustment_rp is absent when the kill cap never triggered.
        combat_conflict = constrain(
            "combat_rp_total",
            ["kills_rp", "assists_rp", "participations_rp", "kill_cap_adjustment_rp"],
            optional=frozenset({"kill_cap_adjustment_rp"}),
        )
        if combat_conflict:
            # The total is almost certainly a badge-digit merge (e.g. "7189" → 189).
            # Override it with the component sum, which is computed from individual cells.
            kr = get("kills_rp")
            ar = get("assists_rp")
            pr = get("participations_rp")
            kc = get("kill_cap_adjustment_rp") or 0
            if all(v is not None for v in [kr, ar, pr]):
                corrected = kr + ar + pr + kc
                old = r.get("combat_rp_total")
                r["combat_rp_total"] = str(corrected)
                issues.append(f"combat_rp_total corrected {old!r} → {corrected!r} (component sum)")

        # ── 2b. Derive base_combat_value from kills/assists if OCR text missed it ──
        # Try kills then assists; prefer whichever gives a clean integer quotient.
        # Both are tried and if they agree the value is confirmed; if they disagree
        # kills takes precedence (larger counts tend to be read with higher confidence).
        if r.get("base_combat_value") is None:
            candidates = []
            for count_key, rp_key in [("kills", "kills_rp"), ("assists", "assists_rp")]:
                n  = get(count_key)
                rp = get(rp_key)
                if n and rp and rp % n == 0:
                    candidates.append(rp // n)
            if candidates:
                put("base_combat_value", candidates[0])

        # ── 2c. Count derivation via base combat value ───────────────────────
        # kills = kills_rp / base_combat_value (each kill/assist is worth exactly bcv RP)
        # Only applied when exact integer division holds; guards against RP formula variants.
        bcv = get("base_combat_value")
        if bcv and bcv > 0:
            for count_key, rp_key in [("kills", "kills_rp"), ("assists", "assists_rp")]:
                if r.get(count_key) is None:
                    rp = get(rp_key)
                    if rp is not None and rp % bcv == 0:
                        put(count_key, rp // bcv)

        # ── 3. Bonuses ───────────────────────────────────────────────────────
        # promotion_rp is absent in matches where no rank promotion occurred.
        bonus_conflict = constrain(
            "bonus_rp_total",
            ["challenger_rp", "top5_streak_rp", "promotion_rp"],
            optional=frozenset({"promotion_rp"}),
        )
        if bonus_conflict:
            issues.append(
                f"bonus_rp_total {r.get('bonus_rp_total')!r} ≠ "
                f"challenger {r.get('challenger_rp')} + "
                f"streak {r.get('top5_streak_rp')} + "
                f"promotion {r.get('promotion_rp')}"
            )

        # ── 4. Cross-section ─────────────────────────────────────────────────
        # point_change = combat_rp_total + bonus_rp_total + placement_rp_total
        cross_conflict = constrain(
            "point_change",
            ["combat_rp_total", "bonus_rp_total", "placement_rp_total"],
        )
        if cross_conflict:
            pc   = get("point_change")
            calc = (get("combat_rp_total") or 0) + (get("bonus_rp_total") or 0) + (get("placement_rp_total") or 0)
            issues.append(f"point_change {pc!r} ≠ section sum {calc!r}")

        if issues:
            r["sanity_issues"] = issues
        return r

    # ------------------------------------------------------------------
    # Public detection
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Invalidate all per-screen caches.

        Call this when the screen transitions away from EXPERIENCE (i.e. when
        ScreenClassifier.classify() first returns GAME or UNKNOWN after a run
        of EXPERIENCE frames).  Without this, the cached result from the last
        match will be returned for the new match's experience screen.
        """
        self._cached_result = None
        self._cached_combat_header_frac = None

    def detect(self, full_bgr: np.ndarray, bar_bottom_y: int = 0) -> dict:
        """Return a dict of all experience-screen values.
        bar_bottom_y should be the y-coordinate returned by ScreenClassifier.classify().

        The experience screen is static: the first call does full OCR (4 calls,
        down from the previous 22), then the result is cached so all subsequent
        frames of the same experience screen are free.  Call reset() when the
        screen transitions away from EXPERIENCE.

        OCR calls on first frame: 4 total (down from 22).
          1. _sections_band_ocr — one broad pass replacing 19 individual cell calls
          2. _detect_rank
          3. _detect_point_change
          4. _detect_current_rp  (needs special connected-component preprocessing)
        """
        if self._cached_result is not None:
            return self._cached_result

        sh, sw = full_bgr.shape[:2]

        # ONE OCR pass over the sections area replaces 19 individual readtext calls
        # (1 header scan + 18 value-cell calls across combat/bonuses/placement).
        batch = self._sections_band_ocr(full_bgr)
        combat_header_y = self._find_combat_header_y_from_batch(batch, sh, sw)

        current_rank, next_rank = self._detect_rank(full_bgr, bar_bottom_y)

        result = {
            "current_rank": current_rank,
            "next_rank":    next_rank,
            "point_change": self._detect_point_change(full_bgr, bar_bottom_y),
            "current_rp":   self._detect_current_rp(full_bgr, bar_bottom_y),
        }
        result.update(self._detect_combat(full_bgr, combat_header_y, batch=batch))
        result.update(self._detect_bonuses(full_bgr, combat_header_y, batch=batch))
        result.update(self._detect_placement(full_bgr, combat_header_y, batch=batch))
        result = self._apply_sanity_checks(result)
        self._cached_result = result
        return result

"""
All configuration constants for the Apex Legends HUD detector.

Fraction values were derived by template-matching each UI panel against a
2696×1520 full-screen reference image.  Nothing in this file should compute
anything more than a derived constant from other constants defined here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
ROI_FILE                  = os.path.join(_HERE, "rois.json")
ICON_DIR                  = os.path.join(_HERE, "images", "stat-icons")
MAPS_DIR                  = os.path.join(_HERE, "images", "maps")        # base maps + <slug>_pois.json
EXPERIENCE_TEMPLATE       = os.path.join(_HERE, "images", "ui-images", "experience_tab_template.png")
RANKED_LOADING_TEMPLATE   = os.path.join(_HERE, "images", "ui-images", "ranked_loading_template.png")
MATCH_HISTORY_CSV         = os.path.join(_HERE, "match_history.csv")   # one row per game (experience)
MATCH_REPLAY_CSV          = os.path.join(_HERE, "match_replay.csv")    # one row per X s during GAME
OUTPUT_DIR                = os.path.join(_HERE, "output")               # per-game subdirs: output/<game_id>/

# ---------------------------------------------------------------------------
# Ranked-loading screen classifier
# ---------------------------------------------------------------------------

# Template is a 123×118 crop of the Apex-logo area from a 2808-px-wide reference.
# Match scores: ranked-loading=0.925–1.00, game=0.48–0.56, experience=0.53–0.56.
RANKED_LOADING_THRESHOLD   = 0.65   # minimum TM_CCOEFF_NORMED score
RANKED_LOADING_SEARCH_FRAC = 0.25   # search top 25% of screen
RANKED_LOADING_REF_WIDTH   = 2808   # pixel width of the template source image

# Map-name sub-banner ROI (fractions of full-screen height/width)
RANKED_LOADING_MAP_Y1 = 0.150
RANKED_LOADING_MAP_Y2 = 0.220
RANKED_LOADING_MAP_X1 = 0.005
RANKED_LOADING_MAP_X2 = 0.190   # stops before the game-scene edge that creates noise

# Canonical Apex BR map names as displayed on the loading screen.
# Squad rank distribution bar (ranked loading screen)
RANKED_DIST_ICON_DIR        = os.path.join(_HERE, "images", "rank-icons")
RANKED_DIST_SEARCH_Y1       = 0.87    # top of bar search band (screen-height fraction)
RANKED_DIST_SEARCH_Y2       = 0.97    # bottom of bar search band
RANKED_DIST_SEARCH_X1       = 0.20    # left of bar search band (screen-width fraction)
RANKED_DIST_SEARCH_X2       = 0.80    # right of bar search band
RANKED_DIST_FILL_V_MIN      = 140     # min V for a coloured fill pixel
RANKED_DIST_FILL_S_MIN      = 60      # min S for a coloured fill pixel
RANKED_DIST_COL_THRESH      = 0.10    # column fill-fraction threshold (below = divider/border)
RANKED_DIST_SEG_MIN_PX      = 20      # minimum segment width in pixels
RANKED_DIST_ICON_SEARCH_PX  = 200     # pixels above bar_top to search for rank icons
RANKED_DIST_ICON_THRESH     = 0.55    # template-match score threshold for icon identification
RANKED_SQUAD_TOTAL          = 20      # expected total squads per ranked lobby

APEX_MAP_NAMES: list[str] = [
    "Kings Canyon",
    "World's Edge",
    "Olympus",
    "Storm Point",
    "Broken Moon",
    "E-District",
]

# ---------------------------------------------------------------------------
# Session / output cadence
# ---------------------------------------------------------------------------

REPLAY_INTERVAL_SEC        = 5     # seconds between match_replay rows (CLI-overridable)
EXPERIENCE_CAPTURE_DELAY_SEC = 5.0  # wait this long after EXPERIENCE first seen before OCR
PHASE_DEBOUNCE_FRAMES      = 4     # consecutive identical classifications to commit a phase
NEW_GAME_GAP_SEC           = 20.0  # non-GAME gap this long forces a new game_id on next GAME

# ---------------------------------------------------------------------------
# Screen classification
# ---------------------------------------------------------------------------

# Template-match score above which the screen is classified as EXPERIENCE.
# Measured: experience=1.00, game=0.47 — 0.75 gives clean separation.
SCREEN_CLASSIFY_THRESHOLD    = 0.60   # full 0-70% template: s1=1.00 s2=0.77 s3=0.62 game=0.26
SCREEN_CLASSIFY_INTERVAL     = 5    # frames between classification checks
# Search the top 20% of the screen for the purple bar (handles minor vertical offsets)
SCREEN_CLASSIFY_SEARCH_FRAC  = 0.20

# ---------------------------------------------------------------------------
# Ranked tier progression  (lowest → highest)
# Rookie has no sub-tiers; Bronze–Diamond have IV→I; Master and Apex Predator
# have no sub-tiers.  Roman numerals match the in-game display exactly.
# ---------------------------------------------------------------------------

_TIERED_RANKS = ("Bronze", "Silver", "Gold", "Platinum", "Diamond")
_TIERS        = ("IV", "III", "II", "I")

RANK_PROGRESSION: list[str] = (
    ["Rookie"]
    + [f"{rank} {tier}" for rank in _TIERED_RANKS for tier in _TIERS]
    + ["Master", "Apex Predator"]
)

# Map normalised uppercase → canonical title-case name (e.g. "PLATINUM III" → "Platinum III")
RANK_LOOKUP: dict[str, str] = {r.upper(): r for r in RANK_PROGRESSION}

# point_change (+363) — the RP earned this match, above "Ranked Points Earned"
# y fractions are relative to bar_bottom; x fractions are absolute screen width.
# All y-fractions below are relative to bar_bottom (bottom of LIVE/SUMMARY bar).
# Reference: 2726x1526 screen, bar_bottom=68, content_height=1458.

# point_change (+363) — above "Ranked Points Earned"
EXPERIENCE_POINT_CHANGE_TOP = 0.131  # (259-68)/1458
EXPERIENCE_POINT_CHANGE_BOT = 0.207  # (370-68)/1458
EXPERIENCE_POINT_CHANGE_X1  = 0.350
EXPERIENCE_POINT_CHANGE_X2  = 0.620

# current_rp — yellow progress bar (actual marker found by HSV)
# x range is between the left rank badge and the right next-rank chevron icon
EXPERIENCE_CURRENT_RP_TOP   = 0.215  # (381-68)/1458
EXPERIENCE_CURRENT_RP_BOT   = 0.236  # (412-68)/1458
EXPERIENCE_CURRENT_RP_X1    = 0.410  # just right of the left rank badge
EXPERIENCE_CURRENT_RP_X2    = 0.570  # just left of the right next-rank chevron

# NEXT RANK text
EXPERIENCE_NEXT_RANK_TOP = 0.246  # (427-68)/1458
EXPERIENCE_NEXT_RANK_BOT = 0.372  # (610-68)/1458

# ---------------------------------------------------------------------------
# Section layout (COMBAT / BONUSES / MATCH PLACEMENT)
# Measured from all three reference screens (1526–1558px tall, bar_bottom≈68).
# Row positions are anchored to the COMBAT header detected by OCR.
# ---------------------------------------------------------------------------

EXPERIENCE_SECTIONS_SCAN_TOP  = 0.48    # y-frac to start scanning for headers
EXPERIENCE_SECTIONS_SCAN_BOT  = 0.60    # y-frac where headers are expected
EXPERIENCE_BIG_ROW_GAP        = 0.047   # header→totals and totals→row_a gap
EXPERIENCE_SMALL_ROW_GAP      = 0.033   # row_a→b, b→c, c→d gap
EXPERIENCE_ROW_HALF_H         = 0.015   # half-height of a row for OCR crop (38px overshot; 23px safe)

# Column x-fractions (fraction of screen width, left-edge of each value cell)
EXPERIENCE_COMBAT_COUNT_X     = (0.20, 0.27)   # kills/assists count; participations formula
EXPERIENCE_COMBAT_RP_X        = (0.26, 0.33)   # all COMBAT RP values
EXPERIENCE_BONUSES_COUNT_X    = (0.52, 0.61)   # challenger count; top5 fraction
EXPERIENCE_BONUSES_RP_X       = (0.58, 0.67)   # all BONUSES RP values
EXPERIENCE_PLACEMENT_TEXT_X   = (0.83, 0.94)   # placement "#N"; cost-of-entry tier name
EXPERIENCE_PLACEMENT_RP_X     = (0.93, 1.00)   # PLACEMENT RP values (0.93 clears the #N numbers)

# ---------------------------------------------------------------------------
# Shield detector  (bottom-left player card)
# ---------------------------------------------------------------------------

SHIELD_BAR_X1      = 0.091   # left edge of bar area (past portrait)
SHIELD_BAR_X2      = 0.202   # right edge of bar area
SHIELD_STRIP_Y1    = 0.912   # top of combined shield+health search band
SHIELD_STRIP_Y2    = 0.960   # bottom of combined shield+health search band
SHIELD_BAR_MID_Y   = 0.940   # divides shield bar (above) from health bar (below)

# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

CAPTURE_PADDING    = 20    # extra pixels added to every ROI edge when capturing
VERTICAL_BUFFER_PX = 0     # pixels to strip from top+bottom for non-fullscreen game windows

# ---------------------------------------------------------------------------
# Minimap localisation  (top-left HUD minimap → position on the full map)
# Fractions of full-screen width/height; calibrated on a 2696×1520 capture.
# ---------------------------------------------------------------------------

MINIMAP_LEFT_FRAC        = 0.012
MINIMAP_TOP_FRAC         = 0.035
MINIMAP_RIGHT_FRAC       = 0.145
MINIMAP_BOT_FRAC         = 0.250
MINIMAP_CENTER_MASK_FRAC = 0.12   # radius (of minimap min-dim) to mask the player chevron
MAP_MIN_INLIERS          = 12     # min RANSAC inliers to trust a localisation

# ---------------------------------------------------------------------------
# Weapon bar  (bottom-right)
# ---------------------------------------------------------------------------

WEAPON_LEFT_FRAC   = 0.768
WEAPON_TOP_FRAC    = 0.841
WEAPON_WIDTH_FRAC  = 0.230
WEAPON_HEIGHT_FRAC = 0.159
WEAPON_UPSCALE     = 2     # upscale factor applied before OCR
MIN_CONFIDENCE     = 0.5

WEAPON_NAMES: list[str] = [
    # Assault Rifles
    "R-301 Carbine", "Hemlok Burst AR", "HAVOC Rifle",
    "VK-47 Flatline", "Nemesis Burst AR",
    # SMGs
    "R-99 SMG", "Alternator SMG", "Volt SMG", "CAR SMG",
    # LMGs
    "Devotion LMG", "Spitfire", "L-STAR EMG", "Rampage LMG",
    # Sniper Rifles
    "Longbow DMR", "Triple Take", "Sentinel", "Charge Rifle",
    "Kraber 50-Cal Sniper",
    # Shotguns
    "Peacekeeper", "Mastiff Shotgun", "Mozambique Shotgun", "EVA-8 Auto",
    # Pistols
    "Wingman", "RE-45 Auto", "P2020",
    # Marksman
    "G7 Scout", "30-30 Repeater", "Bocek Compound Bow",
]

# Flat set of every meaningful token (≥3 chars) across all weapon names.
WEAPON_TOKENS: set[str] = {
    token.lower()
    for name in WEAPON_NAMES
    for token in name.replace("-", " ").replace(".", "").split()
    if len(token) >= 3
}

# ---------------------------------------------------------------------------
# Armor-level triangle  (bottom-left)
# ---------------------------------------------------------------------------

ARMOR_LEFT_FRAC    = 0.000
ARMOR_TOP_FRAC     = 0.834
ARMOR_WIDTH_FRAC   = 0.026
ARMOR_HEIGHT_FRAC  = 0.161
ARMOR_UPSCALE      = 3
ARMOR_MIN_CONF     = 0.7

# ---------------------------------------------------------------------------
# Top-right UI panel
# ---------------------------------------------------------------------------

TR_LEFT_FRAC         = 0.7285
TR_TOP_FRAC          = 0.0053
TR_WIDTH_FRAC        = 0.2685
TR_HEIGHT_FRAC       = 0.1592
TR_BADGE_CUTOFF_FRAC = 0.801   # x-fraction past which the ranked badge sits
TR_UPSCALE           = 3
TR_MIN_CONF          = 0.4     # lower than MIN_CONFIDENCE; icon-adjacent digits score ~0.6

# Sub-row positions as fractions of the panel height (reference panel = 242 px)
TR_SQUADS_ROW_TOP = 0.248      # y ≈ 60
TR_SQUADS_ROW_BOT = 0.475      # y ≈ 115
TR_STATS_ROW_TOP  = 0.496      # y ≈ 120
TR_STATS_ROW_BOT  = 0.723      # y ≈ 175  (FPS/ping debug line below is dropped by the OCR height filter)

# Vertical offset from squads-row centre to stats-row centre (panel-height fraction)
TR_ROW_Y_OFFSET = (
    (TR_STATS_ROW_TOP  + TR_STATS_ROW_BOT)  / 2
    - (TR_SQUADS_ROW_TOP + TR_SQUADS_ROW_BOT) / 2
)

# Stat icon matching
ICON_REF_PANEL_H   = 242    # height of the panel the icons were cropped from
ICON_MATCH_THRESH  = 0.45   # template-match confidence threshold
ICON_NUMBER_WINDOW = 120    # upscaled pixels to search past icon right-edge

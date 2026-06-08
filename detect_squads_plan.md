# Plan: Squad Rank Distribution detector

Goal: on the **RANKED_LOADING** screen, read the bottom "SQUAD RANK DISTRIBUTION"
bar and return, per rank, how many squads are that rank — e.g.
`{"Diamond": 13, "Master": 4, "Apex Predator": 3}`.

Run everything with `python3` (Homebrew 3.11). Test against all four reference
screens: `images/ui-images/ranked-loading-{1,2,3,4}.png` (all 2808×1570).

## What the bar looks like (verified)

The bar is a horizontal, dark-bordered rounded rectangle centred at the bottom,
with the text "SQUAD RANK DISTRIBUTION" above it and rank **icons** above each
segment. It is divided left→right into colour segments, one per rank present,
**ordered low→high rank** (lower ranks on the left). Each segment shows its squad
count as white text centred in the segment. Segment **width is proportional to
count**.

Ground truth from the references (note **every screen sums to 20 squads**):

| screen | segments (rank = count) | total |
|--------|--------------------------|-------|
| ranked-loading-1 | Diamond 13, Master 4, Apex Predator 3 | 20 |
| ranked-loading-2 | Diamond 16, Master 3, Apex Predator **1** | 20 |
| ranked-loading-3 | Diamond 16, Master 3, Apex Predator **1** | 20 |
| ranked-loading-4 | Diamond 20 (single full-width segment)  | 20 |

The `=1` predator segments in screens 2 & 3 are the hard edge case: the segment is
~1/20 of the bar (~5% width), so the "1" is tiny and OCR is unreliable.

### Measured fill colours (OpenCV HSV, brightest interior pixel)

Sampled from the bar interior; use these as the colour→rank table:

| rank | colour | H (0‑179) | S | V | notes |
|------|--------|-----------|---|---|-------|
| Apex Predator | red    | 0–5 (or 175–179) | ~165 | ~236 | wraps at 0 |
| Master        | purple | 128–136 | ~140 | ~219 | |
| Diamond       | blue   | 100–108 | 120–170 | ~240 | |
| Platinum      | teal/cyan | ~88–98 (greenish-blue) | — | — | **estimate, no sample — verify** |
| Gold          | gold/yellow | ~20–32 | high | — | **estimate — verify** |
| Silver        | grey   | any H, **low S (<60)**, mid V | — | — | **estimate — verify** |
| Bronze        | brown/orange | ~8–18 | mid | mid V | **estimate — verify** |

Diamond (blue ~104) and Platinum (teal ~92) are the closest pair — rely on the
**icon** to disambiguate them, not colour alone.

The bar's dark border/dividers read as near-black (V<40); use that to find
segment boundaries.

## Rank identity: colour + icon (both, as the user asked)

`images/rank-icons/` holds clean reference icons:
`bronze, silver, gold, platinum, diamond, master, apex_predator` (each is a
4-channel PNG with alpha — composite over a mid-grey or use grayscale).

Use **two signals per segment** and require agreement:
1. **Colour** of the segment fill → rank via the table above (fast, reliable for
   the distinct hues).
2. **Icon** above the segment → template-match the segment's icon crop against all
   seven reference icons (grayscale `cv2.matchTemplate`, `TM_CCOEFF_NORMED`, best
   score wins), reusing the scaling approach in
   `TopRightDetector._match_icons` (`detectors.py:343`).

If colour and icon disagree, trust the icon (it disambiguates teal/blue and the
muted gold/silver/bronze). Colour is the safety net when an icon match is weak.

The icon is centred above its segment's centre-x and is full-size even when the
segment is a 1-squad sliver — so **icon detection stays reliable on the hard
edge case** even though the number does not.

## Algorithm

Add a method to `RankedLoadingDetector` in `detectors.py`
(e.g. `detect_squad_distribution(full_bgr) -> dict[str, int] | None`), alongside
the existing `detect_map_name`. Wire constants into `config.py` next to the other
`RANKED_LOADING_*` values. Reuse `RANK_PROGRESSION` ordering for rank names.

### Step 1 — Locate the bar
Search the bottom band of the screen (~`y` 0.88–1.0, `x` 0.30–0.70 — refine).
Build an HSV mask of "vivid bar fill" = saturated (S>80) **and** bright (V>120)
pixels. The bar is the dominant horizontal blob in this band.
- Take the bounding box of that blob → `(bar_x1, bar_y1, bar_x2, bar_y2)`.
- Optionally confirm position by OCR-ing "SQUAD RANK DISTRIBUTION" just above and
  anchoring to it (mirrors `detect_map_name`), but the colour blob alone should
  suffice.
- Sample colour from a single representative row at the **vertical centre** of the
  bar (`(bar_y1+bar_y2)//2`) to avoid the dark top/bottom border.

Store the bar region as height/width **fractions** in `config.py` so it survives
the 2808×1570 vs other resolutions (the project also runs at 2726/2778 widths) —
follow the fraction convention already used everywhere else.

### Step 2 — Segment the bar by colour
Walk the centre row left→right, classify each column into one of
{predator, master, diamond, platinum, gold, silver, bronze, **border/none**}
using the colour table. Smooth/denoise (e.g. require a run ≥ N px) and split into
**contiguous runs**. Each run with a real rank colour = one segment. Dark dividers
between segments become short border runs and are dropped.

For each segment record: `rank_by_colour`, `x1`, `x2`, `width`, `cx=(x1+x2)//2`.

Sanity: segments should be left→low / right→high in `RANK_PROGRESSION` order — use
this to reject spurious tiny mis-coloured runs (e.g. antialiasing at a divider).

### Step 3 — Confirm rank via icon
For each segment, crop an icon-sized box centred at `cx`, located **above** the
bar (the icon row sits just above `bar_y1`; find its `y` once by template-matching
the strongest icon and reuse for all). Template-match against the seven reference
icons; take the best `TM_CCOEFF_NORMED` score. Final `rank` = icon match if its
score ≥ threshold (calibrate ~0.5), else fall back to `rank_by_colour`.

### Step 4 — Read the count per segment (with fallbacks)
Primary: OCR the white number centred in each segment.
- Crop the segment fill (`x1..x2`, bar y-range), threshold near-white text
  (`gray > ~180`), upscale ~3–4× (`INTER_CUBIC`), `reader.readtext(..,
  allowlist="0123456789")`. Reuse the binarize-then-upscale pattern from
  `TopRightDetector._prep_ocr` / `_ocr_digits` (`detectors.py:325,371`).
- Keep the highest-confidence pure-digit hit whose box is reasonably tall.

Fallbacks for tiny / failed segments (the `=1` predator case):
1. **Width proportion.** `count_est_i = round(TOTAL * width_i / sum(width))`.
   With `TOTAL=20` this directly yields 1 for a ~5%-width sliver. Verified to match
   all four references.
2. **Sum constraint.** The bar represents a full ranked lobby = **20 squads**, so
   `sum(counts) == 20`. If exactly one segment's OCR is missing/low-conf, set it to
   `20 - sum(others)`.

Reconciliation order:
- If every segment OCR'd and `sum == TOTAL` → accept OCR.
- Else fill missing/low-conf segments from the width estimate.
- If still `sum != TOTAL`, correct the **single least-confident** (usually
  smallest) segment to force `sum == TOTAL`; if more than one is uncertain, trust
  the width proportions over OCR.

Make `TOTAL` a config constant (`RANKED_SQUAD_TOTAL = 20`) and treat it as a strong
prior, not an unbreakable rule — log a warning if confident OCR contradicts it
rather than silently overriding (lobbies are 20 squads at drop, but keep it
overridable).

### Step 5 — Return
`{rank_name: count}` for each detected segment, rank names matching
`RANK_PROGRESSION` strings (base ranks: "Diamond", "Master", "Apex Predator", …).
Return `None` if no bar is found (so the caller can hold last state, like the rest
of the pipeline).

## Edge cases to handle explicitly
- **Single full-width segment** (screen 4): one colour spans the whole bar → one
  rank = 20. No dividers. Don't require ≥2 segments.
- **1-squad sliver** (screens 2,3): OCR will likely fail → resolved by width
  proportion + sum-to-20. This is the primary thing to get right.
- **Gold/Silver/Bronze**: no reference images exist. Colour table values for these
  are *estimates* — the icon match must carry identity here. Add a TODO to verify
  hues once a real screen appears. Silver especially (low saturation) may collide
  with the grey border; lean on the icon.
- **Adjacent similar colours**: Diamond↔Platinum — disambiguate by icon.
- **Antialiasing at dividers**: enforce a minimum run width and the
  low→high rank-order monotonicity to discard 1–2px false segments.

## How to build & verify (do this incrementally)
1. Write a throwaway `inspect-squads.py` that, for each `ranked-loading-{1..4}`,
   draws the detected bar box, segment boundaries, per-segment colour label, icon
   match, and OCR number onto a debug image in `/tmp/` — eyeball it (this mirrors
   `inspect-game.py` / `inspect-experience.py`).
2. Calibrate the colour ranges and bar-location fractions against screens 1–4.
3. Confirm results exactly: 1→{D:13,M:4,P:3}, 2→{D:16,M:3,P:1},
   3→{D:16,M:3,P:1}, 4→{D:20}.
4. Add a test (follow `test-session.py` / `test-integration.py` style) asserting
   those four outputs.
5. Verify icon template-matching actually fires: crop the on-screen badge above a
   segment and confirm the correct reference icon wins by a clear margin; tune the
   match scale (icon-on-screen height vs reference icon height) and threshold.

## Files to touch
- `detectors.py` — new `detect_squad_distribution` on `RankedLoadingDetector`;
  reuse `_match_icons`-style matching and `_prep_ocr`-style OCR.
- `config.py` — `RANKED_DIST_*` bar-location fractions, colour HSV ranges,
  `RANKED_SQUAD_TOTAL = 20`, icon dir for rank icons (`images/rank-icons`).
- new test file (e.g. `test-squad-distribution.py`).
- `session.py` — optionally call it on RANKED_LOADING frames and store the result
  (mirrors how `_map_name` is handled at `session.py:143`).

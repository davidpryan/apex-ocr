# Experience Screen Detection — Improvement Plan

## Current State

Four fields are implemented and working on two of three reference screens:

| Field | screen-1 | screen-2 | screen-3 |
|---|---|---|---|
| `current_rank` | Platinum IV ✓ | Platinum IV ✓ | — (not classified) |
| `next_rank` | Platinum III ✓ | Platinum III ✓ | — |
| `point_change` | +363 ✓ | +246 ✓ | — |
| `current_rp` | 186 ✓ | 509 ✓ | — |

All other fields in `ExperienceDetector.detect()` return `None` (stub).

---

## Known Issues

### 1. Screen Classifier fails on exp-screen-3

**Score:** 0.6235 (threshold: 0.65 — just below).  
**Cause:** exp-screen-3 has two requeue buttons in the top-right corner instead of one, creating a slight colour difference in the top bar region that the template uses. The "LIVE / SUMMARY" area is identical but the surrounding area shifts the normalised correlation score.  
**Fix:** Lower threshold to 0.60, which still provides clean separation (game screen scores 0.26). Alternatively extract a tighter template that excludes the right-hand area where the requeue buttons appear (e.g. cap x at 55% of screen width instead of 70%).

### 2. `current_rp` returns None when bar is near-zero

**Cause:** `_detect_current_rp` uses HSV yellow detection to find the bar's right edge. When the bar is near-empty (exp-screen-3 shows 0 RP), there are no yellow pixels and the method short-circuits to `None`.  
**Fix:** Treat "no yellow detected" as current RP = 0. The zero case is valid; the current early-return should become `return "0"`.

### 3. Combat RP Total misread as "7189 RP" (screen-2)

**Cause:** Same rank-badge digit-merging problem as `current_rp`. The "7" is a fragment from a nearby icon being merged by EasyOCR into the "188" value.  
**Fix:** Apply the same connected-component portrait-aspect-ratio filter used for `current_rp` to the combat values, OR use the section-header-anchored crop approach described in §Section-based layout below.

### 4. All COMBAT / BONUSES / MATCH PLACEMENT fields are stubs

The OCR dump confirms all values are legible across all three screens. They need crops and parsers.

---

## Values to Implement

### Upper section (already implemented)
| Field | Example values | Notes |
|---|---|---|
| `point_change` | +363, +246, +15 | Signed int above "Ranked Points Earned" |
| `current_rp` | 186, 509, 0 | On yellow progress bar; 0 is a valid edge case |
| `current_rank` | Platinum IV | Derived: one step before `next_rank` in progression |
| `next_rank` | Platinum III | "NEXT RANK: ..." text below bar |

### COMBAT section (y ≈ 54–73% of screen)
| Field | screen-1 | screen-2 | screen-3 |
|---|---|---|---|
| `combat_rp_total` | 165 RP | 188 RP | 8 RP |
| `kills` | 6 | 12 | 0 |
| `kills_rp` | 120 RP | 216 RP | 0 RP |
| `assists` | 1 | 0 | 0 |
| `assists_rp` | 20 RP | 0 RP | 0 RP |
| `participations_formula` | 3×50%−15 | 2×50%−1.0 | 1×50%−0.5 |
| `participations_rp` | 30 RP | 18 RP | 8 RP |
| `kill_cap_adjustment_rp` | present | −45 RP | 0 RP |

### BONUSES section (y ≈ 54–73%, centre column)
| Field | screen-1 | screen-2 | screen-3 |
|---|---|---|---|
| `bonus_rp_total` | 286 RP | (not legible) | 0 RP |
| `challenger_count` | 4 | 0 | 0 |
| `challenger_rp` | 16 RP | 0 RP | 0 RP |
| `top5_streak` | 3/5 | 4/5 | 1/5 |
| `top5_streak_rp` | 20 RP | 30 RP | 0 RP |
| `promotion_rp` | 250 RP | — | — |

### MATCH PLACEMENT section (y ≈ 54–73%, right column)
| Field | screen-1 | screen-2 | screen-3 |
|---|---|---|---|
| `placement_rp_total` | 62 RP | 75 RP | 7 RP |
| `placement` | #2 | #3 | #4 |
| `placement_rp` | 100 RP | 75 RP | 55 RP |
| `cost_of_entry_tier` | Gold I | Platinum IV | Platinum IV |
| `cost_of_entry_rp` | −38 RP | −48 RP | −48 RP |

---

## Implementation Plan

### A. Fix the classifier (immediate, low-risk)

1. Lower `SCREEN_CLASSIFY_THRESHOLD` from 0.65 → 0.60 in `config.py`.
2. Optionally cap the template's x range to 55% of screen width (removes the requeue button area that varies between screens).
3. Re-verify against all three screens and the game screen.

### B. Fix `current_rp` zero-bar edge case

In `_detect_current_rp`, change the early-return when no yellow is found:
```python
if not col_yellow.any():
    return "0"   # bar is at the start of the tier
```

### C. Section-based layout approach

Instead of fixed y-fractions from `bar_bottom_y` for the lower three sections, detect the section headers ("COMBAT", "BONUSES", "MATCH PLACEMENT") by OCR and use their y-coordinates as anchors. This makes all crops self-calibrating.

```
COMBAT header found at y=C
  → combat_rp_total row: y ≈ C + row_gap
  → kills row: y ≈ C + 2*row_gap
  → assists row: y ≈ C + 3*row_gap
  → participations row: y ≈ C + 4*row_gap
  → kill_cap_adjustment (if present): y ≈ C + 5*row_gap
```

The row gap is consistent (~52–55px on 1558px tall screens). Run one broad OCR pass on the lower 50% to locate all three headers, then derive all value crops from their positions.

### D. Number parsing helpers

Several fields need specific parsers beyond a simple digit regex:

| Field | Raw OCR | Parser needed |
|---|---|---|
| `combat_rp_total` | "165 RP", "7189 RP" | Strip leading noise digit if > 999; extract last 3 digits |
| `participations_formula` | "3×50%−15", "1×50%−0.5" | Regex `(\d+)×(\d+)%[−-](\d+\.?\d*)` |
| `top5_streak` | "3/5", "4/5" | Regex `(\d)/5` |
| `placement` | "#2", "#3" | Regex `#(\d+)` |
| `cost_of_entry_rp` | "~38 RP", "−48 RP" | Regex `[~−-]?(\d+)\s*RP`; negate |
| `cost_of_entry_tier` | "Gold I", "Platinum IV" | Match against RANK_LOOKUP |

### E. White-on-yellow robustness for value labels

For fields in the BONUSES column where values appear against the yellow highlighted background (e.g. `bonus_rp_total`), apply the same connected-component portrait-filter pipeline used for `current_rp`.

---

## Best Practices for Game UI Text Detection

### 1. Anchor to detected text, not fixed fractions
Fixed fractions break when the game window isn't fullscreen or at a different resolution. Prefer finding a stable text anchor (e.g. "Ranked Points Earned", "COMBAT") via OCR, then computing all sub-crops relative to its bounding box.

### 2. White-pixel threshold before OCR
Game HUD text is almost always white regardless of background colour (dark panel, yellow bar, etc.). Thresholding at brightness > 200 before OCR normalises contrast across all background types and avoids background-colour-specific preprocessing paths.

### 3. Connected component portrait filter
After thresholding, run `connectedComponentsWithStats` and keep only components where:
- height ≥ 40px (at 4× upscale) — removes small noise
- width/height < 1.2 — removes wide icon fragments and horizontal bar artefacts
- area > 300px — removes single-pixel noise

This reliably removes rank badge chevrons, progress bar fills, and UI decoration while preserving digit strokes.

### 4. Upscale before OCR; use allowlists
Upscale 3–4× with `cv2.INTER_CUBIC` before passing to EasyOCR. Use `allowlist="0123456789RP "` for numeric fields — this halves the character search space and eliminates icon-glyph misreadings like `S→5`, `O→0`, `l→1`.

### 5. Left-heavy crops around the yellow bar marker
The current RP number sits to the left of where the yellow bar ends when the bar is mostly filled, and to the right when mostly empty. Always extend the crop further left than right from the bar's right edge (15% left, 4% right) to capture the number in both states.

### 6. Range validation before accepting
After extracting a digit string, validate against the expected range:
- current_rp: [0, 750] (or up to tier max)
- placement: [1, 60]
- RP values: non-negative integers

Reject and retry with a looser crop rather than returning an out-of-range value.

### 7. Multiple template variants for screen classification
Maintain separate templates for each visual state of the LIVE/SUMMARY bar (tabs selected, different requeue button counts, etc.). Try each in sequence; classify as EXPERIENCE if any exceeds the threshold. Templates should be narrow horizontal slices of the consistent purple region to minimise sensitivity to surrounding UI chrome.

### 8. Sticky last-known-good values
Never overwrite a valid detection with None (`update_if_valid`). If OCR fails on a frame, the previous valid value persists until a new valid one is found. This is especially important for slowly-changing fields like `current_rank`.

### 9. Section presence is conditional
`kill_cap_adjustment` and `promotion_rp` are absent in some matches. Design parsers to return `None` (not error) when the label is not detected, and document which fields are always present vs optional.

### 10. Validate with multiple reference images before shipping
For each new crop/parser, confirm correct output on all three reference screens before treating it as done. Fields that work on one screen commonly fail on another due to:
- Different bar fill levels (current_rp)
- Different active tab highlighting (classifier)
- Zero-value rows that EasyOCR may skip (kills=0, assists=0)
- Number of requeue buttons changing top-bar appearance

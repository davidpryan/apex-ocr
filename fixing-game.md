# Fixing game-screen OCR (Top-Right HUD)

## Goal
Read the top-right HUD correctly on **all** game screens, not just bright-background
ones. Reference ground truth:

| field    | blue | purple |
|----------|------|--------|
| squads   | 16   | 10     |
| players  | 44   | 23     |
| kills    | 1    | 3      |
| assists  | 0    | 1      |
| particip | 2    | 2      |
| damage   | 69   | 1144   |

Before this work: blue = 6/6, purple = 0/6 (every value `None` or wrong).

## Diagnosis

The current `TopRightDetector._prep` only does `sharpen + upscale`. That is enough
when the HUD sits on a **bright** background (blue screen → sky), but collapses when
the HUD sits on a **dark, textured** background (purple screen → red rock):

1. **Low contrast.** Raw OCR confidence on the purple stats row drops to ~0.00–0.03.
2. **Icon interference.** The white skull/handshake/ball/airplane icons pass any
   brightness threshold and EasyOCR reads them as characters, corrupting the
   adjacent digit (kills `3` merges with the skull → `0`).
3. **Icon template-matching fails.** Matching the grayscale icon templates against
   the dark background scores 0.28–0.44 (< 0.45 threshold), so the per-stat
   number windows are never located and the positional fallback OCRs the whole
   contaminated row.
4. **Debug-text bleed.** The `FPS / ping` overlay line sits just under the stats row
   and falls inside the band, producing long garbage tokens.
5. **Mixed polarity.** The highlighted stat cell is *dark digit on white*, while the
   others are *white digit on dark* — a single threshold direction can't read both.

## What reliably works (implemented)

- **Binarization preprocessing** (`threshold`) for the squads/players row and the
  stats row. This isolates HUD text from any background and **fixes squads + players
  on purple (10, 23)** while keeping blue (16, 44).
- **OCR bounding-box height filtering**: drop any OCR token shorter than ~30 % of the
  row height. Removes the FPS/ping debug tokens.
- **Tighter stats band bottom** so the debug line is excluded from the crop.
- Icon matching kept for the bright path (blue, score 1.0); positional fallback for
  the dark path.

## What remains hard (documented, not fully solved)

The four **purple stat digits** (3, 1, 2, 1144) are degraded beyond reliable
single-frame OCR: when binarized, the thin digits merge into ambiguous blobs on the
busy background (`3`→`0`), and the white-highlighted participation cell inverts
polarity. Per-cell, dual-polarity, Otsu, icon-masking — none recover all four on
this single frame.

### Recommended production strategy for the hard digits
This detector runs on **video at 30 fps**; a stat value is stable for many seconds
(hundreds of frames). The robust answer is **temporal aggregation**: keep a
confidence-weighted running value per stat and only update when a frame reads it with
high confidence. As the player moves, the background behind the HUD changes and an
easier frame will read the true value. A single worst-case still frame should not be
expected to yield all four digits.

## Plan executed
1. Added `_prep_ocr()` (threshold-binarized upscale) alongside `_prep()` (tonal).
2. Squads row: OCR binarized **without** a digit allowlist so "16 SQUADS LEFT"
   stays one token; squads via OCR-tolerant `(\d+)\s*S?QUA[D0O]S` regex, players via
   the rightmost pure-digit token.
3. Stats row: `_ocr_digits` now OCRs **both** tonal and binarized renders and merges
   — the binary pass only fills positions the tonal pass missed (so bright screens
   are unchanged, dark screens gain digits). No allowlist (keeps the airplane icon
   from becoming a spurious "4"). Height filter drops FPS/ping debug tokens.
4. Damage cleanup: drop leading digit(s) when the value exceeds 4 digits (icon
   artifact), mirroring `_parse_rp`'s badge-merge handling.
5. Kept `TR_STATS_ROW_BOT` at 0.723 (tightening broke icon-template fit; the height
   filter handles the debug line instead).

## Results

| field    | blue (was 6/6) | purple (was 0/6) |
|----------|:--------------:|:----------------:|
| squads   | 16 ✓           | **None** ✗ ("10"→"JU") |
| players  | 44 ✓           | 23 ✓ (fixed)     |
| kills    | 1 ✓            | 3 ✓ (fixed)      |
| assists  | 0 ✓            | **None** ✗ (thin "1") |
| particip | 2 ✓            | 2 ✓ (fixed)      |
| damage   | 69 ✓           | 1144 ✓ (fixed)   |
| **total**| **6/6**        | **4/6**          |

Blue is preserved at 6/6; purple improved from 0/6 to 4/6.

### The two remaining purple misses
Both are a **thin "1"** digit on the dark/busy background: squads "10" binarizes to
an ambiguous "JU"; assists "1" is too faint to detect at all. These are the
worst-case single-frame reads. The production fix is **temporal aggregation** across
video frames (see above) — not further single-frame preprocessing, which risks
regressing the bright-screen path.

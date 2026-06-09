"""
Extract POI polygons + names from the labelled map renders and re-project them
onto the base maps in images/maps/.

For each images/labeled-maps/labeled-<map>.png it:
  1. Registers the labelled render onto the matching base map via SIFT + a
     similarity transform (RANSAC).  The labelled and base renders share the
     underlying map art, so the white polygon/label overlay is rejected as
     outliers during fitting.
  2. Extracts POI polygons by sealing the white outline layer and flood-filling
     the exterior — each enclosed cell is one POI (separates touching polygons).
  3. OCRs the label text and assigns each text box to the polygon that contains
     it, joining fragments in reading order ("Survey" + "Camp" → "Survey Camp").
  4. Transforms polygon vertices + centroid into base-map pixel coordinates.

Outputs, per map:
  images/maps/<map>_pois.json        {map, image_size, registration, pois:[…]}
  images/maps/<map>_pois_preview.png base map with polygons + names drawn

OCR is imperfect (character clipping, leading glyphs) — open each preview and
fix the few wrong names in the JSON.  Polygons and centres are reliable.

Usage:
  python3 extract-pois.py                 # all labelled maps
  python3 extract-pois.py worlds_edge     # one map
"""

import glob
import json
import os
import re
import sys

import cv2
import easyocr
import numpy as np

HERE       = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR   = os.path.join(HERE, "images", "maps")
LABELED_DIR = os.path.join(HERE, "images", "labeled-maps")

# Tuning constants
WHITE_THRESH       = 190     # min B,G,R for an overlay (polygon/label) pixel
GAP_CLOSE_KERNEL   = 9       # seal antialiasing gaps in outlines
REGION_DILATE      = 5       # grow each interior cell out to the outline midline
MIN_AREA_FRAC      = 0.0008  # min interior area as fraction of the labelled image
POLY_EPS_FRAC      = 0.012   # approxPolyDP epsilon as fraction of contour perimeter
OCR_MIN_CONF       = 0.25
SIFT_FEATURES      = 6000
LOWE_RATIO         = 0.75


def base_name_for(labeled_path: str) -> str:
    """labeled-worlds-edge.png → worlds_edge"""
    stem = os.path.splitext(os.path.basename(labeled_path))[0]
    return stem.replace("labeled-", "").replace("-", "_")


def _clean(text: str) -> str:
    """Strip leading/trailing non-alphanumeric OCR noise."""
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", text).strip()


def register(labeled_bgr: np.ndarray, base_bgr: np.ndarray):
    """Return (M 2x3 labeled→base, inliers, total_matches) or (None, 0, 0)."""
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    kb, db = sift.detectAndCompute(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY), None)
    kl, dl = sift.detectAndCompute(cv2.cvtColor(labeled_bgr, cv2.COLOR_BGR2GRAY), None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [m for m, n in bf.knnMatch(dl, db, k=2) if m.distance < LOWE_RATIO * n.distance]
    if len(good) < 10:
        return None, 0, len(good)
    src = np.float32([kl[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=6)
    return M, int(inl.sum()) if inl is not None else 0, len(good)


def extract_polygons(labeled_bgr: np.ndarray) -> list[np.ndarray]:
    """Return a list of polygon vertex arrays (in labelled-image coords)."""
    h, w = labeled_bgr.shape[:2]
    b, g, r = cv2.split(labeled_bgr)
    outline = ((b > WHITE_THRESH) & (g > WHITE_THRESH) & (r > WHITE_THRESH)).astype(np.uint8) * 255
    outline = cv2.morphologyEx(outline, cv2.MORPH_CLOSE,
                               np.ones((GAP_CLOSE_KERNEL, GAP_CLOSE_KERNEL), np.uint8))

    # Flood the exterior; enclosed free space (==1) are POI interiors.
    free = (outline == 0).astype(np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(free, mask, (0, 0), 2)
    interior = (free == 1).astype(np.uint8) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(interior, 8)
    min_area = MIN_AREA_FRAC * h * w
    polys = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        comp = cv2.dilate(comp, np.ones((REGION_DILATE, REGION_DILATE), np.uint8))
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        ap = cv2.approxPolyDP(c, POLY_EPS_FRAC * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(ap) >= 3:
            polys.append(ap)
    return polys


def assign_names(polys: list[np.ndarray], ocr_results) -> dict[int, str]:
    """Assign OCR text boxes to the containing polygon, joined in reading order."""
    buckets: dict[int, list] = {i: [] for i in range(len(polys))}
    for box, txt, conf in ocr_results:
        if conf < OCR_MIN_CONF:
            continue
        cx = float(np.mean([p[0] for p in box]))
        cy = float(np.mean([p[1] for p in box]))
        for i, poly in enumerate(polys):
            if cv2.pointPolygonTest(poly.astype(np.int32), (cx, cy), False) >= 0:
                buckets[i].append((cy, cx, _clean(txt)))
                break
    names = {}
    for i, parts in buckets.items():
        joined = " ".join(t for *_, t in sorted(parts) if t)
        names[i] = joined or None
    return names


def process_map(labeled_path: str, reader: easyocr.Reader) -> None:
    base = base_name_for(labeled_path)
    base_path = os.path.join(MAPS_DIR, f"{base}.png")
    if not os.path.exists(base_path):
        print(f"  [skip] no base map images/maps/{base}.png")
        return

    labeled = cv2.imread(labeled_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
    base_img = cv2.imread(base_path)
    Hb, Wb = base_img.shape[:2]

    M, inliers, total = register(labeled, base_img)
    if M is None:
        print(f"  [fail] registration failed ({total} matches) — maps may differ")
        return
    print(f"  registration: {inliers}/{total} inliers")

    polys = extract_polygons(labeled)
    ocr = reader.readtext(labeled, low_text=0.3, text_threshold=0.4)
    names = assign_names(polys, ocr)

    def to_base(pts: np.ndarray) -> np.ndarray:
        return cv2.transform(pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)

    pois = []
    for i, poly in enumerate(polys):
        pb = to_base(poly)
        cen = pb.mean(axis=0)
        if not (0 <= cen[0] < Wb and 0 <= cen[1] < Hb):
            continue   # drop off-map artifacts (legend boxes etc.)
        pois.append({
            "name":    names[i],
            "x":       int(round(cen[0])),
            "y":       int(round(cen[1])),
            "polygon": [[int(round(x)), int(round(y))] for x, y in pb],
        })

    named = sum(1 for p in pois if p["name"])
    out = {
        "map":        base,
        "image_size": [Wb, Hb],
        "source":     os.path.basename(labeled_path),
        "registration": {"inliers": inliers, "matches": total},
        "pois":       pois,
    }
    json_path = os.path.join(MAPS_DIR, f"{base}_pois.json")
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)

    # verification preview
    prev = base_img.copy()
    for p in pois:
        poly = np.array(p["polygon"], np.int32)
        cv2.polylines(prev, [poly], True, (60, 220, 255), 3)
        cv2.circle(prev, (p["x"], p["y"]), 6, (0, 0, 255), -1)
        label = p["name"] or "???"
        for col, th in [((0, 0, 0), 4), ((60, 220, 255), 1)]:
            cv2.putText(prev, label, (p["x"] + 8, p["y"]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
    prev_path = os.path.join(MAPS_DIR, f"{base}_pois_preview.png")
    cv2.imwrite(prev_path, prev)

    print(f"  POIs: {len(pois)}  named: {named}/{len(pois)}")
    print(f"  → {os.path.basename(json_path)}  +  {os.path.basename(prev_path)}")


def main() -> None:
    targets = sorted(glob.glob(os.path.join(LABELED_DIR, "labeled-*.png")))
    if len(sys.argv) > 1:
        want = sys.argv[1]
        targets = [t for t in targets if base_name_for(t) == want]
        if not targets:
            raise SystemExit(f"No labelled map for '{want}'")

    print("Loading EasyOCR model…")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    for t in targets:
        print(f"\n{base_name_for(t)}  ({os.path.basename(t)})")
        process_map(t, reader)


if __name__ == "__main__":
    main()

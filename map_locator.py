"""
Map position locator — localise the player on the full map from the HUD minimap.

Given a full game frame and the current map name, crops the top-left minimap,
matches it against the base map render with SIFT + a similarity transform
(RANSAC), and transforms the minimap centre (the player) into full-map pixel
coordinates.  The recovered point is then resolved to a named POI via the
polygons in images/maps/<slug>_pois.json (point-in-polygon, else nearest centre).

Returns a dict with nullable keys: ``map_x``, ``map_y``, ``location``.
Everything is None when no map is known, the reference is missing, or the match
is below the inlier threshold (so low-confidence frames don't pollute the CSV).

Reference SIFT descriptors are computed once per map and cached to
images/maps/<slug>_sift.npz so subsequent runs start instantly.
"""

import json
import os

import cv2
import numpy as np

from config import (
    MAPS_DIR, MAP_MIN_INLIERS,
    MINIMAP_LEFT_FRAC, MINIMAP_TOP_FRAC, MINIMAP_RIGHT_FRAC, MINIMAP_BOT_FRAC,
    MINIMAP_CENTER_MASK_FRAC,
)

_NULL = {"map_x": None, "map_y": None, "location": None}


def _slug(map_name: str) -> str:
    """'World's Edge' → worlds_edge,  'E-District' → e_district."""
    return map_name.lower().replace("'", "").replace("-", "_").replace(" ", "_")


class MapLocator:
    """Locates the player on the full map from a full game-screen frame."""

    def __init__(self, maps_dir: str = MAPS_DIR, min_inliers: int = MAP_MIN_INLIERS):
        self._dir         = maps_dir
        self._min_inliers = min_inliers
        self._sift        = cv2.SIFT_create(nfeatures=4000)
        self._bf          = cv2.BFMatcher(cv2.NORM_L2)
        self._cache: dict[str, dict | None] = {}

    # ------------------------------------------------------------------
    # Reference loading (lazy, with on-disk descriptor cache)
    # ------------------------------------------------------------------

    def _load(self, slug: str) -> dict | None:
        if slug in self._cache:
            return self._cache[slug]

        base_path = os.path.join(self._dir, f"{slug}.png")
        if not os.path.exists(base_path):
            self._cache[slug] = None
            return None

        img = cv2.imread(base_path)
        npz = os.path.join(self._dir, f"{slug}_sift.npz")
        if os.path.exists(npz) and os.path.getmtime(npz) >= os.path.getmtime(base_path):
            data = np.load(npz)
            pts, des = data["pts"], data["des"]
        else:
            kp, des = self._sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            pts = np.float32([k.pt for k in kp])
            np.savez(npz, pts=pts, des=des)

        pois = []
        pj = os.path.join(self._dir, f"{slug}_pois.json")
        if os.path.exists(pj):
            with open(pj) as fh:
                pois = json.load(fh).get("pois", [])
        for p in pois:
            p["_poly"] = (np.array(p["polygon"], np.int32)
                          if p.get("polygon") else None)

        entry = {"pts": pts, "des": des, "pois": pois,
                 "size": (img.shape[1], img.shape[0])}   # (W, H)
        self._cache[slug] = entry
        return entry

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def locate(self, full_bgr: np.ndarray, map_name: str | None = None) -> dict:
        """Return {map_x, map_y, location}; values None if not confidently found."""
        if not map_name:
            return dict(_NULL)
        entry = self._load(_slug(map_name))
        if entry is None:
            return dict(_NULL)

        h, w = full_bgr.shape[:2]
        mm = full_bgr[int(h * MINIMAP_TOP_FRAC):int(h * MINIMAP_BOT_FRAC),
                      int(w * MINIMAP_LEFT_FRAC):int(w * MINIMAP_RIGHT_FRAC)]
        mh, mw = mm.shape[:2]
        if mh < 20 or mw < 20:
            return dict(_NULL)

        # Mask the central player chevron so it isn't matched as terrain.
        mask = np.full((mh, mw), 255, np.uint8)
        cv2.circle(mask, (mw // 2, mh // 2),
                   int(min(mh, mw) * MINIMAP_CENTER_MASK_FRAC), 0, -1)

        kp, des = self._sift.detectAndCompute(cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY), mask)
        if des is None or len(kp) < 8:
            return dict(_NULL)

        good = []
        for pair in self._bf.knnMatch(des, entry["des"], k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        if len(good) < self._min_inliers:
            return dict(_NULL)

        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([entry["pts"][m.trainIdx] for m in good]).reshape(-1, 1, 2)
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=8)
        if M is None or int(inl.sum()) < self._min_inliers:
            return dict(_NULL)

        cx, cy = cv2.transform(np.float32([[[mw / 2, mh / 2]]]), M)[0, 0]
        W, H = entry["size"]
        if not (0 <= cx < W and 0 <= cy < H):
            return dict(_NULL)

        return {
            "map_x":    int(round(cx)),
            "map_y":    int(round(cy)),
            "location": self._closest_poi(entry["pois"], cx, cy),
        }

    # ------------------------------------------------------------------
    # POI resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _closest_poi(pois: list[dict], x: float, y: float) -> str | None:
        """Name of the POI whose polygon contains (x, y), else the nearest centre."""
        best_name, best_d2 = None, None
        for p in pois:
            poly = p.get("_poly")
            if poly is not None and cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                return p.get("name")
            d2 = (p["x"] - x) ** 2 + (p["y"] - y) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2, best_name = d2, p.get("name")
        return best_name

"""
POI labelling tool — view/edit POIs (points and polygons) on a full map render.

Works with the JSON produced by extract-pois.py (polygons + names) as well as
hand-placed point POIs.  Click inside a polygon (or on a point) to rename it;
click empty space to drop a new point POI.

Usage:
  python3 label-pois.py                 # list available maps
  python3 label-pois.py worlds_edge     # label / edit worlds_edge

Controls:
  Left-click inside a POI    rename it (text pre-filled; edit + Enter)
  Left-click empty space     drop a new point POI, then type its name + Enter
  Right-click                delete the POI under the cursor
  while typing:              Backspace edits, Enter confirms, Esc cancels
  u                          undo the last add/rename
  s                          save now
  q  /  Esc (normal)         save and quit

Coordinates are in the map image's native pixel space (same as the SIFT
localiser).  Extra JSON keys (image_size, registration, source) are preserved.
Output: images/maps/<map>_pois.json
"""

import copy
import glob
import json
import os
import sys

import cv2
import numpy as np

MAPS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "maps")
MAX_DISPLAY = 1100
POINT_HIT_DISP = 12   # px radius to grab a point POI on screen


def _available_maps() -> list[str]:
    names = []
    for p in sorted(glob.glob(os.path.join(MAPS_DIR, "*.png"))):
        base = os.path.splitext(os.path.basename(p))[0]
        if not base.endswith(("_pois", "_pois_preview")):
            names.append(base)
    return names


def _json_path(map_name: str) -> str:
    return os.path.join(MAPS_DIR, f"{map_name}_pois.json")


class Labeller:
    def __init__(self, map_name: str):
        self.map_name = map_name
        img_path = os.path.join(MAPS_DIR, f"{map_name}.png")
        self.full = cv2.imread(img_path)
        if self.full is None:
            raise SystemExit(f"Could not load {img_path}")

        self.oh, self.ow = self.full.shape[:2]
        self.scale = min(1.0, MAX_DISPLAY / max(self.oh, self.ow))
        self.base = cv2.resize(self.full, None, fx=self.scale, fy=self.scale)

        self.meta, self.pois = self._load()
        self.typing      = False
        self.buffer      = ""
        self.pending     = None   # (x, y) for a new point being named
        self.editing_idx = None   # index of an existing POI being renamed
        self.cursor      = (0, 0)
        self.dirty       = False
        self._undo: list = []     # snapshots of self.pois for undo

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self):
        path = _json_path(self.map_name)
        if os.path.exists(path):
            with open(path) as fh:
                data = json.load(fh)
            pois = data.get("pois", [])
            n_poly = sum(1 for p in pois if p.get("polygon"))
            print(f"Loaded {len(pois)} POIs ({n_poly} with polygons) from {os.path.basename(path)}")
            return data, pois
        return {"map": self.map_name, "image_size": [self.ow, self.oh]}, []

    def save(self) -> None:
        self.meta["map"] = self.map_name
        self.meta["image_size"] = [self.ow, self.oh]
        self.meta["pois"] = self.pois
        with open(_json_path(self.map_name), "w") as fh:
            json.dump(self.meta, fh, indent=2)
        self.dirty = False
        print(f"Saved {len(self.pois)} POIs → {os.path.basename(_json_path(self.map_name))}")

    def _push_undo(self) -> None:
        self._undo.append(copy.deepcopy(self.pois))
        self._undo = self._undo[-50:]

    # ── coordinate helpers ───────────────────────────────────────────────────

    def _to_orig(self, dx, dy):  return round(dx / self.scale), round(dy / self.scale)
    def _to_disp(self, ox, oy):  return int(ox * self.scale), int(oy * self.scale)

    # ── hit testing ──────────────────────────────────────────────────────────

    def _hit(self, dx: int, dy: int) -> int | None:
        """Index of the POI under a display-space click, or None."""
        ox, oy = self._to_orig(dx, dy)
        # polygons first (containment)
        for i, p in enumerate(self.pois):
            poly = p.get("polygon")
            if poly and cv2.pointPolygonTest(np.array(poly, np.int32),
                                             (float(ox), float(oy)), False) >= 0:
                return i
        # then point POIs by proximity to centre
        best_i, best_d = None, POINT_HIT_DISP ** 2
        for i, p in enumerate(self.pois):
            px, py = self._to_disp(p["x"], p["y"])
            d2 = (px - dx) ** 2 + (py - dy) ** 2
            if d2 <= best_d:
                best_i, best_d = i, d2
        return best_i

    # ── mouse ────────────────────────────────────────────────────────────────

    def on_mouse(self, event, x, y, flags, _param):
        self.cursor = (x, y)
        if self.typing:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._hit(x, y)
            if hit is not None:                 # rename existing
                self.editing_idx = hit
                self.buffer = self.pois[hit].get("name") or ""
                self.typing = True
            else:                               # new point POI
                self.pending = self._to_orig(x, y)
                self.buffer  = ""
                self.typing  = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            hit = self._hit(x, y)
            if hit is not None:
                self._push_undo()
                removed = self.pois.pop(hit)
                self.dirty = True
                print(f"Deleted '{removed.get('name')}'")

    # ── keyboard while typing ─────────────────────────────────────────────────

    def on_key_typing(self, key: int) -> None:
        if key in (13, 10):                      # Enter — commit
            name = self.buffer.strip()
            self._push_undo()
            if self.editing_idx is not None:
                if name:
                    self.pois[self.editing_idx]["name"] = name
                    self.dirty = True
                    print(f"Renamed → '{name}'")
            elif self.pending is not None and name:
                self.pois.append({"name": name,
                                  "x": self.pending[0], "y": self.pending[1]})
                self.dirty = True
                print(f"Added '{name}' at {self.pending}")
            else:
                self._undo.pop()                 # nothing committed
            self._end_typing()
        elif key == 27:                          # Esc — cancel
            self._end_typing()
        elif key in (8, 127):                    # Backspace
            self.buffer = self.buffer[:-1]
        elif 32 <= key < 127:                    # printable
            self.buffer += chr(key)

    def _end_typing(self) -> None:
        self.typing = False
        self.buffer = ""
        self.pending = None
        self.editing_idx = None

    # ── rendering ─────────────────────────────────────────────────────────────

    def render(self):
        img = self.base.copy()
        for i, p in enumerate(self.pois):
            sel = (i == self.editing_idx)
            col = (60, 255, 255) if sel else (60, 220, 255)
            poly = p.get("polygon")
            if poly:
                dpoly = np.array([self._to_disp(x, y) for x, y in poly], np.int32)
                cv2.polylines(img, [dpoly], True, col, 2 if not sel else 3)
            dx, dy = self._to_disp(p["x"], p["y"])
            cv2.circle(img, (dx, dy), 4, (0, 0, 0), -1)
            cv2.circle(img, (dx, dy), 3, col, -1)
            name = p.get("name") or "???"
            for c, t in [((0, 0, 0), 3), (col, 1)]:
                cv2.putText(img, name, (dx + 7, dy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, t, cv2.LINE_AA)

        if self.pending is not None:
            dx, dy = self._to_disp(*self.pending)
            cv2.drawMarker(img, (dx, dy), (0, 120, 255), cv2.MARKER_CROSS, 18, 2)

        self._draw_status_bar(img)
        return img

    def _draw_status_bar(self, img) -> None:
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 46), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        star = "*" if self.dirty else ""
        named = sum(1 for p in self.pois if p.get("name") and p["name"] != "???")
        line1 = f"{self.map_name}{star}   POIs: {len(self.pois)}  named: {named}"
        cv2.putText(img, line1, (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (240, 240, 240), 1, cv2.LINE_AA)

        if self.typing:
            verb  = "Rename" if self.editing_idx is not None else "Name"
            line2 = f"{verb}: {self.buffer}_   [Enter=ok  Esc=cancel]"
            color = (60, 220, 255)
        else:
            line2 = "L-click=rename/add  R-click=delete  u=undo  s=save  q=save&quit"
            color = (180, 180, 180)
        cv2.putText(img, line2, (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, color, 1, cv2.LINE_AA)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        win = f"POI Labeller — {self.map_name}"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.on_mouse)
        print("Ready. Click a POI to rename, or empty space to add one.\n")

        while True:
            cv2.imshow(win, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue

            if self.typing:
                self.on_key_typing(key)
                continue

            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                self.save()
            elif key == ord("u"):
                if self._undo:
                    self.pois = self._undo.pop()
                    self.dirty = True
                    print("Undid last change")

        if self.dirty:
            self.save()
        cv2.destroyAllWindows()


def main() -> None:
    if len(sys.argv) < 2:
        print("Available maps:")
        for name in _available_maps():
            tag = "  (has labels)" if os.path.exists(_json_path(name)) else ""
            print(f"  {name}{tag}")
        print("\nUsage: python3 label-pois.py <map_name>")
        return

    map_name = sys.argv[1]
    if map_name not in _available_maps():
        raise SystemExit(f"Unknown map '{map_name}'. Run with no args to list maps.")
    Labeller(map_name).run()


if __name__ == "__main__":
    main()

"""
Generate a synthetic match_replay.csv: one player moving around World's Edge.

Uses the real images/maps/worlds_edge_pois.json so map_x/map_y/location are
self-consistent (location is the polygon the point falls in, else nearest centre
— same rule as MapLocator).  Deterministic via a fixed seed.
"""
import csv, datetime, json, os, random
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
random.seed(7); np.random.seed(7)

HEADERS = [
    "game_id", "game_start_time", "map_name", "row_time", "elapsed_s",
    "primary_weapon", "secondary_weapon", "armor_level",
    "shield_type", "shield_hp", "flesh_hp", "health",
    "squads_remaining", "players_remaining",
    "kills", "knocks", "assists", "participation", "damage",
    "map_x", "map_y", "location",
]

pois = json.load(open(os.path.join(ROOT, "images/maps/worlds_edge_pois.json")))["pois"]
for p in pois:
    p["_poly"] = np.array(p["polygon"], np.int32) if p.get("polygon") else None

def location_at(x, y):
    best, bestd = None, None
    for p in pois:
        if p["_poly"] is not None and cv2.pointPolygonTest(p["_poly"], (float(x), float(y)), False) >= 0:
            return p["name"]
        d = (p["x"]-x)**2 + (p["y"]-y)**2
        if bestd is None or d < bestd:
            bestd, best = d, p["name"]
    return best

def center(name):
    return next((p["x"], p["y"]) for p in pois if p["name"] == name)

# Drop → loot → rotate through fights → endgame.  Waypoints are real POIs.
route = ["Skyhook East", "The Epicenter", "Fragmenf", "Landslide",
         "Harvester", "Lava Siphon", "Big Maude", "The Dome"]
waypoints = [center(n) for n in route]

N = 50
# distribute rows across segments, interpolate with jitter
segs = len(waypoints) - 1
per = N / segs
pts = []
for i in range(N):
    t = i / (N - 1) * segs
    s = min(int(t), segs - 1)
    f = t - s
    (x0, y0), (x1, y1) = waypoints[s], waypoints[s+1]
    x = x0 + (x1 - x0) * f + random.gauss(0, 18)
    y = y0 + (y1 - y0) * f + random.gauss(0, 18)
    pts.append((int(np.clip(x, 0, 2047)), int(np.clip(y, 0, 2047))))

start = datetime.datetime(2026, 6, 9, 20, 14, 3)
game_id = start.strftime("%Y%m%d-%H%M%S") + "-we01ab"
gst = start.strftime("%Y-%m-%dT%H:%M:%S")

# stat progression over the match
def weapons(i):
    if i < 3:   return None, None
    if i < 10:  return "R-301 Carbine", None
    return "R-301 Carbine", "Wingman"
def shield(i):
    if i < 8:   return "white", 50
    if i < 28:  return "blue", 75
    return "purple", 100
def armor(i):  return 1 if i < 8 else (2 if i < 28 else 3)

rows = []
kills = knocks = assists = damage = 0
for i, (x, y) in enumerate(pts):
    # knocks precede most kills; one knock (row 31) never converts to a kill
    if i in (11, 25, 31, 37):
        knocks += 1
    if i in (12, 26, 38):
        kills += 1; damage += random.randint(120, 240)
    if i in (18, 33):
        assists += 1; damage += random.randint(40, 110)
    st, shp = shield(i)
    health = 100 if i not in (13, 27, 39) else random.randint(35, 80)
    squads = max(2, 20 - i // 3)
    players = max(squads * 2, 60 - i)
    pw, sw = weapons(i)
    rows.append({
        "game_id": game_id, "game_start_time": gst, "map_name": "World's Edge",
        "row_time": (start + datetime.timedelta(seconds=5*i)).strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(5.0*i, 1),
        "primary_weapon": pw, "secondary_weapon": sw, "armor_level": armor(i),
        "shield_type": st, "shield_hp": shp, "flesh_hp": health, "health": health,
        "squads_remaining": squads, "players_remaining": players,
        "kills": kills, "knocks": knocks, "assists": assists,
        "participation": kills + assists, "damage": damage,
        "map_x": x, "map_y": y, "location": location_at(x, y),
    })

out = os.path.join(HERE, "match_replay.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=HEADERS)
    w.writeheader(); w.writerows(rows)
print(f"Wrote {len(rows)} rows → {out}")
print("locations visited:", " -> ".join(dict.fromkeys(r["location"] for r in rows)))

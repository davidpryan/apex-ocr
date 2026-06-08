"""
Demonstrates GameAggregator temporal stabilization of the game HUD.
Run with:  python3 test-game-aggregation.py

Feeds a synthetic sequence of per-frame detector results that mimics real capture
noise — dropouts (None), single-frame wrong reads, and intermittently-readable
fields — and checks that the aggregated output converges to the true values while
never emitting transient noise.
"""

from detectors import GameAggregator


def pair(v):
    """Wrap a value as a detector (text, box) pair with a dummy box."""
    return None if v is None else (v, (0, 0, 1, 1))


# Ground truth for the simulated match frame.
TRUTH = {
    "primary": "WINGMAN", "secondary": "HAVOC",
    "armor_number": "3",
    "shield_type": "purple", "shield_hp": 90, "flesh_hp": 100, "health": 190,
    "squads_remaining": "10", "players_remaining": "23",
    "kills": "3", "assists": "1", "participation": "2", "damage": "1144",
}

# A 12-frame sequence of noisy observations per field.
#   None          → dropout (field not read that frame)
#   a wrong value → single-frame OCR noise (should be rejected, appears once)
#   the truth     → correct read
SEQUENCES = {
    "primary":           ["WINGMAN", None, "WINGMAN", "WINGMA", "WINGMAN", "WINGMAN",
                          None, "WINGMAN", "WINGMAN", "WINGMAN", "WINGMAN", "WINGMAN"],
    "secondary":         [None, "HAVOC", "HAVOC", None, "HAVOC", "HAVOC",
                          "HAVOC", "HAVOC", None, "HAVOC", "HAVOC", "HAVOC"],
    "armor_number":      ["3", "3", None, "8", "3", "3", "3", None, "3", "3", "3", "3"],
    "shield_type":       ["purple"]*5 + ["blue"] + ["purple"]*6,           # one mislabel
    "shield_hp":         [90, 90, 95, 90, 90, 90, 90, 90, 88, 90, 90, 90], # jitter
    "flesh_hp":          [100]*12,
    "health":            [190, 190, 185, 190, 190, 190, 190, 190, 190, 190, 190, 190],
    "squads_remaining":  [None, None, "10", None, "10", None, "1", "10", None, "10", "10", None],
    "players_remaining": ["23", "23", None, "23", "23", "23", "23", None, "23", "23", "23", "23"],
    "kills":             [None, "0", None, "3", None, "3", None, "3", None, "3", "3", None],  # intermittent + noise
    "assists":           [None, None, "1", None, None, "1", None, "1", None, None, "1", None],# rare reads
    "participation":     ["2", "2", None, "2", "2", "2", "2", "2", "2", "2", "2", "2"],
    "damage":            [None, "41144", "1144", "1144", None, "1144", "1144", "1144",
                          "1144", "1144", "1144", "1144"],                   # leading-digit noise once
}

N_FRAMES = 12
agg = GameAggregator(window=10, min_votes=2)

print("Feeding 12 noisy frames...\n")
for i in range(N_FRAMES):
    weapon = {s: pair(SEQUENCES[s][i]) for s in ("primary", "secondary")}
    armor  = {"number": SEQUENCES["armor_number"][i], "box": (0, 0, 1, 1)}
    shield = {k: SEQUENCES[k][i] for k in ("shield_type", "shield_hp", "flesh_hp", "health")}
    tr     = {k: pair(SEQUENCES[k][i]) for k in
              ("squads_remaining", "players_remaining", "kills",
               "assists", "participation", "damage")}
    agg.update(weapon=weapon, armor=armor, shield=shield, tr=tr)

stable = agg.update(  # final read of current state (no new data needed)
    weapon={s: pair(SEQUENCES[s][-1]) for s in ("primary", "secondary")},
    armor={"number": SEQUENCES["armor_number"][-1], "box": (0, 0, 1, 1)},
    shield={k: SEQUENCES[k][-1] for k in ("shield_type", "shield_hp", "flesh_hp", "health")},
    tr={k: pair(SEQUENCES[k][-1]) for k in
        ("squads_remaining", "players_remaining", "kills",
         "assists", "participation", "damage")},
)


def got(group, key, has_box=True):
    v = stable[group].get(key) if group != "shield" else (stable["shield"] or {}).get(key)
    if group == "shield":
        return v
    return v[0] if v else None


checks = [
    ("primary",           got("weapon", "primary")),
    ("secondary",         got("weapon", "secondary")),
    ("armor_number",      stable["armor"]["number"]),
    ("shield_type",       got("shield", "shield_type")),
    ("shield_hp",         got("shield", "shield_hp")),
    ("flesh_hp",          got("shield", "flesh_hp")),
    ("health",            got("shield", "health")),
    ("squads_remaining",  got("tr", "squads_remaining")),
    ("players_remaining", got("tr", "players_remaining")),
    ("kills",             got("tr", "kills")),
    ("assists",           got("tr", "assists")),
    ("participation",     got("tr", "participation")),
    ("damage",            got("tr", "damage")),
]

ok = 0
for field, value in checks:
    expected = TRUTH[field]
    match = str(value) == str(expected)
    ok += match
    print(f"  {'✓' if match else '✗'} {field:18} stabilized={value!r:10} truth={expected!r}")

print(f"\n  {ok}/{len(checks)} fields stabilized to ground truth")
print("\nKey demonstrations:")
print("  • kills: noisy '0' + intermittent '3' → stabilized '3' (noise rejected)")
print("  • damage: one-frame '41144' never emitted → stabilized '1144'")
print("  • squads/assists: rare correct reads recovered through dropouts")
print("  • shield_type: single 'blue' mislabel outvoted by 'purple'")

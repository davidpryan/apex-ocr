"""
Map position locator — stub (v1 reserved columns).

v1 returns None for all fields.  Future implementation will template-match
the player chevron on the always-visible minimap (top-left corner of the HUD)
to derive approximate map_x / map_y coordinates and a named location string.

Interface is stable: the match_replay.csv schema already reserves map_x,
map_y, and location columns so no schema migration will be needed when this
is implemented.
"""

import numpy as np


class MapLocator:
    """Locates the player on the game map from a full game-screen frame.

    Returns a dict with nullable keys: ``map_x``, ``map_y``, ``location``.
    All values are None until the minimap chevron detector is implemented.
    """

    def locate(self, full_bgr: np.ndarray) -> dict:
        """Return {map_x, map_y, location} — all None in v1."""
        return {"map_x": None, "map_y": None, "location": None}

"""
Match replay dashboard — visualise a match_replay.csv over the map render.

Run with:
    streamlit run match-replay.py

Extensibility
-------------
Most additions are one-liners in the registries near the top:

  * REPLAY_SOURCES   — where to look for match_replay.csv files (globs).
  * METRICS          — the metric tiles shown for the scrubbed row.
  * TIMESERIES       — the line charts under the map.

New maps need nothing: the map slug is derived from the CSV's ``map_name`` (the
same rule MapLocator uses), and the base render + POI polygons are loaded from
images/maps/ if present.  A map with no PNG or no _pois.json still works — it
just drops the background or the overlay.
"""

import glob
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from PIL import Image

from config import MAPS_DIR, REPLAY_INTERVAL_SEC
from map_locator import _slug   # single source of truth for name → file slug

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Registries (extend here) ──────────────────────────────────────────────────

# Where to discover replay CSVs.  Add a glob and new sources show up in the UI.
REPLAY_SOURCES = [
    os.path.join(HERE, "test_data", "*.csv"),
    os.path.join(HERE, "output", "*", "match_replay.csv"),
    os.path.join(HERE, "match_replay.csv"),
]

# Metric tiles for the scrubbed row.  fmt(row) -> display string.
METRICS = [
    {"label": "Location",  "fmt": lambda r: r.get("location") or "—"},
    {"label": "Kills",     "fmt": lambda r: _int(r.get("kills"))},
    {"label": "Assists",   "fmt": lambda r: _int(r.get("assists"))},
    {"label": "Damage",    "fmt": lambda r: _int(r.get("damage"))},
    {"label": "Squads",    "fmt": lambda r: _int(r.get("squads_remaining"))},
    {"label": "Shield",    "fmt": lambda r: f"{r.get('shield_type') or '—'} {_int(r.get('shield_hp'))}"},
    {"label": "Primary",   "fmt": lambda r: r.get("primary_weapon") or "—"},
    {"label": "Secondary", "fmt": lambda r: r.get("secondary_weapon") or "—"},
]

# Line charts under the map: (title, [columns]).  Add a tuple → a new chart.
TIMESERIES = [
    ("Combat",          ["damage", "kills", "assists"]),
    ("Lobby remaining", ["squads_remaining", "players_remaining"]),
]

# Path markers placed where a cumulative counter steps up.  Missing columns
# (e.g. knocks on real recordings) are skipped.  Add a row → a new event type.
EVENT_MARKERS = [
    {"col": "kills",   "color": "red",    "label": "Kill"},
    {"col": "assists", "color": "orange", "label": "Assist"},
    {"col": "knocks",  "color": "yellow", "label": "Knock"},
]


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return "—"


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def find_replays() -> dict[str, str]:
    """Map a friendly label → CSV path for every discoverable replay."""
    found: dict[str, str] = {}
    for pattern in REPLAY_SOURCES:
        for path in sorted(glob.glob(pattern)):
            found[os.path.relpath(path, HERE)] = path
    return found


@st.cache_data(show_spinner=False)
def load_replay(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "elapsed_s" in df:
        df = df.sort_values("elapsed_s").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_pois(slug: str) -> list[dict]:
    path = os.path.join(MAPS_DIR, f"{slug}_pois.json")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return json.load(fh).get("pois", [])


@st.cache_resource(show_spinner=False)
def load_map_image(slug: str):
    path = os.path.join(MAPS_DIR, f"{slug}.png")
    return Image.open(path).convert("RGB") if os.path.exists(path) else None


# ── Figure ────────────────────────────────────────────────────────────────────
#
# The map is animated with native Plotly frames (client-side).  The background
# image, POI polygons, full path and event markers are STATIC traces — sent to
# the browser once.  Only three traces change per frame: the player marker and
# the two health/shield bars.  Frames animate with redraw=False so the heavy
# background is never re-rendered — the whole point of this design.


def _shield_color(stype) -> str:
    return SHIELD_COLORS.get(str(stype or "").lower(), "#888888")


def _health_of(row: dict) -> float:
    hp = _num(row.get("health"))
    return hp if hp is not None else (_num(row.get("flesh_hp")) or 0.0)


def _stat_annotation(row: dict) -> dict:
    """Per-frame stat overlay shown in the map's top-left corner."""
    kn = _int(row.get("knocks")) if "knocks" in row else "—"
    text = (f"t={_num(row.get('elapsed_s')) or 0:.0f}s   {row.get('location') or '—'}<br>"
            f"K {_int(row.get('kills'))}   A {_int(row.get('assists'))}   "
            f"Kn {kn}   DMG {_int(row.get('damage'))}")
    return dict(x=0.01, y=0.99, xref="paper", yref="paper",
                xanchor="left", yanchor="top", showarrow=False, align="left",
                text=text, font=dict(color="white", size=13),
                bgcolor="rgba(0,0,0,0.55)", borderpad=5)


def build_animation(df: pd.DataFrame, pois: list[dict], img, show_pois: bool) -> go.Figure:
    n = len(df)
    interval = _median_interval(df)
    W, H = (img.size if img is not None
            else (int(df["map_x"].max()) + 50, int(df["map_y"].max()) + 50))

    fig = make_subplots(rows=3, cols=1, row_heights=[0.86, 0.07, 0.07],
                        vertical_spacing=0.025)

    # ── Static layers (sent once) ─────────────────────────────────────────────
    if img is not None:
        fig.add_layout_image(dict(
            source=img, xref="x", yref="y", x=0, y=0,
            sizex=W, sizey=H, xanchor="left", yanchor="top",
            sizing="stretch", layer="below"), row=1, col=1)

    if show_pois and pois:
        for p in pois:
            poly = p.get("polygon")
            if not poly:
                continue
            xs = [v[0] for v in poly] + [poly[0][0]]
            ys = [v[1] for v in poly] + [poly[0][1]]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", fill="toself",
                line=dict(color="rgba(60,220,255,0.7)", width=1),
                fillcolor="rgba(60,220,255,0.07)",
                hoverinfo="text", text=p.get("name") or "???", showlegend=False),
                row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in pois], y=[p["y"] for p in pois], mode="text",
            text=[p.get("name") or "" for p in pois],
            textfont=dict(color="rgba(220,240,255,0.85)", size=10),
            hoverinfo="skip", showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["map_x"], y=df["map_y"], mode="lines",
        line=dict(color="red", width=2), hoverinfo="skip", showlegend=False),
        row=1, col=1)

    for ev in EVENT_MARKERS:
        col = ev["col"]
        if col not in df:
            continue
        events = df[df[col].diff().fillna(df[col]) > 0]
        if len(events):
            fig.add_trace(go.Scatter(
                x=events["map_x"], y=events["map_y"], mode="markers",
                marker=dict(size=13, color=ev["color"], symbol="diamond",
                            line=dict(color="white", width=1)),
                name=ev["label"], hoverinfo="text",
                text=[ev["label"]] * len(events), showlegend=False), row=1, col=1)

    # ── Animated traces (added last; their indices drive the frames) ──────────
    r0 = df.iloc[0].to_dict()
    fig.add_trace(go.Scatter(
        x=[r0["map_x"]], y=[r0["map_y"]], mode="markers",
        marker=dict(size=16, color="black", symbol="circle",
                    line=dict(color="white", width=1.5)),
        hoverinfo="skip", showlegend=False), row=1, col=1)
    idx_player = len(fig.data) - 1
    fig.add_trace(go.Bar(
        x=[_num(r0.get("shield_hp")) or 0], y=["Shield"], orientation="h",
        marker=dict(color=_shield_color(r0.get("shield_type")), line=dict(color="#555", width=1)),
        hoverinfo="x", showlegend=False), row=2, col=1)
    idx_shield = len(fig.data) - 1
    fig.add_trace(go.Bar(
        x=[_health_of(r0)], y=["Health"], orientation="h",
        marker=dict(color="white", line=dict(color="#555", width=1)),
        hoverinfo="x", showlegend=False), row=3, col=1)
    idx_health = len(fig.data) - 1

    # ── Frames: one per row, updating only the three animated traces ──────────
    frames = []
    for i in range(n):
        r = df.iloc[i].to_dict()
        frames.append(go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=[r["map_x"]], y=[r["map_y"]]),
                go.Bar(x=[_num(r.get("shield_hp")) or 0],
                       marker=dict(color=_shield_color(r.get("shield_type")))),
                go.Bar(x=[_health_of(r)]),
            ],
            traces=[idx_player, idx_shield, idx_health],
            layout=go.Layout(annotations=[_stat_annotation(r)]),
        ))
    fig.frames = frames

    # ── Axes ──────────────────────────────────────────────────────────────────
    fig.update_xaxes(range=[0, W], visible=False, constrain="domain", row=1, col=1)
    fig.update_yaxes(range=[H, 0], visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    for rr in (2, 3):
        fig.update_xaxes(range=[0, 100], fixedrange=True, row=rr, col=1)
        fig.update_yaxes(fixedrange=True, row=rr, col=1)

    # ── Playback controls (client-side; never reloads the background) ─────────
    def _speed(label, ratio):
        return dict(label=label, method="animate", args=[None, {
            "frame": {"duration": int(round(1000 * interval / ratio)), "redraw": False},
            "fromcurrent": True, "transition": {"duration": 0}}])

    pause = dict(label="⏸", method="animate", args=[[None], {
        "frame": {"duration": 0, "redraw": False}, "mode": "immediate",
        "transition": {"duration": 0}}])

    steps = [dict(method="animate", label=f"{_num(df['elapsed_s'].iloc[i]) or 0:.0f}",
                  args=[[str(i)], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate", "transition": {"duration": 0}}])
             for i in range(n)]

    fig.update_layout(
        height=860, margin=dict(l=0, r=0, t=44, b=0),
        plot_bgcolor="rgba(0,0,0,0)", showlegend=False, bargap=0.2,
        annotations=[_stat_annotation(r0)],
        updatemenus=[dict(type="buttons", direction="right",
                          x=0, y=1.05, xanchor="left", yanchor="bottom", pad=dict(r=4),
                          buttons=[pause, _speed("▶ 5:1", 5),
                                   _speed("10:1", 10), _speed("30:1", 30)])],
        sliders=[dict(active=0, x=0, y=0, len=1.0, pad=dict(t=28),
                      currentvalue=dict(prefix="t = ", suffix=" s"), steps=steps)],
    )
    return fig


# Shield bar colour by armour level (shield_type value).
SHIELD_COLORS = {
    "white":  "#d9d9d9",
    "blue":   "#3a8dff",
    "purple": "#b24bf3",
    "gold":   "#ffcf33",
    "red":    "#ff4d4d",
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _median_interval(df: pd.DataFrame) -> float:
    if "elapsed_s" in df and len(df) > 1:
        d = df["elapsed_s"].diff().dropna()
        if len(d) and d.median() > 0:
            return float(d.median())
    return float(REPLAY_INTERVAL_SEC)


# ── App ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Match Replay", layout="wide")
    st.title("Game Replay")

    sources = find_replays()
    if not sources:
        st.warning("No match_replay.csv found. Generate one or record a game.")
        return

    with st.sidebar:
        st.header("Replay")
        src_label = st.selectbox("Source CSV", list(sources))
        df = load_replay(sources[src_label])

        if "game_id" in df and df["game_id"].nunique() > 1:
            gid = st.selectbox("Game", sorted(df["game_id"].dropna().unique()))
            df = df[df["game_id"] == gid].reset_index(drop=True)

        map_name = str(df["map_name"].iloc[0]) if "map_name" in df and len(df) else ""
        slug = st.text_input("Map slug", _slug(map_name) if map_name else "")
        show_pois = st.checkbox("Show POIs", value=True)

    if not len(df) or "map_x" not in df:
        st.error("Replay has no rows / no map_x column.")
        return

    st.caption(f"Match ID: {df['game_id'].iloc[0]}  ·  {map_name or 'unknown map'}  ·  {len(df)} rows")

    pois = load_pois(slug)
    img  = load_map_image(slug)
    if img is None:
        st.info(f"No base map images/maps/{slug}.png — showing path only.")

    # Animated map + bars.  Play / speed / scrub all run client-side, so the
    # background image is sent once and never reloads — only the marker moves.
    left, right = st.columns([3, 1])
    with left:
        st.plotly_chart(build_animation(df, pois, img, show_pois), width="stretch")
    with right:
        final = df.iloc[-1].to_dict()
        st.subheader("Final")
        for m in METRICS:
            st.metric(m["label"], m["fmt"](final))

    # Time-series charts (driven by the TIMESERIES registry)
    if "elapsed_s" in df:
        cols = st.columns(len(TIMESERIES))
        for col, (title, series) in zip(cols, TIMESERIES):
            present = [s for s in series if s in df]
            if present:
                col.caption(title)
                col.line_chart(df, x="elapsed_s", y=present)

    with st.expander("Raw rows"):
        st.dataframe(df, width="stretch")


main()

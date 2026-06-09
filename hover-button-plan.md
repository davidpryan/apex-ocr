# Hover Overlay + Record Button — Implementation Plan

A small always-on-top window that floats over the game, showing **FPS** and
**mode** (MENU / RANKED_LOADING / GAME / EXPERIENCE) with a **Record** toggle.
Cross-platform: macOS + Windows.

## Goal

Replace the full-screen `cv2.imshow` preview window in `detect-weapons.py` with a
compact floating widget:

```
┌──────────────────────────────┐
│ ● REC      GAME      58 fps   │
│ game_id: 20260608-143022-x9k2 │
└──────────────────────────────┘
```

- **Record button** — start/stop writing CSV rows. Detection/classification keeps
  running while stopped so the mode/FPS display stays live; you arm recording just
  before you play.
- **Mode** — live `ScreenType`, colour-coded.
- **FPS** — rolling average of the capture loop.

## Library choice: PySide6

| Option | Verdict |
|---|---|
| **PySide6** (Qt for Python) | ✅ Chosen. Clean always-on-top on both OSes, native frameless windows, LGPL (commercial-friendly), good threading story. |
| PyQt6 | Works, but GPL/commercial license. |
| tkinter | Always-on-top is flaky on macOS; weak styling. Avoid. |
| Dear PyGui / imgui | Extra GPU dependency, overkill. |

Add `PySide6` to the environment (`pip install PySide6`). No other new deps —
`mss`, `cv2`, `easyocr` unchanged.

## Architecture

Today `main()` does **everything on one thread**: capture → classify → branch →
CSV write → `cv2.imshow`. The OCR runs in a worker thread already.

Split into two clean halves:

```
┌─────────────────────────┐         ┌──────────────────────────┐
│  Qt main thread          │  poll   │  DetectorEngine thread    │
│  OverlayWindow           │◀────────│  (the old main() loop,    │
│  - QLabel mode/fps        │ shared  │   minus all cv2 GUI calls)│
│  - QPushButton Record     │ state   │  + existing ocr_worker    │
│  - QTimer (200 ms)        │────────▶│    thread (unchanged)     │
└─────────────────────────┘ recording└──────────────────────────┘
                                              │ writes (gated)
                                              ▼
                                      match_history.csv
                                      match_replay.csv
```

- **Qt owns the main thread** (required on macOS — UI must be on the main thread).
- **DetectorEngine** runs the capture/detection loop on a background thread and
  publishes a small thread-safe stats snapshot.
- Communication is one shared `EngineState` object guarded by a `threading.Lock`
  (engine writes, GUI reads) plus a `threading.Event` for the record toggle.

## Files

**New:**
- `overlay.py` — `OverlayWindow` (PySide6 widget) + `EngineState` dataclass.
- `engine.py` — `DetectorEngine` class wrapping the capture/detection loop.

**Changed:**
- `detect-weapons.py` — `main()` becomes: parse args → build detectors → start
  `DetectorEngine` thread → launch Qt app with `OverlayWindow`. The `configure`
  and `debug` cv2 pre-flight stay as-is (they run before the overlay opens).

`detectors.py`, `session.py`, `sinks.py`, `roi_manager.py` etc. are untouched.

## EngineState (shared snapshot)

```python
# overlay.py
from dataclasses import dataclass, field
import threading

@dataclass
class EngineState:
    lock:    threading.Lock = field(default_factory=threading.Lock)
    recording: threading.Event = field(default_factory=threading.Event)
    stop:      threading.Event = field(default_factory=threading.Event)
    # published by engine, read by GUI (hold lock):
    fps:       float = 0.0
    mode:      str   = "—"          # ScreenType.name
    game_id:   str | None = None
    rows_written: int = 0

    def snapshot(self) -> dict:
        with self.lock:
            return {"fps": self.fps, "mode": self.mode,
                    "game_id": self.game_id, "rows": self.rows_written,
                    "recording": self.recording.is_set()}
```

## DetectorEngine

Lift the body of the current `with mss.MSS() as sct:` loop into
`DetectorEngine.run()`. Changes:

1. **Remove all `cv2` GUI** — no `namedWindow`, `imshow`, `waitKey`,
   `setMouseCallback`, no `display`/`frozen`/`draw_all` for the preview, no pause
   click handler. (The `draw_all`/preview rendering is dropped; the overlay is the
   new UI. Keep `draw_all` import only if you still want an optional debug window.)
2. **Loop exit** — replace `if k == ord("q")` with `while not state.stop.is_set()`.
3. **Gate CSV writes on `state.recording`** — the cleanest seam is to wrap the two
   sinks so writes are no-ops when not recording:

   ```python
   class Gate:
       def __init__(self, sink, state): self._s, self._st = sink, state
       def write(self, *a, **k):
           if self._st.recording.is_set():
               self._st.bump_rows()
               return self._s.write(*a, **k)
   exp_writer    = Gate(ExperienceWriter(), state)
   replay_writer = Gate(MatchReplayWriter(), state)
   ```

   The session state machine keeps advancing regardless, so arming Record
   mid-match works immediately. (v1 nuance — see Open Questions.)
4. **Publish stats each frame**:
   ```python
   with state.lock:
       state.fps = fps
       state.mode = screen_type.name
       state.game_id = session.game_id
   ```
5. **Proper rolling FPS** (current calc resets to ~0 each second). Use a deque of
   recent frame timestamps:
   ```python
   from collections import deque
   ts = deque(maxlen=30); ts.append(time.monotonic())
   # each frame:
   ts.append(time.monotonic())
   fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
   ```

Constructor takes the already-built detectors + `EngineState`; `run()` is the
thread target. The existing `ocr_worker` thread is started from inside `run()`
exactly as today.

## OverlayWindow (PySide6)

```python
# overlay.py
from PySide6 import QtCore, QtWidgets, QtGui

_MODE_COLORS = {
    "GAME": "#5dff8f", "EXPERIENCE": "#9db4ff",
    "RANKED_LOADING": "#a0ffa0", "UNKNOWN": "#9a9a9a", "—": "#9a9a9a",
}

class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool                      # no taskbar/dock entry
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.rec   = QtWidgets.QPushButton("● REC")
        self.mode  = QtWidgets.QLabel("—")
        self.fps   = QtWidgets.QLabel("-- fps")
        self.gid   = QtWidgets.QLabel("game_id: —")
        self.rec.setCheckable(True)
        self.rec.toggled.connect(self._toggle_record)

        row = QtWidgets.QHBoxLayout()
        for w in (self.rec, self.mode, self.fps): row.addWidget(w)
        col = QtWidgets.QVBoxLayout(self)
        col.addLayout(row); col.addWidget(self.gid)
        self.setStyleSheet("…rounded dark pill, monospace…")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(200)                     # 5 Hz UI refresh

    def _toggle_record(self, on):
        (self.state.recording.set if on else self.state.recording.clear)()
        self.rec.setText("■ STOP" if on else "● REC")

    def _refresh(self):
        s = self.state.snapshot()
        self.mode.setText(s["mode"])
        self.mode.setStyleSheet(f"color:{_MODE_COLORS.get(s['mode'], '#ccc')}")
        self.fps.setText(f"{s['fps']:.0f} fps")
        self.gid.setText(f"game_id: {s['game_id'] or '—'}")

    # frameless → implement click-drag to move
    def mousePressEvent(self, e):   self._drag = e.globalPosition().toPoint()
    def mouseMoveEvent(self, e):
        if self._drag:
            d = e.globalPosition().toPoint() - self._drag
            self.move(self.pos() + d); self._drag = e.globalPosition().toPoint()
    def mouseReleaseEvent(self, e): self._drag = None

    def closeEvent(self, e):
        self.state.stop.set(); super().closeEvent(e)
```

Frameless + translucent gives a clean floating "pill". Drag-to-move since there's
no title bar. Closing the window signals the engine to stop.

## detect-weapons.py wiring

```python
def main():
    # …parse args, build roi_mgr, reader, all detectors (unchanged)…
    # …optional configure / debug cv2 pre-flight (unchanged)…

    state  = EngineState()
    engine = DetectorEngine(state, reader, detectors..., replay_interval)
    t = threading.Thread(target=engine.run, daemon=True); t.start()

    app = QtWidgets.QApplication(sys.argv)
    win = OverlayWindow(state); win.show()
    app.exec()                      # blocks on main thread until window closed
    state.stop.set(); t.join(timeout=2)
```

## Cross-platform specifics

**macOS**
- **Screen Recording permission** — `mss` capture returns black frames until the
  terminal/app is granted *System Settings → Privacy & Security → Screen
  Recording*. Detect all-black frames at startup and show a hint. (Already a
  latent requirement today.)
- **Floating above the game** — a window can only float over a game running in
  **borderless/windowed** mode. A true exclusive-fullscreen app draws over
  everything. Document "set the game to Borderless Window." `Qt.Tool` keeps it off
  the Dock.
- UI must be on the main thread (satisfied — Qt owns main, engine is the worker).

**Windows**
- `WindowStaysOnTopHint` maps to `WS_EX_TOPMOST` — works over borderless games.
  Same exclusive-fullscreen caveat.
- No special capture permission. `mss` works out of the box.
- For an entry that never steals focus, optionally add the
  `WindowDoesNotAcceptFocus` flag.

**Both**
- DPI scaling: rely on Qt's automatic high-DPI handling (default in Qt6); use
  point-sized fonts, not pixel sizes.
- Multi-monitor: overlay position is in global desktop coords; clamp the initial
  position to the primary screen's geometry.

## Record semantics (v1)

- Record OFF → engine runs, classifies, publishes FPS/mode; **no CSV rows
  written**.
- Record ON → `exp_writer` / `replay_writer` writes flow through.
- The `GameSession` state machine runs continuously either way, so the game_id and
  EXPERIENCE-capture timing are already correct when you arm Record mid-match.

## Phases / effort

1. **Engine extraction** — move loop into `DetectorEngine`, strip cv2 GUI, add
   `EngineState` publishing + rolling FPS. (~1 h) — verify against the 3 reference
   screens that mode classification still matches.
2. **Overlay window** — PySide6 pill, always-on-top, drag-move, 5 Hz refresh.
   (~1 h)
3. **Record gating** — `Gate` wrapper on the two sinks, button wiring. (~30 min)
4. **Cross-platform polish** — black-frame/permission hint, fullscreen-mode note,
   DPI/multi-monitor clamp, packaging notes. (~1 h)

Total **~3.5 h**.

## Open questions

1. **Record = gate writes only, or also reset the session?** v1 gates writes.
   Alternative: starting Record forces a fresh `game_id`. Gating is simpler and
   keeps mid-match arming correct.
2. **Keep an optional full debug preview?** Could add a "Debug" button that opens
   the old `draw_all` cv2 window on demand. Out of scope for v1.
3. **Packaging** — ship as a `pyinstaller` app per-OS, or keep `python
   detect-weapons.py`? Affects where the macOS permission prompt attaches.

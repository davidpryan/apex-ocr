"""
PySide6 always-on-top overlay for the Apex Legends detector.

Displays mode, FPS, and a Record toggle.  Drag the window to reposition it.
"""

import time

from PySide6 import QtCore, QtWidgets, QtGui

from engine import EngineState

_MODE_COLORS = {
    "GAME":           "#4dff91",
    "EXPERIENCE":     "#7ab4ff",
    "RANKED_LOADING": "#a0ffa0",
    "READY":          "#ffdd55",
    "UNKNOWN":        "#888888",
    "—":              "#888888",
}

_STYLE = """
QFrame#overlay {
    background-color: rgba(15, 15, 15, 220);
    border-radius: 10px;
    border: 1px solid rgba(80, 80, 80, 160);
}
QPushButton {
    color: #aaaaaa;
    background-color: rgba(45, 45, 45, 200);
    border: 1px solid rgba(90, 90, 90, 160);
    border-radius: 5px;
    padding: 3px 10px;
    font-family: monospace;
    font-size: 12px;
    min-width: 60px;
}
QPushButton:checked {
    color: #ff5555;
    background-color: rgba(70, 15, 15, 220);
    border-color: rgba(160, 40, 40, 200);
}
QPushButton:hover {
    background-color: rgba(65, 65, 65, 200);
}
QLabel {
    color: #cccccc;
    font-family: monospace;
    font-size: 12px;
    background: transparent;
}
QLabel#mode {
    font-weight: bold;
    font-size: 13px;
    min-width: 120px;
}
QLabel#gid {
    color: #777777;
    font-size: 11px;
}
QLabel#rows {
    color: #666666;
    font-size: 11px;
}
"""


class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, state: EngineState):
        super().__init__()
        self.state     = state
        self._drag_pos = None

        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Apex Detector")

        # ── Widgets ───────────────────────────────────────────────────────────
        self.rec_btn  = QtWidgets.QPushButton("● REC")
        self.mode_lbl = QtWidgets.QLabel("—")
        self.fps_lbl  = QtWidgets.QLabel("-- fps")
        self.gid_lbl  = QtWidgets.QLabel("game_id: —")
        self.rows_lbl = QtWidgets.QLabel("0 rows")

        self.rec_btn.setCheckable(True)
        self.rec_btn.toggled.connect(self._toggle_record)

        self.mode_lbl.setObjectName("mode")
        self.gid_lbl.setObjectName("gid")
        self.rows_lbl.setObjectName("rows")

        # ── Layout ────────────────────────────────────────────────────────────
        container = QtWidgets.QFrame(self)
        container.setObjectName("overlay")

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(10)
        top_row.addWidget(self.rec_btn)
        top_row.addWidget(self.mode_lbl)
        top_row.addStretch()
        top_row.addWidget(self.fps_lbl)

        bot_row = QtWidgets.QHBoxLayout()
        bot_row.setSpacing(10)
        bot_row.addWidget(self.gid_lbl)
        bot_row.addStretch()
        bot_row.addWidget(self.rows_lbl)

        inner = QtWidgets.QVBoxLayout(container)
        inner.setContentsMargins(12, 8, 12, 8)
        inner.setSpacing(5)
        inner.addLayout(top_row)
        inner.addLayout(bot_row)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(container)

        self.setStyleSheet(_STYLE)
        self.setFixedWidth(320)

        # ── Position top-right of primary screen ──────────────────────────────
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - self.width() - 20, 20)

        # ── Refresh timer ─────────────────────────────────────────────────────
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(200)

    # ── Record toggle ─────────────────────────────────────────────────────────

    def _toggle_record(self, on: bool) -> None:
        with self.state.lock:
            if on:
                self.state.recording.set()
                self.state.rec_started_mono = time.monotonic()
            else:
                self.state.recording.clear()
                self.state.rec_started_mono = None
        self.rec_btn.setText("■ 0:00" if on else "● REC")

    # ── 5 Hz refresh from shared state ───────────────────────────────────────

    def _refresh(self) -> None:
        s = self.state.snapshot()

        mode = s["mode"]
        self.mode_lbl.setText(mode)
        color = _MODE_COLORS.get(mode, "#888888")
        self.mode_lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")

        self.fps_lbl.setText(f"{s['fps']:.0f} fps")
        self.gid_lbl.setText(f"game_id: {s['game_id'] or '—'}")
        self.rows_lbl.setText(f"{s['rows']} rows")

        # Update button label with elapsed recording time
        elapsed = s["rec_elapsed"]
        if s["recording"] and elapsed is not None:
            m, sec = divmod(int(elapsed), 60)
            self.rec_btn.setText(f"■ {m}:{sec:02d}")
        elif not s["recording"]:
            self.rec_btn.setText("● REC")

        # Keep checked state in sync if recording was cleared externally
        if self.rec_btn.isChecked() != s["recording"]:
            self.rec_btn.blockSignals(True)
            self.rec_btn.setChecked(s["recording"])
            self.rec_btn.blockSignals(False)

    # ── Frameless drag-to-move ────────────────────────────────────────────────

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if (event.buttons() == QtCore.Qt.MouseButton.LeftButton
                and self._drag_pos is not None):
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._drag_pos = None

    # ── Signal engine to stop when window is closed ───────────────────────────

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._timer.stop()
        self.state.stop.set()
        super().closeEvent(event)

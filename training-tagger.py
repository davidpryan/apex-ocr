"""
training-tagger.py — GUI for turning Apex Legends gameplay mp4s into OCR
training data.

Workflow
--------
1. "Add Videos…" to queue one or more mp4 files (multi-game VODs or
   single-game clips).
2. For a single-game clip: pick the screen type and click "Label Whole File".
   For a multi-game VOD: either click "Auto-Detect" to let the existing
   ScreenClassifier propose segments, or scrub the video and mark segments by
   hand (I = mark in, O = mark out + add).
3. "Export Frames" writes every Nth frame (default 5) of each labeled segment
   to training_data/<label>/<video-stem>_<frame>.png and records it in
   training_data/manifest.csv.

Segment labels are saved to training_data/segments/<video-stem>.json as you
work, and reloaded when the video is opened again.

Keyboard: Space play/pause · ←/→ ±1 frame · Shift+←/→ ±1 s · I mark in ·
O mark out + add segment · 1-4 select label.
"""

import csv
import json
import os
import sys
import threading

import cv2
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QKeySequence, QPainter, QPixmap, QShortcut, QColor
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

HERE          = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR  = os.path.join(HERE, "training_data")
SEGMENTS_DIR  = os.path.join(TRAINING_DIR, "segments")
MANIFEST_CSV  = os.path.join(TRAINING_DIR, "manifest.csv")
MANIFEST_COLS = ["video", "frame", "time_sec", "label", "path"]

EXPORT_EVERY_N_DEFAULT = 5     # every Nth frame inside a segment is exported
AUTODETECT_SAMPLE_STEP = 30    # classify one frame out of every 30 when scanning

# Screen-type labels (mirrors detectors.ScreenType, plus "menu" for lobby /
# loading / spectate screens that the classifier defaults to GAME).
LABELS = ["game", "ranked_loading", "experience", "menu"]

LABEL_COLORS = {
    "game":           QColor("#2e7d32"),
    "ranked_loading": QColor("#1565c0"),
    "experience":     QColor("#6a1b9a"),
    "menu":           QColor("#616161"),
}


# ---------------------------------------------------------------------------
# Frame export / auto-detect (run on a worker thread)
# ---------------------------------------------------------------------------

def export_training_frames(video_path: str, segments: list[dict], every: int,
                           progress, cancel: threading.Event) -> int:
    """Write every Nth frame of each segment to training_data/<label>/.

    Returns the number of frames written.  Re-exporting a video replaces its
    rows in the manifest, so the manifest never holds stale entries.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps  = cap.get(cv2.CAP_PROP_FPS) or 60.0
    stem = os.path.splitext(os.path.basename(video_path))[0]

    target = sum((s["end"] - s["start"]) // every + 1 for s in segments)
    rows: list[dict] = []

    for seg in sorted(segments, key=lambda s: s["start"]):
        label_dir = os.path.join(TRAINING_DIR, seg["label"])
        os.makedirs(label_dir, exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg["start"])
        for f in range(seg["start"], seg["end"] + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if (f - seg["start"]) % every != 0:
                continue
            if cancel.is_set():
                cap.release()
                _update_manifest(video_path, rows)
                return len(rows)
            path = os.path.join(label_dir, f"{stem}_{f:06d}.png")
            cv2.imwrite(path, frame)
            rows.append({
                "video":    os.path.basename(video_path),
                "frame":    f,
                "time_sec": round(f / fps, 3),
                "label":    seg["label"],
                "path":     os.path.relpath(path, TRAINING_DIR),
            })
            progress(len(rows), target)

    cap.release()
    _update_manifest(video_path, rows)
    return len(rows)


def _update_manifest(video_path: str, rows: list[dict]) -> None:
    """Replace this video's rows in manifest.csv with the freshly written set."""
    video = os.path.basename(video_path)
    kept: list[dict] = []
    if os.path.exists(MANIFEST_CSV):
        with open(MANIFEST_CSV, newline="", encoding="utf-8") as fh:
            kept = [r for r in csv.DictReader(fh) if r.get("video") != video]
    os.makedirs(TRAINING_DIR, exist_ok=True)
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
        writer.writeheader()
        writer.writerows(kept + rows)


def autodetect_segments(video_path: str, progress,
                        cancel: threading.Event) -> list[dict]:
    """Scan the video with ScreenClassifier and propose labeled segments.

    Samples one frame per AUTODETECT_SAMPLE_STEP, classifies it, median-smooths
    single-sample blips, and merges consecutive identical labels into segments.
    Imports are deferred: detectors pulls in easyocr/torch, which takes several
    seconds and is only needed for this feature.
    """
    from detectors import ScreenClassifier, ScreenType
    from engine import _hud_present

    classifier = ScreenClassifier()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    samples: list[tuple[int, str]] = []   # (frame_idx, label)
    f = 0
    while f < total and not cancel.is_set():
        if not cap.grab():                 # grab() skips colour conversion
            break
        if f % AUTODETECT_SAMPLE_STEP == 0:
            ok, frame = cap.retrieve()
            if ok:
                stype, _ = classifier.classify(frame)
                if stype is ScreenType.EXPERIENCE:
                    label = "experience"
                elif stype is ScreenType.RANKED_LOADING:
                    label = "ranked_loading"
                else:
                    label = "game" if _hud_present(frame) else "menu"
                samples.append((f, label))
                progress(f, total)
        f += 1
    cap.release()

    if not samples:
        return []

    # Median-smooth: a single sample differing from identical neighbours is noise
    labels = [lbl for _, lbl in samples]
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] != labels[i]:
            labels[i] = labels[i - 1]

    segments: list[dict] = []
    run_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[run_start]:
            start_f = samples[run_start][0]
            end_f   = samples[i][0] - 1 if i < len(labels) else total - 1
            segments.append({"label": labels[run_start],
                             "start": start_f, "end": end_f})
            run_start = i
    return segments


class Worker(QThread):
    """Runs a function with (args..., progress=cb, cancel=Event) off the GUI thread."""

    progress = Signal(int, int)
    done     = Signal(object)
    failed   = Signal(str)

    def __init__(self, fn, *args, parent=None):
        super().__init__(parent)
        self._fn    = fn
        self._args  = args
        self.cancel = threading.Event()

    def run(self):
        try:
            result = self._fn(*self._args,
                              progress=self.progress.emit, cancel=self.cancel)
            self.done.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Segment sidecar persistence
# ---------------------------------------------------------------------------

def _sidecar_path(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(SEGMENTS_DIR, f"{stem}.json")


def load_segments(video_path: str) -> list[dict]:
    path = _sidecar_path(video_path)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("segments", [])


def save_segments(video_path: str, segments: list[dict], fps: float,
                  total_frames: int) -> None:
    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    with open(_sidecar_path(video_path), "w", encoding="utf-8") as fh:
        json.dump({
            "video":        video_path,
            "fps":          fps,
            "total_frames": total_frames,
            "segments":     segments,
        }, fh, indent=2)


# ---------------------------------------------------------------------------
# Timeline widget — coloured segment ranges with a playhead
# ---------------------------------------------------------------------------

class SegmentBar(QWidget):
    seekRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self._total    = 0
        self._segments: list[dict] = []
        self._playhead = 0

    def set_state(self, total: int, segments: list[dict], playhead: int):
        self._total, self._segments, self._playhead = total, segments, playhead
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1e1e1e"))
        if self._total <= 0:
            return
        w, h = self.width(), self.height()
        for seg in self._segments:
            x1 = int(seg["start"] / self._total * w)
            x2 = int((seg["end"] + 1) / self._total * w)
            p.fillRect(x1, 2, max(x2 - x1, 2), h - 4,
                       LABEL_COLORS.get(seg["label"], QColor("#999999")))
        px = int(self._playhead / self._total * w)
        p.fillRect(px, 0, 2, h, QColor("#ffffff"))

    def mousePressEvent(self, event):
        if self._total > 0:
            frac = min(max(event.position().x() / self.width(), 0.0), 1.0)
            self.seekRequested.emit(int(frac * (self._total - 1)))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

def _fmt_time(frame: int, fps: float) -> str:
    sec = frame / fps if fps else 0
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


class TaggerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Apex OCR Training Tagger")
        self.resize(1400, 860)

        self._cap: cv2.VideoCapture | None = None
        self._video_path: str | None = None
        self._fps          = 60.0
        self._total        = 0
        self._cur          = 0
        self._mark_in: int | None = None
        self._segments: list[dict] = []
        self._last_frame   = None          # BGR ndarray of the displayed frame
        self._worker: Worker | None = None

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

        self._build_ui()
        self._bind_keys()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Left panel: video queue
        self.video_list = QListWidget()
        self.video_list.itemSelectionChanged.connect(self._on_video_selected)
        add_btn = QPushButton("Add Videos…")
        add_btn.clicked.connect(self._add_videos)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(add_btn)
        ll.addWidget(self.video_list)

        # Centre: video display + transport
        self.frame_label = QLabel("Add a video to begin")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("background:#101010; color:#888;")
        self.frame_label.setMinimumSize(640, 360)

        self.seg_bar = SegmentBar()
        self.seg_bar.seekRequested.connect(self._seek)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self._seek)
        self.slider.sliderPressed.connect(self._pause)

        self.pos_label = QLabel("frame 0 / 0   00:00.00")

        transport = QHBoxLayout()
        for text, step in (("-10s", "sec-10"), ("-1s", "sec-1"), ("-1f", -1)):
            b = QPushButton(text)
            b.clicked.connect(lambda _, s=step: self._step(s))
            transport.addWidget(b)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        transport.addWidget(self.play_btn)
        for text, step in (("+1f", 1), ("+1s", "sec+1"), ("+10s", "sec+10")):
            b = QPushButton(text)
            b.clicked.connect(lambda _, s=step: self._step(s))
            transport.addWidget(b)
        transport.addStretch()
        transport.addWidget(self.pos_label)

        # Segment controls
        self.label_combo = QComboBox()
        self.label_combo.addItems(LABELS)
        mark_in_btn  = QPushButton("Mark In  [I]")
        mark_in_btn.clicked.connect(self._mark_in_here)
        mark_out_btn = QPushButton("Mark Out + Add  [O]")
        mark_out_btn.clicked.connect(self._mark_out_here)
        whole_btn    = QPushButton("Label Whole File")
        whole_btn.clicked.connect(self._label_whole_file)
        auto_btn     = QPushButton("Auto-Detect")
        auto_btn.clicked.connect(self._auto_detect)
        self.mark_label = QLabel("in: —")

        seg_controls = QHBoxLayout()
        seg_controls.addWidget(QLabel("Type:"))
        seg_controls.addWidget(self.label_combo)
        seg_controls.addWidget(mark_in_btn)
        seg_controls.addWidget(mark_out_btn)
        seg_controls.addWidget(self.mark_label)
        seg_controls.addStretch()
        seg_controls.addWidget(whole_btn)
        seg_controls.addWidget(auto_btn)

        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.addWidget(self.frame_label, stretch=1)
        cl.addWidget(self.seg_bar)
        cl.addWidget(self.slider)
        cl.addLayout(transport)
        cl.addLayout(seg_controls)

        # Right panel: segment table + export
        self.seg_table = QTableWidget(0, 5)
        self.seg_table.setHorizontalHeaderLabels(
            ["Type", "Start", "End", "In", "Out"])
        self.seg_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.seg_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.seg_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.seg_table.cellDoubleClicked.connect(
            lambda row, _col: self._seek(self._segments[row]["start"]))

        del_btn = QPushButton("Delete Segment  [Del]")
        del_btn.clicked.connect(self._delete_selected_segment)

        every_row = QHBoxLayout()
        every_row.addWidget(QLabel("Every Nth frame:"))
        self.every_spin = QSpinBox()
        self.every_spin.setRange(1, 600)
        self.every_spin.setValue(EXPORT_EVERY_N_DEFAULT)
        every_row.addWidget(self.every_spin)
        every_row.addStretch()

        self.export_btn = QPushButton("Export Frames")
        self.export_btn.clicked.connect(self._export)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_worker)
        self.cancel_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("Segments"))
        rl.addWidget(self.seg_table, stretch=1)
        rl.addWidget(del_btn)
        rl.addLayout(every_row)
        rl.addWidget(self.export_btn)
        rl.addWidget(self.cancel_btn)
        rl.addWidget(self.progress)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(centre)
        splitter.addWidget(right)
        splitter.setSizes([220, 800, 380])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage(f"Training data → {TRAINING_DIR}")

    def _bind_keys(self):
        binds = {
            Qt.Key_Space:               self._toggle_play,
            Qt.Key_Left:                lambda: self._step(-1),
            Qt.Key_Right:               lambda: self._step(1),
            Qt.SHIFT | Qt.Key_Left:     lambda: self._step("sec-1"),
            Qt.SHIFT | Qt.Key_Right:    lambda: self._step("sec+1"),
            Qt.Key_I:                   self._mark_in_here,
            Qt.Key_O:                   self._mark_out_here,
            Qt.Key_Delete:              self._delete_selected_segment,
        }
        for i, _label in enumerate(LABELS[:9]):
            binds[getattr(Qt, f"Key_{i + 1}")] = (
                lambda idx=i: self.label_combo.setCurrentIndex(idx))
        for key, fn in binds.items():
            QShortcut(QKeySequence(key), self, activated=fn)

    # ------------------------------------------------------------------
    # Video loading / queue
    # ------------------------------------------------------------------

    def _add_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add gameplay videos", HERE,
            "Videos (*.mp4 *.mkv *.mov *.avi)")
        for path in paths:
            existing = [self.video_list.item(i).data(Qt.UserRole)
                        for i in range(self.video_list.count())]
            if path not in existing:
                item = QListWidgetItem(os.path.basename(path))
                item.setData(Qt.UserRole, path)
                item.setToolTip(path)
                self.video_list.addItem(item)
        if paths and self.video_list.count() == len(paths):
            self.video_list.setCurrentRow(0)

    def _on_video_selected(self):
        items = self.video_list.selectedItems()
        if items:
            self._open_video(items[0].data(Qt.UserRole))

    def _open_video(self, path: str):
        self._pause()
        if self._cap is not None:
            self._cap.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.warning(self, "Open failed", f"Could not open:\n{path}")
            return
        self._cap        = cap
        self._video_path = path
        self._fps        = cap.get(cv2.CAP_PROP_FPS) or 60.0
        self._total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cur        = 0
        self._mark_in    = None
        self._segments   = load_segments(path)
        self.mark_label.setText("in: —")
        self.slider.setRange(0, max(self._total - 1, 0))
        self._refresh_segment_views()
        self._seek(0)
        self.statusBar().showMessage(
            f"{os.path.basename(path)} — {self._total} frames @ "
            f"{self._fps:.2f} fps ({_fmt_time(self._total, self._fps)})")

    # ------------------------------------------------------------------
    # Playback / seeking
    # ------------------------------------------------------------------

    def _show_frame(self, frame) -> None:
        self._last_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.frame_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.frame_label.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation))
        self.slider.blockSignals(True)
        self.slider.setValue(self._cur)
        self.slider.blockSignals(False)
        self.pos_label.setText(
            f"frame {self._cur} / {self._total - 1}   "
            f"{_fmt_time(self._cur, self._fps)}")
        self.seg_bar.set_state(self._total, self._segments, self._cur)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_frame is not None:
            self._show_frame(self._last_frame)

    def _seek(self, frame: int):
        if self._cap is None:
            return
        frame = min(max(frame, 0), self._total - 1)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = self._cap.read()
        if ok:
            self._cur = frame
            self._show_frame(img)

    def _step(self, step):
        if self._cap is None:
            return
        self._pause()
        if isinstance(step, str):                  # "sec-10", "sec+1", …
            step = int(round(float(step[3:]) * self._fps))
        if step == 1:
            self._advance_frame()                  # sequential read, no seek
        else:
            self._seek(self._cur + step)

    def _advance_frame(self):
        if self._cap is None:
            return
        ok, img = self._cap.read()
        if not ok:
            self._pause()
            return
        self._cur = min(self._cur + 1, self._total - 1)
        self._show_frame(img)

    def _toggle_play(self):
        if self._cap is None:
            return
        if self._play_timer.isActive():
            self._pause()
        else:
            self._play_timer.start(int(1000 / self._fps))
            self.play_btn.setText("Pause")

    def _pause(self):
        self._play_timer.stop()
        self.play_btn.setText("Play")

    # ------------------------------------------------------------------
    # Segment editing
    # ------------------------------------------------------------------

    def _mark_in_here(self):
        if self._cap is None:
            return
        self._mark_in = self._cur
        self.mark_label.setText(
            f"in: {self._cur} ({_fmt_time(self._cur, self._fps)})")

    def _mark_out_here(self):
        if self._cap is None:
            return
        if self._mark_in is None:
            self.statusBar().showMessage("Mark an in-point first (I)", 4000)
            return
        start, end = sorted((self._mark_in, self._cur))
        self._segments.append({
            "label": self.label_combo.currentText(),
            "start": start,
            "end":   end,
        })
        self._mark_in = None
        self.mark_label.setText("in: —")
        self._segments_changed()

    def _label_whole_file(self):
        if self._cap is None:
            return
        self._segments = [{
            "label": self.label_combo.currentText(),
            "start": 0,
            "end":   self._total - 1,
        }]
        self._segments_changed()

    def _delete_selected_segment(self):
        row = self.seg_table.currentRow()
        if 0 <= row < len(self._segments):
            del self._segments[row]
            self._segments_changed()

    def _segments_changed(self):
        self._segments.sort(key=lambda s: s["start"])
        save_segments(self._video_path, self._segments, self._fps, self._total)
        self._refresh_segment_views()

    def _refresh_segment_views(self):
        self.seg_table.setRowCount(len(self._segments))
        for row, seg in enumerate(self._segments):
            cells = (
                seg["label"],
                str(seg["start"]),
                str(seg["end"]),
                _fmt_time(seg["start"], self._fps),
                _fmt_time(seg["end"], self._fps),
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setForeground(
                        LABEL_COLORS.get(seg["label"], QColor("#999999")))
                self.seg_table.setItem(row, col, item)
        self.seg_bar.set_state(self._total, self._segments, self._cur)

    # ------------------------------------------------------------------
    # Workers: auto-detect + export
    # ------------------------------------------------------------------

    def _start_worker(self, fn, *args, on_done):
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("A task is already running", 4000)
            return
        self._pause()
        self._worker = Worker(fn, *args, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(on_done)
        self._worker.failed.connect(self._on_worker_failed)
        self.export_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self._worker.start()

    def _cancel_worker(self):
        if self._worker is not None:
            self._worker.cancel.set()

    def _on_progress(self, done: int, total: int):
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)

    def _worker_finished(self):
        self.export_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_worker_failed(self, msg: str):
        self._worker_finished()
        QMessageBox.warning(self, "Task failed", msg)

    def _auto_detect(self):
        if self._video_path is None:
            return
        if self._segments and QMessageBox.question(
                self, "Replace segments?",
                "Auto-detect will replace the current segments. Continue?",
        ) != QMessageBox.Yes:
            return
        self.statusBar().showMessage(
            "Auto-detecting… (first run loads easyocr, give it a moment)")
        self._start_worker(autodetect_segments, self._video_path,
                           on_done=self._on_autodetect_done)

    def _on_autodetect_done(self, segments: list[dict]):
        self._worker_finished()
        self._segments = segments
        self._segments_changed()
        self.statusBar().showMessage(
            f"Auto-detect proposed {len(segments)} segment(s) — review and "
            "adjust before exporting", 8000)

    def _export(self):
        if self._video_path is None:
            return
        if not self._segments:
            self.statusBar().showMessage("No segments to export", 4000)
            return
        self._start_worker(export_training_frames, self._video_path,
                           list(self._segments), self.every_spin.value(),
                           on_done=self._on_export_done)

    def _on_export_done(self, written: int):
        self._worker_finished()
        self.statusBar().showMessage(
            f"Exported {written} frames → {TRAINING_DIR}", 10000)

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._pause()
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel.set()
            self._worker.wait(5000)
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = TaggerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

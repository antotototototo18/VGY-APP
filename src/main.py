# src/main.py
from __future__ import annotations
from PySide6.QtGui import QTextOption

import sys
import os
import math
import time
from typing import Callable

# ==========================================================
# CLEAN ENV LOADING (NO HARDCODED PATHS)
# ==========================================================
from pathlib import Path
from dotenv import load_dotenv

def resource_path(*relative_parts):
    if hasattr(sys, "_MEIPASS"):
        return str(Path(sys._MEIPASS, *relative_parts))
    return str(Path(__file__).resolve().parent.parent.joinpath(*relative_parts))

def _load_project_env():
    """
    Loads .env cleanly without hardcoded paths.
    Priority:
        1) <repo>/.env
        2) <repo>/src/.env
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / ".env",
        here.parent / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p)
            return p

    load_dotenv()  # fallback
    return None

ENV_USED = _load_project_env()

# ==========================================================
# QT / APP IMPORTS
# ==========================================================

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QPixmap,
    QTextCursor,
    QFontDatabase,
    QFont,
    QPainterPath,
    QLinearGradient,
    QRadialGradient,
    QBrush,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
    QSpinBox,
    QMessageBox,
    QComboBox,
    QPlainTextEdit,
    QGridLayout,
    QListView,
)

# ==========================================================
# PROJECT PATHS
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOGO_PATH = resource_path("assets", "vignity_logo.png")
# QSS url() wants forward slashes
CHEVRON_DOWN = Path(resource_path("assets", "icons", "chevron-down.svg")).as_posix()
CHEVRON_UP = Path(resource_path("assets", "icons", "chevron-up.svg")).as_posix()
FIELD_LABEL_WIDTH = 150
FIELD_WIDTH = 180

# ==========================================================
# INTERNAL MODULES
# ==========================================================
from src.spotify_tools import (
    learn_from_liked_songs,
    create_playlists_by_genre,
    create_or_update_playlist,
    get_user_spotify_client,
    _fetch_liked_tracks,
)
from src.curator_tools import ensure_memory_file, load_memory, apply_memory_bias
from src.library_organizer import create_curated_playlists_by_artist
from src.gpt_curator import gpt_cluster_and_create_playlists
from src.spotify_settings_page import SpotifySettingsPage


# ==========================================================
# HUD BACKGROUND
# ==========================================================
class HudBackground(QWidget):
    """
    Subtle grid background matching the Cosmos aesthetic.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(5, 5, 5))

        pen_minor = QPen(QColor(20, 20, 20))
        pen_minor.setWidth(1)
        pen_major = QPen(QColor(30, 30, 30))
        pen_major.setWidth(1)

        w, h = self.width(), self.height()
        grid = 32
        major_every = 4

        for x in range(0, w, grid):
            painter.setPen(pen_major if (x // grid) % major_every == 0 else pen_minor)
            painter.drawLine(x, 0, x, h)

        for y in range(0, h, grid):
            painter.setPen(pen_major if (y // grid) % major_every == 0 else pen_minor)
            painter.drawLine(0, y, 0 + w, y)


# ==========================================================
# LOADER OVERLAY
# ==========================================================
class LoaderOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.hide()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(16)

    def _animate(self):
        self._angle = (self._angle + 6) % 360
        if self.isVisible():
            self.update()

    def show_loader(self):
        if self.parent():
            self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()

    def hide_loader(self):
        self.hide()

    def paintEvent(self, event):
        if not self.isVisible():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor(0, 0, 0, 160))

        cx = self.width() // 2
        cy = self.height() // 2
        radius = 40

        for i in range(12):
            ang = math.radians(self._angle + i * 30)
            alpha = int(255 * (i + 1) / 12)
            color = QColor(240, 240, 240, alpha)

            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            x = cx + radius * math.cos(ang)
            y = cy + radius * math.sin(ang)
            painter.drawEllipse(int(x), int(y), 10, 10)


# ==========================================================
# COSMOS CONSOLE
# ==========================================================
class CosmosConsole(QPlainTextEdit):
    """
    Futuristic animated console with typing + waveform effects.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setReadOnly(True)
        self.setFrameStyle(0)
        self.setWordWrapMode(QTextOption.NoWrap)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # Styling
        self.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #030303;
                border: 1px solid #2b2e32;
                color: #c9d1d9;
                padding: 10px 16px;
                selection-background-color: #135d4f;
                font-size: 11px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 6px;
                margin: 4px 2px;
            }
            QScrollBar::handle:vertical {
                background: rgba(120,255,200,60);
                min-height: 24px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
            """
        )
        self.setFont(QFont("IBM Plex Mono", 10))

        # Typing animation
        self._buffer = ""
        self._queue = []
        self._current = ""
        self._index = 0

        self._typing_timer = QTimer(self)
        self._typing_timer.setInterval(26)
        self._typing_timer.timeout.connect(self._type_step)

        # Wave animation (time-based so speed doesn't depend on timer jitter)
        self._t0 = time.monotonic()
        self._t = 0.0
        self._active = False
        self._energy = 0.0  # eases toward 1 while a job runs, back to 0 after
        self._wave_timer = QTimer(self)
        self._wave_timer.setInterval(33)
        self._wave_timer.timeout.connect(self._advance_wave)
        self._wave_timer.start()

    def add_message(self, text: str, level="info"):
        icons = {"info": "◉", "running": "▶", "success": "✔", "error": "✖"}
        prefix = icons.get(level, "◉")
        self._queue.append(f"{prefix} {text}")

        if not self._current and not self._typing_timer.isActive():
            self._start_next()

    def _start_next(self):
        if not self._queue:
            return
        self._current = self._queue.pop(0)
        self._index = 0
        if self._buffer:
            self._buffer += "\n"
        self._typing_timer.start()

    def _type_step(self):
        if self._index < len(self._current):
            self._buffer += self._current[self._index]
            self.setPlainText(self._buffer)
            self.moveCursor(QTextCursor.End)
            self._index += 1
        else:
            self._typing_timer.stop()
            self._current = ""
            self._start_next()

    def set_active(self, active: bool):
        self._active = active

    def _advance_wave(self):
        self._t = time.monotonic() - self._t0
        target = 1.0 if self._active else 0.0
        self._energy += (target - self._energy) * 0.06
        self.viewport().update()

    # -- ECG model ----------------------------------------------------------
    _IDLE_BPM, _ACTIVE_BPM = 72, 128
    _BEATS_ON_SCREEN = 4
    _AXIS_MAX = 10

    @staticmethod
    def _ecg(phase: float) -> float:
        """One PQRST complex in axis units (-10..+10) for a beat phase in [0, 1)."""
        g = lambda c, w, a: a * math.exp(-(((phase - c) / w) ** 2))
        return (
            g(0.14, 0.025, 1.6)     # P
            + g(0.24, 0.011, -1.8)  # Q
            + g(0.265, 0.012, 10.0)  # R
            + g(0.292, 0.012, -10.0)  # S
            + g(0.48, 0.05, 2.8)     # T
        )

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self.viewport())
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.viewport().width(), self.viewport().height()
        if w < 80 or h < 80:
            return

        axis_w = 30
        left, right = axis_w + 10, w - 16
        plot_w = right - left
        mid = h * 0.56
        scale = (h * 0.26) / self._AXIS_MAX
        to_y = lambda v: mid - v * scale
        teal = lambda a: QColor(0, 255, 180, max(0, min(255, int(a))))

        # Axis: quiet labels + short ticks, dashed baseline
        p.setFont(QFont("IBM Plex Mono", 8))
        for v in (10, 5, 0, -5, -10):
            y = to_y(v)
            p.setPen(teal(55 if v in (10, 0, -10) else 30))
            p.drawText(QRectF(0, y - 7, axis_w, 14), Qt.AlignRight | Qt.AlignVCenter, f"{v:+d}" if v else "0")
            p.setPen(QPen(teal(45), 1))
            p.drawLine(QPointF(left - 5, y), QPointF(left - 1, y))
        base = QPen(teal(22), 1, Qt.CustomDashLine)
        base.setDashPattern([2, 6])
        p.setPen(base)
        p.drawLine(QPointF(left, mid), QPointF(right, mid))

        # Sweep geometry
        e = self._energy
        bpm = self._IDLE_BPM + (self._ACTIVE_BPM - self._IDLE_BPM) * e
        beat_s = 60.0 / bpm
        speed = plot_w / (self._BEATS_ON_SCREEN * beat_s)
        t = self._t
        pen_x = (t * speed) % plot_w
        gap = 26

        def y_at(x: float) -> float:
            age = ((pen_x - x) % plot_w) / speed
            return to_y(self._ecg(((t - age) / beat_s) % 1.0))

        def stroke_parts():
            # (x_from, x_to, age_at_from, age_at_to) — age is linear in x, so a
            # horizontal gradient reproduces the phosphor fade exactly
            sweep_s = plot_w / speed
            parts = [(0.0, pen_x, pen_x / speed, 0.0)]
            if pen_x + gap < plot_w:
                parts.append((pen_x + gap, plot_w, (plot_w - gap) / speed, pen_x / speed))
            return parts, sweep_s

        parts, sweep_s = stroke_parts()
        for x0, x1, age0, age1 in parts:
            if x1 - x0 < 2:
                continue
            line = QPainterPath()
            xs = [x0 + i for i in range(int(x1 - x0) + 1)] + [x1]
            for i, xi in enumerate(xs):
                pt = QPointF(left + xi, y_at(xi))
                line.moveTo(pt) if i == 0 else line.lineTo(pt)
            fill = QPainterPath(line)
            fill.lineTo(left + x1, mid)
            fill.lineTo(left + x0, mid)
            fill.closeSubpath()

            def grad(alpha: float, hot: bool = False) -> QLinearGradient:
                g = QLinearGradient(left + x0, 0, left + x1, 0)
                for k in range(9):
                    f = k / 8
                    age = age0 + (age1 - age0) * f
                    life = max(0.0, 1.0 - age / sweep_s) ** 1.6
                    c = teal(alpha * life)
                    if hot and age < 0.08:  # white-hot just behind the pen
                        c = QColor(200, 255, 235, int(alpha * life))
                    g.setColorAt(f, c)
                return g

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(grad(18 + 10 * e)))
            p.drawPath(fill)
            p.setBrush(Qt.NoBrush)
            for width, alpha in ((14, 6), (8, 12), (4, 30)):
                pen = QPen(QBrush(grad(alpha + 8 * e)), width)
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                p.setPen(pen)
                p.drawPath(line)
            core = QPen(QBrush(grad(235, hot=True)), 1.6)
            core.setCapStyle(Qt.RoundCap)
            core.setJoinStyle(Qt.RoundJoin)
            p.setPen(core)
            p.drawPath(line)

        # Pen head bloom
        hx, hy = left + pen_x, y_at(pen_x)
        bloom = QRadialGradient(QPointF(hx, hy), 16)
        bloom.setColorAt(0.0, QColor(200, 255, 235, 150))
        bloom.setColorAt(0.25, teal(70))
        bloom.setColorAt(1.0, teal(0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(bloom))
        p.drawEllipse(QPointF(hx, hy), 16, 16)
        p.setBrush(QColor(235, 255, 248, 255))
        p.drawEllipse(QPointF(hx, hy), 2.2, 2.2)

        # BPM readout with a dot that pulses on each R wave
        phase = (t / beat_s) % 1.0
        pulse = math.exp(-(((phase - 0.265) % 1.0) / 0.12))
        ry = to_y(-10) + 22
        p.setBrush(teal(60 + 180 * pulse))
        p.drawEllipse(QPointF(right - 70, ry), 2.5 + 1.5 * pulse, 2.5 + 1.5 * pulse)
        p.setPen(teal(150))
        p.setFont(QFont("IBM Plex Mono", 9))
        p.drawText(QRectF(right - 62, ry - 8, 62, 16), Qt.AlignRight | Qt.AlignVCenter, f"{bpm:.0f} BPM")


# ==========================================================
# MAIN WINDOW
# ==========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_memory_file()
        self.setWindowTitle("Vignity – AI Spotify Curator")
        self.setMinimumSize(1120, 680)

        # PULSE EFFECT
        self._panels_pulsing = False
        self._pulse_on = False
        self._panel_pulse_timer = QTimer(self)
        self._panel_pulse_timer.setInterval(130)
        self._panel_pulse_timer.timeout.connect(self._pulse_panels)

        # Background
        central = HudBackground()
        self.setCentralWidget(central)

        root = QHBoxLayout()
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(24)
        central.setLayout(root)

        # L/R Panels
        self.left_panel = QFrame(objectName="LeftPanel")
        self.right_panel = QFrame(objectName="RightPanel")
        root.addWidget(self.left_panel, 3)
        root.addWidget(self.right_panel, 4)

        # --------------------------------------------------
        # LEFT PANEL
        # --------------------------------------------------
        left = QVBoxLayout()
        left.setContentsMargins(24, 20, 24, 20)
        self.left_panel.setLayout(left)

        # Header
        header = QHBoxLayout()
        logo = QLabel()
        pix = QPixmap(LOGO_PATH)
        if pix and not pix.isNull():
            pix = pix.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo.setPixmap(pix)
        else:
            logo.setText("VIGNITY")
        header.addWidget(logo)

        title_box = QVBoxLayout()
        title = QLabel("VIGNITY", objectName="AppTitle")
        subtitle = QLabel("AI LIBRARY CURATION MODULE", objectName="AppSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()

        settings_btn = QPushButton("SETTINGS")
        settings_btn.setObjectName("SettingsButton")
        settings_btn.clicked.connect(self.open_settings)
        header.addWidget(settings_btn)

        left.addLayout(header)
        left.addSpacing(24)

        # Liked song limit
        self.scope_spin = QSpinBox(objectName="ScopeSpin")
        self.scope_spin.setRange(50, 2000)
        self.scope_spin.setValue(50)
        left.addLayout(self._field_grid([("SCAN LIKED SONGS LIMIT", self.scope_spin)]))
        left.addSpacing(16)

        # Actions
        left.addWidget(QLabel("CURATION ACTIONS", objectName="SectionLabel"))
        left.addSpacing(8)

        self.buttons = []

        self._add_action_button(left, "SCAN LIKED SONGS", self.action_scan_liked_songs)
        self._add_action_button(left, "BUILD GENRE PLAYLISTS", self.action_build_genre_grid)
        self._add_action_button(left, "ORGANIZE BY ARTIST", self.action_organize_by_artist)
        self._add_action_button(left, "VIGNITY PICKS", self.action_generate_vignity_picks)

        # GPT SECTION
        left.addSpacing(16)
        left.addWidget(QLabel("SMART SET: GPT POWERED", objectName="SectionLabel"))
        left.addSpacing(8)

        self.gpt_mode_select = QComboBox(objectName="GptModeSelect")
        # A QListView popup honours the stylesheet; the default one is the native macOS menu
        self.gpt_mode_select.setView(QListView())
        self.gpt_mode_select.addItems(["Balanced", "Mood", "Energy", "Discovery", "Background"])

        self.playlist_count_spin = QSpinBox(objectName="ScopeSpin")
        self.playlist_count_spin.setRange(3, 20)
        self.playlist_count_spin.setValue(7)

        left.addLayout(self._field_grid([
            ("GPT MODE", self.gpt_mode_select),
            ("PLAYLIST COUNT", self.playlist_count_spin),
        ]))
        left.addSpacing(12)

        self._add_action_button(left, "RUN GPT SMART SETS", self.action_gpt_smart_sets)

        left.addStretch()

        # --------------------------------------------------
        # RIGHT PANEL
        # --------------------------------------------------
        right = QVBoxLayout()
        right.setContentsMargins(24, 20, 24, 20)
        self.right_panel.setLayout(right)

        right.addWidget(QLabel("VIGNITY CONSOLE", objectName="SectionLabel"))
        self.log_view = CosmosConsole(self)
        right.addWidget(self.log_view, 5)

        right.addWidget(QLabel("VIGNITY // LIBRARY TELEMETRY", objectName="FooterLabel"))

        # Loader
        self.loader_overlay = LoaderOverlay(self)

        # Spotify init
        try:
            self.spotify_client = get_user_spotify_client()
            self.append_log("Spotify link established.", "success")
        except Exception as e:
            self.append_log(f"Spotify not connected: {e}", "error")

        self.apply_styles()
        self.append_log("SYSTEM READY. Awaiting instructions…")

    # -----------------------------------------------------
    # UTILS
    # -----------------------------------------------------
    def resizeEvent(self, event):
        if self.loader_overlay:
            self.loader_overlay.setGeometry(self.rect())
        super().resizeEvent(event)

    def append_log(self, text, level="info"):
        self.log_view.add_message(text, level)

    def set_loading(self, loading: bool):
        for b in self.buttons:
            b.setEnabled(not loading)
        for w in (self.gpt_mode_select, self.scope_spin, self.playlist_count_spin):
            w.setEnabled(not loading)

        if loading:
            self.loader_overlay.show_loader()
            self.log_view.set_active(True)
            if not self._panels_pulsing:
                self._panels_pulsing = True
                self._panel_pulse_timer.start()
        else:
            self.loader_overlay.hide_loader()
            self.log_view.set_active(False)
            self._panels_pulsing = False
            self._panel_pulse_timer.stop()
            self.apply_styles()

    def _run_with_loader(self, func: Callable):
        def inner():
            self.set_loading(True)
            QApplication.processEvents()
            try:
                func()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                self.append_log(str(e), "error")
            finally:
                self.set_loading(False)
        return inner

    def _add_action_button(self, layout: QVBoxLayout, text: str, handler: Callable):
        btn = QPushButton(text, objectName="ActionButton")
        btn.clicked.connect(self._run_with_loader(handler))
        layout.addWidget(btn)
        self.buttons.append(btn)

    def _field_grid(self, rows) -> QGridLayout:
        """Label/control rows with a shared label column so controls line up."""
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        for i, (text, control) in enumerate(rows):
            label = QLabel(text, objectName="FieldLabel")
            label.setFixedWidth(FIELD_LABEL_WIDTH)
            control.setFixedWidth(FIELD_WIDTH)
            grid.addWidget(label, i, 0, Qt.AlignVCenter)
            grid.addWidget(control, i, 1, Qt.AlignVCenter)
        grid.setColumnStretch(2, 1)
        return grid

    def _pulse_panels(self):
        if not self._panels_pulsing:
            return
        self._pulse_on = not self._pulse_on
        self.apply_styles(pulse=self._pulse_on)

    # -----------------------------------------------------
    # ACTIONS
    # -----------------------------------------------------
    def action_scan_liked_songs(self):
        limit = self.scope_spin.value()
        self.append_log(f"Scanning {limit} liked songs → updating memory…", "running")
        learn_from_liked_songs(limit)
        self.append_log("Memory updated from liked songs.", "success")

    def action_build_genre_grid(self):
        self.append_log("Building genre playlists…", "running")
        create_playlists_by_genre()
        self.append_log("Genre playlists created.", "success")

    def action_organize_by_artist(self):
        self.append_log("Organizing by artist…", "running")
        create_curated_playlists_by_artist()
        self.append_log("Artist-based playlists created.", "success")

    def action_generate_vignity_picks(self):
        self.append_log("Generating Vignity Picks…", "running")

        sp = get_user_spotify_client()
        user_id = sp.current_user()["id"]
        liked = _fetch_liked_tracks(sp, limit=500)
        mem = load_memory()

        scored = []
        for item in liked:
            track = item.get("track") or {}
            if not track:
                continue
            artists = track.get("artists") or []
            row = {
                "artist": artists[0]["name"] if artists else "",
                "name": track.get("name", ""),
                "id": track.get("id"),
            }
            row["_score"] = apply_memory_bias(row, mem)
            scored.append(row)

        scored.sort(key=lambda r: r["_score"], reverse=True)
        picks = scored[:30]

        tracks = [{"id": t["id"]} for t in picks if t.get("id")]

        create_or_update_playlist(sp, user_id, "Vignity Picks", tracks)
        self.append_log(f"Vignity Picks created with {len(tracks)} tracks.", "success")

    def action_gpt_smart_sets(self):
        mode = self.gpt_mode_select.currentText()
        limit = self.scope_spin.value()
        target = self.playlist_count_spin.value()
        self.append_log(f"Running Vignity Smart Sets ({mode}, {target} playlists)…", "running")

        summaries = gpt_cluster_and_create_playlists(
            limit=limit,
            mode=mode,
            target_playlists=target,
            on_progress=self.append_log,
        )

        if summaries:
            for s in summaries:
                self.append_log(s, "info")

        self.append_log("Vignity Smart Sets complete.", "success")

    def open_settings(self):
        dlg = SpotifySettingsPage(self)
        dlg.exec()

    # -----------------------------------------------------
    # STYLING
    # -----------------------------------------------------
    def apply_styles(self, pulse=False):
        border_alpha = 32 if not pulse else 80
        right_alpha = 24 if not pulse else 70

        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: #050505;
            }}
            QWidget#LeftPanel, QWidget#RightPanel {{
                background-color: rgba(5,5,5,230);
                border: 1px solid rgba(255,255,255,{border_alpha});
            }}
            QWidget#RightPanel {{
                border-color: rgba(120,255,200,{right_alpha});
            }}
            QLabel#AppTitle {{
                color: white;
                font-size: 18px;
                letter-spacing: 4px;
            }}
            QLabel#AppSubtitle {{
                color: rgba(255,255,255,0.65);
                font-size: 10px;
            }}
            QLabel#SectionLabel {{
                color: rgba(255,255,255,0.85);
                font-size: 11px;
                letter-spacing: 3px;
            }}
            QLabel#FieldLabel {{
                color: rgba(255,255,255,0.7);
                font-size: 10px;
            }}
            QPushButton#ActionButton {{
                background-color: rgba(10,10,10,230);
                border: 1px solid rgba(255,255,255,40);
                color: white;
                padding: 10px 16px;
                text-align: left;
                font-size: 11px;
            }}
            QPushButton#ActionButton:hover {{
                border-color: rgba(120,255,200,160);
            }}
            QPushButton#SettingsButton {{
                background-color: rgba(255,255,255,8);
                border: 1px solid rgba(255,255,255,35);
                color: white;
                padding: 6px 12px;
                letter-spacing: 2px;
            }}
            QSpinBox#ScopeSpin, QComboBox#GptModeSelect {{
                background-color: rgba(10,10,10,230);
                border: 1px solid rgba(255,255,255,35);
                color: white;
                padding: 5px 10px;
                font-size: 11px;
                min-height: 18px;
            }}
            QSpinBox#ScopeSpin:hover, QComboBox#GptModeSelect:hover,
            QSpinBox#ScopeSpin:focus, QComboBox#GptModeSelect:focus,
            QComboBox#GptModeSelect:on {{
                border-color: rgba(120,255,200,160);
            }}
            QSpinBox#ScopeSpin:disabled, QComboBox#GptModeSelect:disabled {{
                color: rgba(255,255,255,0.35);
            }}
            QSpinBox#ScopeSpin {{
                padding-right: 26px;
            }}
            QComboBox#GptModeSelect {{
                combobox-popup: 0;
            }}
            QSpinBox#ScopeSpin::up-button, QSpinBox#ScopeSpin::down-button {{
                subcontrol-origin: border;
                width: 22px;
                background: transparent;
                border-left: 1px solid rgba(255,255,255,35);
            }}
            QSpinBox#ScopeSpin::up-button {{
                subcontrol-position: top right;
                border-bottom: 1px solid rgba(255,255,255,20);
            }}
            QSpinBox#ScopeSpin::down-button {{
                subcontrol-position: bottom right;
            }}
            QSpinBox#ScopeSpin::up-button:hover, QSpinBox#ScopeSpin::down-button:hover {{
                background: rgba(120,255,200,30);
            }}
            QSpinBox#ScopeSpin::up-arrow {{
                image: url("{CHEVRON_UP}");
                width: 8px;
                height: 5px;
            }}
            QSpinBox#ScopeSpin::down-arrow, QComboBox#GptModeSelect::down-arrow {{
                image: url("{CHEVRON_DOWN}");
                width: 8px;
                height: 5px;
            }}
            QComboBox#GptModeSelect::down-arrow {{
                width: 10px;
                height: 6px;
            }}
            QComboBox#GptModeSelect::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 26px;
                border-left: 1px solid rgba(255,255,255,35);
            }}
            QComboBox#GptModeSelect QAbstractItemView {{
                background-color: #0a0a0a;
                border: 1px solid rgba(120,255,200,120);
                color: white;
                outline: 0;
                padding: 4px 0;
                selection-background-color: rgba(120,255,200,40);
                selection-color: white;
            }}
            QComboBox#GptModeSelect QAbstractItemView::item {{
                padding: 6px 10px;
                min-height: 18px;
            }}
            QComboBox#GptModeSelect QAbstractItemView::item:hover {{
                background-color: rgba(120,255,200,25);
            }}
            """
        )


# ==========================================================
# ENTRY
# ==========================================================
def main():
    app = QApplication(sys.argv)

    # Optional font loading
    for name in ("Inter.ttf", "IBMPlexMono.ttf"):
        if QFontDatabase.addApplicationFont(resource_path("assets", "fonts", name)) == -1:
            print(f"Font loading failed: {name}")

    if "Inter" in QFontDatabase.families():
        app.setFont(QFont("Inter", 11))

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

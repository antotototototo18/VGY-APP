# src/main.py
from __future__ import annotations
from PySide6.QtGui import QTextOption

import sys
import os
import math
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

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QPixmap,
    QTextCursor,
    QFontDatabase,
    QFont,
    QPainterPath,
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
)

# ==========================================================
# PROJECT PATHS
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOGO_PATH = resource_path("assets", "vignity_logo.png")

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

        # Wave animation
        self._phase = 0
        self._wave_timer = QTimer(self)
        self._wave_timer.setInterval(60)
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

    def _advance_wave(self):
        self._phase += 0.25
        self.viewport().update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self.viewport())
        p.setRenderHint(QPainter.Antialiasing)

        color = QColor(0, 255, 180, 28)
        pen = QPen(color, 1.4)
        p.setPen(pen)

        w, h = self.viewport().width(), self.viewport().height()
        mid = h * 0.5

        path = QPainterPath()
        path.moveTo(0, mid)

        for x in range(0, w, 12):
            y = mid + math.sin((x / 30) + self._phase) * (h * 0.16)
            path.lineTo(x, y)

        p.drawPath(path)


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
        left.addWidget(QLabel("SCAN LIKED SONGS LIMIT", objectName="FieldLabel"))
        self.scope_spin = QSpinBox()
        self.scope_spin.setRange(50, 2000)
        self.scope_spin.setValue(50)
        self.scope_spin.setObjectName("ScopeSpin")
        left.addWidget(self.scope_spin)
        left.addSpacing(12)

        # Actions
        left.addWidget(QLabel("CURATION ACTIONS", objectName="SectionLabel"))
        left.addSpacing(8)

        self.buttons = []

        self._add_action_button(left, "SCAN LIKED SONGS", self.action_scan_liked_songs)
        self._add_action_button(left, "BUILD GENRE PLAYLISTS", self.action_build_genre_grid)
        self._add_action_button(left, "ORGANIZE BY ARTIST", self.action_organize_by_artist)
        self._add_action_button(left, "VIGNITY PICKS", self.action_generate_vignity_picks)

        # GPT SECTION
        left.addWidget(QLabel("SMART SET: GPT POWERED", objectName="SectionLabel"))
        left.addSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("GPT MODE", objectName="FieldLabel"))

        self.gpt_mode_select = QComboBox(objectName="GptModeSelect")
        self.gpt_mode_select.addItems(["Balanced", "Mood", "Energy", "Discovery", "Background"])
        row.addWidget(self.gpt_mode_select)
        row.addStretch()
        left.addLayout(row)

        # Playlist count
        left.addSpacing(4)
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("PLAYLIST COUNT", objectName="FieldLabel"))

        self.playlist_count_spin = QSpinBox(objectName="ScopeSpin")
        self.playlist_count_spin.setRange(3, 20)
        self.playlist_count_spin.setValue(7)
        count_row.addWidget(self.playlist_count_spin)
        count_row.addStretch()
        left.addLayout(count_row)

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
        self.gpt_mode_select.setEnabled(not loading)

        if loading:
            self.loader_overlay.show_loader()
            if not self._panels_pulsing:
                self._panels_pulsing = True
                self._panel_pulse_timer.start()
        else:
            self.loader_overlay.hide_loader()
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
            QSpinBox#ScopeSpin {{
                background-color: rgba(10,10,10,220);
                border: 1px solid rgba(255,255,255,35);
                color: white;
                padding: 4px 8px;
            }}
            QComboBox#GptModeSelect {{
                background-color: rgba(10,10,10,230);
                border: 1px solid rgba(255,255,255,35);
                color: white;
                padding: 4px 8px;
                font-size: 10px;
            }}
            """
        )


# ==========================================================
# ENTRY
# ==========================================================
def main():
    app = QApplication(sys.argv)

    # Optional font loading
    try:
        QFontDatabase.addApplicationFont(resource_path("assets", "fonts", "Inter.ttf"))
        QFontDatabase.addApplicationFont(resource_path("assets", "fonts", "IBMPlexMono.ttf"))
    except Exception as e:
        print("Font loading failed:", e)

    app.setFont(QFont("Inter", 11))

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

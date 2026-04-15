# spotify_settings_page.py
from __future__ import annotations
from pathlib import Path
import os

from dotenv import load_dotenv
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
)

# Load Spotify client (so we can test connection)
from src.spotify_tools import get_env_path, load_env, get_user_spotify_client


# ----------------------------------
# UI DIALOG
# ----------------------------------

class SpotifySettingsPage(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Vignity — Spotify Settings")
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("SPOTIFY LINK SETUP")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        # ----------------------------------
        # INPUT FIELDS
        # ----------------------------------

        self.client_id = QLineEdit()
        self.client_secret = QLineEdit()
        self.client_secret.setEchoMode(QLineEdit.Password)
        self.redirect_uri = QLineEdit()

        layout.addWidget(QLabel("SPOTIPY_CLIENT_ID"))
        layout.addWidget(self.client_id)

        layout.addWidget(QLabel("SPOTIPY_CLIENT_SECRET"))
        layout.addWidget(self.client_secret)

        layout.addWidget(QLabel("SPOTIPY_REDIRECT_URI"))
        layout.addWidget(self.redirect_uri)

        # Load existing values
        self.load_existing_env()

        # ----------------------------------
        # BUTTONS
        # ----------------------------------

        btn_box = QHBoxLayout()

        save_btn = QPushButton("SAVE")
        save_btn.clicked.connect(self.save_env)

        test_btn = QPushButton("TEST CONNECTION")
        test_btn.clicked.connect(self.test_connection)

        close_btn = QPushButton("CLOSE")
        close_btn.clicked.connect(self.close)

        btn_box.addWidget(test_btn)
        btn_box.addWidget(save_btn)
        btn_box.addWidget(close_btn)

        layout.addLayout(btn_box)
        self.setLayout(layout)

    # ----------------------------------
    # LOAD EXISTING CREDENTIALS
    # ----------------------------------

    def load_existing_env(self):
        """Reads existing env values and fills the form."""
        env_path = get_env_path()
        load_env()

        self.client_id.setText(os.getenv("SPOTIPY_CLIENT_ID", ""))
        self.client_secret.setText(os.getenv("SPOTIPY_CLIENT_SECRET", ""))
        self.redirect_uri.setText(os.getenv("SPOTIPY_REDIRECT_URI", ""))
        self.openai_key.setText(os.getenv("OPENAI_API_KEY", ""))

    # ----------------------------------
    # SAVE CREDENTIALS
    # ----------------------------------

    def save_env(self):
        """Writes credentials to external .env file."""
        env_path = get_env_path()
        env_path.parent.mkdir(parents=True, exist_ok=True)

        client_id = self.client_id.text().strip()
        client_secret = self.client_secret.text().strip()
        redirect_uri = self.redirect_uri.text().strip()
        openai_key = self.openai_key.text().strip()

        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"SPOTIPY_CLIENT_ID={client_id}\n")
            f.write(f"SPOTIPY_CLIENT_SECRET={client_secret}\n")
            f.write(f"SPOTIPY_REDIRECT_URI={redirect_uri}\n")
            f.write(f"OPENAI_API_KEY={self.openai_key.text().strip()}\n")

        # 🔥 force the current running app to use the new values immediately
        load_dotenv(env_path, override=True)
        os.environ["SPOTIPY_CLIENT_ID"] = client_id
        os.environ["SPOTIPY_CLIENT_SECRET"] = client_secret
        os.environ["SPOTIPY_REDIRECT_URI"] = redirect_uri
        os.environ["OPENAI_API_KEY"] = openai_key

        QMessageBox.information(self, "Saved", f"Credentials saved to:\n{env_path}")

    # ----------------------------------
    # TEST CONNECTION
    # ----------------------------------

    def test_connection(self):
        try:
            sp = get_user_spotify_client()
            user = sp.current_user()
            QMessageBox.information(self, "Success", f"Connected as: {user['display_name']}")
        except Exception as e:
            QMessageBox.critical(self, "Connection Failed", str(e))

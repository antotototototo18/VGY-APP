import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ===== Paths =====
MEMORY_PATH = Path(__file__).resolve().parent.parent / "memory" / "user_prefs.json"


# ===== Spotify Song Search =====
def _get_spotify_client_for_search() -> spotipy.Spotify:
    """
    Spotify client for search.
    Uses OAuth with a small set of scopes and cached credentials.
    """
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
            scope="user-library-read",
            cache_path=".cache",
            open_browser=True,
        )
    )


def search_songs(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch songs dynamically from Spotify Search API.

    Returns a list of dicts with:
    - title
    - artist
    - genre (unknown by default; playlist tools work with real genres)
    - mood  (unknown by default, used mainly in scoring if present)
    - spotify_url
    """
    if not query:
        return []

    sp = _get_spotify_client_for_search()
    results = sp.search(q=query, limit=limit, type="track")

    songs: List[Dict[str, Any]] = []
    for item in results.get("tracks", {}).get("items", []):
        songs.append(
            {
                "title": item.get("name", ""),
                "artist": (item.get("artists") or [{}])[0].get("name", ""),
                "genre": "unknown",
                "mood": "unknown",
                "spotify_url": item.get("external_urls", {}).get("spotify", ""),
            }
        )
    return songs


# ===== Memory Management =====
def load_memory() -> dict:
    if MEMORY_PATH.exists():
        return json.loads(MEMORY_PATH.read_text())
    return {}


def save_memory(mem: dict):
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(mem, indent=2))


def ensure_memory_file() -> None:
    """
    Create a minimal memory file on first run.
    """
    if MEMORY_PATH.exists():
        return
    save_memory(
        {
            "play_count": {},
            "liked_genres": [],
            "moods": [],
            "total_tracks": 0,
        }
    )


# ===== Preference Structure =====
@dataclass
class Preferences:
    artists: List[str]
    genres: List[str]
    mood: Optional[str] = None
    energy_min: Optional[float] = None
    energy_max: Optional[float] = None


# ===== Utility Parsing =====
def parse_csv_list(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [t.strip().lower() for t in re.split(r"[,;]", text) if t.strip()]


# ===== Recommendation Logic =====
def score_row(row: Dict[str, Any], prefs: Preferences) -> float:
    score = 0.0
    artist = str(row.get("artist", "")).lower()
    genre = str(row.get("genre", "")).lower()
    mood = str(row.get("mood", "")).lower()
    energy_value = row.get("energy")

    try:
        energy = float(energy_value) if energy_value is not None else 0.5
    except (TypeError, ValueError):
        energy = 0.5

    # Artist match
    if artist and artist in prefs.artists:
        score += 2.0

    # Genre match
    if genre and genre in prefs.genres:
        score += 1.5

    # Mood match
    if mood and prefs.mood and prefs.mood == mood:
        score += 1.0

    # (Optional) Could add energy preference if you start using it in prefs
    # if prefs.energy_min is not None and prefs.energy_max is not None:
    #     if prefs.energy_min <= energy <= prefs.energy_max:
    #         score += 0.5

    return score


def explain_reco(row: Dict[str, Any], prefs: Preferences) -> str:
    reasons: List[str] = []
    artist = row.get("artist", "")
    genre = row.get("genre", "")
    mood = row.get("mood", "")

    if artist and artist.lower() in prefs.artists:
        reasons.append(f"You like {artist}")
    if genre and genre.lower() in prefs.genres:
        reasons.append(f"Matches your genre: {genre}")
    if prefs.mood and mood and mood.lower() == prefs.mood:
        reasons.append(f"Fits your mood: {mood}")
    if not reasons:
        reasons.append("A diverse pick close to your taste")

    return " • ".join(reasons)


# ===== Memory Updates =====
def strengthen_memory(mem: Dict[str, Any], row: Dict[str, Any]) -> None:
    """
    Update memory with a single liked track row.
    Expected keys: artists (comma-separated), name, id.
    """
    mem.setdefault("play_count", {})
    mem.setdefault("liked_genres", [])
    mem.setdefault("moods", [])
    mem.setdefault("total_tracks", 0)

    artists = [
        a.strip() for a in str(row.get("artists", "")).split(",") if a.strip()
    ]
    for artist in artists:
        key = artist.lower()
        mem["play_count"][key] = mem["play_count"].get(key, 0) + 1

    mem["total_tracks"] = int(mem.get("total_tracks", 0)) + 1
    save_memory(mem)


def apply_memory_bias(row: dict, mem: dict) -> float:
    """
    Return a small positive boost if the row matches the user's memory.
    Looks at:
      - play_count (liked artists implicitly)
      - liked_genres
      - recent moods
    """
    if not mem:
        return 0.0

    boost = 0.0
    artist = str(row.get("artist", "")).lower()
    genre = str(row.get("genre", "")).lower()
    mood = str(row.get("mood", "")).lower()

    # Artist bias from play_count (implicit liked_artists)
    play_count = mem.get("play_count", {}).get(artist, 0)
    if play_count:
        # logarithmic-ish boost so big counts don't explode
        boost += min(2.0, 0.5 + 0.3 * (play_count**0.5))

    # Genre bias
    liked_genres = [g.lower() for g in mem.get("liked_genres", [])]
    if genre and genre in liked_genres:
        boost += 0.6

    # Mood recency bias (favor the last mood the user used)
    moods = mem.get("moods", [])
    if moods:
        last_mood = str(moods[-1]).lower()
        if mood and mood == last_mood:
            boost += 0.4

    return boost

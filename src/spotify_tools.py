# src/spotify_tools.py
from __future__ import annotations

from typing import Dict, List, Any
from pathlib import Path
import os
import time

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from src.curator_tools import load_memory, strengthen_memory
from dotenv import load_dotenv


# ==========================================================
# ENV LOADING (CLEAN + UNIVERSAL)
# ==========================================================

def _load_project_env() -> Path:
    """
    Loads .env from the best possible location.

    Priority:
      1) <repo>/.env
      2) <repo>/src/.env
      3) default load_dotenv search (cwd, etc.)

    Returns the path used OR None.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / ".env",   # project root
        here.parent / ".env",       # src/.env
    ]

    for p in candidates:
        if p.exists():
            load_dotenv(p)
            return p

    load_dotenv()
    return None


ENV_PATH = _load_project_env()


# ==========================================================
# HELPER: PERSISTENT ENV PATH FOR PACKAGED APP (OPTIONAL)
# ==========================================================

def get_env_path() -> Path:
    """
    Returns a writable persistent config path for packaged macOS apps.

    Priority:
      1) ~/Library/Application Support/Vignity/.env (packaged app)
      2) <repo>/.env (dev mode)
    """
    # 1 – macOS Application Support
    external_env = (
        Path.home()
        / "Library"
        / "Application Support"
        / "Vignity"
        / ".env"
    )
    if external_env.exists():
        return external_env

    # 2 – dev mode fallback
    local_env = Path(__file__).resolve().parents[1] / ".env"
    if local_env.exists():
        return local_env

    # 3 – create external env folder
    external_env.parent.mkdir(parents=True, exist_ok=True)
    external_env.touch()
    return external_env


def load_env() -> Path:
    """
    Load environment variables from wherever they should come from.
    Always returns the path actually used (good for debugging).
    """
    env_path = get_env_path()
    load_dotenv(env_path, override=True)
    return env_path


# ==========================================================
# SPOTIFY AUTH
# ==========================================================

SCOPES = (
    "user-library-read "
    "user-top-read "
    "user-read-recently-played "
    "playlist-modify-private "
    "playlist-modify-public"
)


def _build_auth_manager() -> SpotifyOAuth:
    """
    Builds an OAuth manager using whatever values are in .env.
    """
    env_path = load_env()

    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

    if not client_id or not client_secret or not redirect_uri:
        raise RuntimeError(
            f"Missing SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET / SPOTIPY_REDIRECT_URI "
            f"in {env_path}"
        )

    cache_file = get_env_path().parent / "token.cache"

    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPES,
        cache_path=str(cache_file),
        open_browser=True,
        show_dialog=True,
    )


def get_user_spotify_client() -> spotipy.Spotify:
    """
    Returns an authenticated Spotify client.
    Opens browser only the first time.
    """
    auth_manager = _build_auth_manager()
    return spotipy.Spotify(auth_manager=auth_manager)


# ==========================================================
# LOW-LEVEL FETCH HELPERS
# ==========================================================

def _fetch_liked_tracks(sp: spotipy.Spotify, limit: int = 2000) -> List[Dict[str, Any]]:
    """
    Fetch up to `limit` liked songs using pagination.
    """
    items: List[Dict[str, Any]] = []
    offset = 0
    page_size = 50

    while len(items) < limit:
        remaining = limit - len(items)
        batch_size = min(remaining, page_size)

        results = sp.current_user_saved_tracks(limit=batch_size, offset=offset)
        batch = results.get("items", [])

        if not batch:
            break

        items.extend(batch)
        offset += batch_size

        if len(batch) < batch_size:
            break

    return items[:limit]


def fetch_liked_tracks(limit: int = 2000) -> List[Dict[str, Any]]:
    """1-line wrapper."""
    sp = get_user_spotify_client()
    return _fetch_liked_tracks(sp, limit)


# ==========================================================
# MEMORY INTEGRATION
# ==========================================================

def learn_from_liked_songs(limit: int = 2000) -> None:
    """
    Feeds liked track rows into the memory system.
    """
    sp = get_user_spotify_client()
    liked = _fetch_liked_tracks(sp, limit)

    try:
        memory = load_memory()
    except Exception:
        return

    for item in liked:
        track = item.get("track") or {}
        if not track:
            continue

        row = {
            "id": track.get("id"),
            "name": track.get("name"),
            "artists": ", ".join(a.get("name", "") for a in track.get("artists", [])),
            "album": (track.get("album") or {}).get("name"),
        }

        try:
            strengthen_memory(memory, row)
        except Exception:
            continue


# ==========================================================
# GENRE-BASED PLAYLIST CREATION
# ==========================================================

def _get_artist_genres(sp: spotipy.Spotify, artist_ids: List[str]) -> Dict[str, List[str]]:
    genres: Dict[str, List[str]] = {}
    batch_size = 50

    for i in range(0, len(artist_ids), batch_size):
        batch = artist_ids[i:i + batch_size]
        data = sp.artists(batch).get("artists", [])

        for artist in data:
            aid = artist.get("id")
            if not aid:
                continue
            genres[aid] = artist.get("genres", []) or []

        time.sleep(0.05)

    return genres


def fetch_audio_features(
    sp: spotipy.Spotify, track_ids: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Batch-fetch Spotify audio features for a list of track IDs.
    Returns {track_id: {energy, danceability, valence, tempo, ...}}.
    """
    AUDIO_KEYS = [
        "energy", "danceability", "valence", "tempo",
        "acousticness", "instrumentalness", "liveness",
        "loudness", "speechiness",
    ]
    features: Dict[str, Dict[str, float]] = {}

    for i in range(0, len(track_ids), 100):
        batch = track_ids[i : i + 100]
        try:
            results = sp.audio_features(batch)
        except Exception as e:
            print(f"audio_features error (batch {i}): {e}")
            continue

        for f in results or []:
            if not f or not f.get("id"):
                continue
            features[f["id"]] = {k: f.get(k, 0.0) for k in AUDIO_KEYS}

        time.sleep(0.05)

    return features


def fetch_all_artist_genres(
    sp: spotipy.Spotify, artist_ids: List[str]
) -> Dict[str, List[str]]:
    """Public wrapper: batch-fetch genres for all unique artist IDs."""
    return _get_artist_genres(sp, artist_ids)


def _group_tracks_by_primary_genre(
    sp: spotipy.Spotify,
    liked_items: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    genre_groups: Dict[str, List[Dict[str, Any]]] = {}

    artist_ids = []
    track_to_artist = {}

    for item in liked_items:
        track = item.get("track") or {}
        tid = track.get("id")
        artists = track.get("artists") or []
        if not tid or not artists:
            continue

        main_artist = artists[0].get("id")
        if not main_artist:
            continue

        track_to_artist[tid] = main_artist
        if main_artist not in artist_ids:
            artist_ids.append(main_artist)

    if not artist_ids:
        return {}

    genre_map = _get_artist_genres(sp, artist_ids)

    for item in liked_items:
        track = item.get("track") or {}
        tid = track.get("id")
        if not tid or tid not in track_to_artist:
            continue

        aid = track_to_artist[tid]
        genres = genre_map.get(aid) or ["Unclassified"]
        primary = genres[0] if genres else "Unclassified"
        label = primary.title()

        genre_groups.setdefault(label, []).append(track)

    return genre_groups


def create_or_update_playlist(
    sp: spotipy.Spotify,
    user_id: str,
    playlist_name: str,
    tracks: List[Dict[str, Any]],
) -> None:
    playlist_id = None
    offset = 0
    while True:
        existing = sp.current_user_playlists(limit=50, offset=offset)
        for p in existing.get("items", []):
            if p.get("name") == playlist_name:
                playlist_id = p.get("id")
                break
        if playlist_id or not existing.get("next"):
            break
        offset += 50

    if playlist_id is None:
        p = sp.user_playlist_create(
            user_id,
            playlist_name,
            public=False,
            description="Generated by Vignity",
        )
        playlist_id = p["id"]
    else:
        sp.playlist_replace_items(playlist_id, [])

    ids = [t.get("id") for t in tracks if t.get("id")]
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        sp.playlist_add_items(playlist_id, chunk)


def create_playlists_by_genre(limit: int = 2000) -> None:
    sp = get_user_spotify_client()
    liked = _fetch_liked_tracks(sp, limit)

    if not liked:
        return

    user = sp.current_user()
    user_id = user["id"]

    groups = _group_tracks_by_primary_genre(sp, liked)

    for genre, tracks in groups.items():
        if len(tracks) < 5:
            continue

        playlist_name = f"Vignity – {genre}"
        create_or_update_playlist(sp, user_id, playlist_name, tracks)

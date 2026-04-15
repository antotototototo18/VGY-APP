# Playlist Generation Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the shallow name-only GPT clustering with a two-stage audio-aware pipeline that processes the full library and produces coherent, DJ-quality playlists.

**Architecture:** Fetch Spotify audio features + artist genres for all liked tracks. Pre-cluster locally with k-means on audio vectors. Send pre-group summaries (not individual tracks) to GPT for semantic merging into user-controlled N playlists. Modes become data-driven (energy mode uses real energy values, etc.).

**Tech Stack:** Python 3, PySide6, LangGraph, OpenAI GPT-4o, Spotipy, scikit-learn, numpy

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/spotify_tools.py` | Modify | Add `fetch_audio_features()`, export `fetch_all_artist_genres()` |
| `src/audio_clustering.py` | Create | Local k-means pre-clustering on audio feature vectors |
| `src/baseline_graph.py` | Rewrite | New LangGraph: enrich → pre_cluster → gpt_curate → create_playlists |
| `src/gpt_curator.py` | Remove | All logic absorbed into baseline_graph.py nodes |
| `src/main.py` | Modify | Add playlist count slider, update GPT Smart Sets action, branding fixes |
| `src/spotify_settings_page.py` | Modify | Fix branding "SpotiSet" → "Vignity" |
| `requirements.txt` | Modify | Add numpy, scikit-learn |
| `tests/test_audio_clustering.py` | Create | Unit tests for the pre-clustering module |

---

### Task 1: Add dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Add numpy and scikit-learn to the dependency list:

```
numpy>=1.24.0
scikit-learn>=1.3.0
```

Append these two lines after the existing `spotipy>=2.23.0` line.

- [ ] **Step 2: Install dependencies**

Run: `pip install numpy scikit-learn`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add numpy and scikit-learn for audio clustering"
```

---

### Task 2: Add audio features fetching to spotify_tools.py

**Files:**
- Modify: `src/spotify_tools.py` (add after `_get_artist_genres` at line 235)

- [ ] **Step 1: Add `fetch_audio_features` function**

Insert after line 235 in `src/spotify_tools.py`:

```python
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
```

- [ ] **Step 2: Add `fetch_all_artist_genres` public wrapper**

Insert after the new `fetch_audio_features` function:

```python
def fetch_all_artist_genres(
    sp: spotipy.Spotify, artist_ids: List[str]
) -> Dict[str, List[str]]:
    """Public wrapper: batch-fetch genres for all unique artist IDs."""
    return _get_artist_genres(sp, artist_ids)
```

- [ ] **Step 3: Verify imports are sufficient**

The file already imports `time`, `Dict`, `List`, `Any`, `spotipy` — no new imports needed.

- [ ] **Step 4: Commit**

```bash
git add src/spotify_tools.py
git commit -m "feat: add fetch_audio_features and fetch_all_artist_genres to spotify_tools"
```

---

### Task 3: Create the audio pre-clustering module

**Files:**
- Create: `src/audio_clustering.py`
- Create: `tests/test_audio_clustering.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_audio_clustering.py`:

```python
import pytest
from src.audio_clustering import (
    build_feature_matrix,
    pre_cluster_tracks,
    summarize_group,
    AUDIO_KEYS,
)


def _make_track(tid, energy=0.5, dance=0.5, valence=0.5, tempo=120.0, acoust=0.3,
                genres=None, artist="TestArtist", name="TestTrack"):
    return {
        "id": tid,
        "name": name,
        "artist": artist,
        "artist_genres": genres or [],
        "audio_features": {
            "energy": energy,
            "danceability": dance,
            "valence": valence,
            "tempo": tempo,
            "acousticness": acoust,
            "instrumentalness": 0.0,
            "liveness": 0.1,
            "loudness": -5.0,
            "speechiness": 0.05,
        },
    }


class TestBuildFeatureMatrix:
    def test_returns_correct_shape(self):
        tracks = [_make_track(f"id{i}") for i in range(10)]
        X, ids = build_feature_matrix(tracks)
        assert X.shape == (10, len(AUDIO_KEYS))
        assert len(ids) == 10

    def test_skips_tracks_without_audio_features(self):
        tracks = [_make_track("id1"), {"id": "id2", "name": "no audio"}]
        X, ids = build_feature_matrix(tracks)
        assert X.shape[0] == 1
        assert ids == ["id1"]

    def test_normalizes_values(self):
        tracks = [
            _make_track("a", energy=0.1, tempo=60),
            _make_track("b", energy=0.9, tempo=180),
        ]
        X, ids = build_feature_matrix(tracks)
        # After StandardScaler, mean ~0 for each column
        assert abs(X[:, 0].mean()) < 1e-10


class TestPreCluster:
    def test_returns_groups_with_required_keys(self):
        tracks = [_make_track(f"id{i}", energy=i * 0.1) for i in range(20)]
        groups = pre_cluster_tracks(tracks, n_clusters=3)
        assert len(groups) == 3
        for g in groups:
            assert "group_id" in g
            assert "track_ids" in g
            assert "size" in g
            assert "avg_features" in g
            assert "top_genres" in g
            assert "top_artists" in g
            assert len(g["track_ids"]) > 0

    def test_all_tracks_assigned(self):
        tracks = [_make_track(f"id{i}") for i in range(30)]
        groups = pre_cluster_tracks(tracks, n_clusters=5)
        all_ids = []
        for g in groups:
            all_ids.extend(g["track_ids"])
        assert sorted(all_ids) == sorted([f"id{i}" for i in range(30)])

    def test_auto_cluster_count(self):
        tracks = [_make_track(f"id{i}") for i in range(100)]
        groups = pre_cluster_tracks(tracks)
        # sqrt(100/2) = ~7, clamped to 8-25
        assert 8 <= len(groups) <= 25


class TestSummarizeGroup:
    def test_averages_features(self):
        tracks = [
            _make_track("a", energy=0.2, valence=0.8),
            _make_track("b", energy=0.8, valence=0.2),
        ]
        summary = summarize_group(tracks, ["a", "b"], group_id=0)
        assert abs(summary["avg_features"]["energy"] - 0.5) < 0.01
        assert abs(summary["avg_features"]["valence"] - 0.5) < 0.01

    def test_top_genres(self):
        tracks = [
            _make_track("a", genres=["rock", "indie"]),
            _make_track("b", genres=["rock", "metal"]),
            _make_track("c", genres=["jazz"]),
        ]
        summary = summarize_group(tracks, ["a", "b", "c"], group_id=0)
        assert summary["top_genres"][0] == "rock"

    def test_top_artists(self):
        tracks = [
            _make_track("a", artist="Radiohead"),
            _make_track("b", artist="Radiohead"),
            _make_track("c", artist="Bjork"),
        ]
        summary = summarize_group(tracks, ["a", "b", "c"], group_id=0)
        assert summary["top_artists"][0] == "Radiohead"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_audio_clustering.py -v`
Expected: ImportError — `src.audio_clustering` does not exist yet.

- [ ] **Step 3: Implement audio_clustering.py**

Create `src/audio_clustering.py`:

```python
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

AUDIO_KEYS = [
    "energy", "danceability", "valence", "tempo", "acousticness",
    "instrumentalness", "liveness", "loudness", "speechiness",
]


def build_feature_matrix(
    tracks: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a normalized feature matrix from tracks with audio_features.
    Returns (X_scaled, track_ids) — tracks without audio data are skipped.
    """
    ids: List[str] = []
    vectors: List[List[float]] = []

    for t in tracks:
        af = t.get("audio_features")
        if not af:
            continue
        vec = [float(af.get(k, 0.0)) for k in AUDIO_KEYS]
        vectors.append(vec)
        ids.append(t["id"])

    if not vectors:
        return np.empty((0, len(AUDIO_KEYS))), []

    X = np.array(vectors)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, ids


def summarize_group(
    all_tracks: List[Dict[str, Any]],
    track_ids: List[str],
    group_id: int,
) -> Dict[str, Any]:
    """
    Build a summary dict for one pre-cluster group.
    """
    id_set = set(track_ids)
    group_tracks = [t for t in all_tracks if t["id"] in id_set]

    # Average audio features (raw, not scaled)
    avg: Dict[str, float] = {}
    for key in AUDIO_KEYS:
        vals = [
            t["audio_features"].get(key, 0.0)
            for t in group_tracks
            if t.get("audio_features")
        ]
        avg[key] = round(sum(vals) / max(len(vals), 1), 3)

    # Top genres
    genre_counter: Counter = Counter()
    for t in group_tracks:
        for g in t.get("artist_genres", []):
            genre_counter[g] += 1
    top_genres = [g for g, _ in genre_counter.most_common(5)]

    # Top artists
    artist_counter: Counter = Counter()
    for t in group_tracks:
        artist_counter[t.get("artist", "")] += 1
    top_artists = [a for a, _ in artist_counter.most_common(5) if a]

    return {
        "group_id": group_id,
        "track_ids": track_ids,
        "size": len(track_ids),
        "avg_features": avg,
        "top_genres": top_genres,
        "top_artists": top_artists,
    }


def pre_cluster_tracks(
    tracks: List[Dict[str, Any]],
    n_clusters: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    K-means pre-clustering on audio feature vectors.
    Returns a list of group summaries.
    """
    X, ids = build_feature_matrix(tracks)

    if len(ids) == 0:
        return []

    if n_clusters is None:
        n_clusters = max(8, min(25, int(len(ids) ** 0.5)))

    n_clusters = min(n_clusters, len(ids))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Group track IDs by cluster label
    clusters: Dict[int, List[str]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(ids[i])

    # Build summaries
    groups: List[Dict[str, Any]] = []
    for label, cluster_ids in sorted(clusters.items()):
        summary = summarize_group(tracks, cluster_ids, group_id=label)
        groups.append(summary)

    return groups
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_audio_clustering.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/audio_clustering.py tests/test_audio_clustering.py
git commit -m "feat: add audio pre-clustering module with k-means on Spotify audio features"
```

---

### Task 4: Rewrite baseline_graph.py with new pipeline

**Files:**
- Rewrite: `src/baseline_graph.py` (full replacement, 327 lines → new)

This is the core of the overhaul. The old graph had 2 nodes (Wikipedia enrich → GPT cluster). The new graph has 4 nodes.

- [ ] **Step 1: Write the new baseline_graph.py**

Replace the entire contents of `src/baseline_graph.py` with:

```python
# src/baseline_graph.py
from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from src.audio_clustering import pre_cluster_tracks


# ==========================
# State definition
# ==========================

class PipelineState(TypedDict):
    # Inputs
    tracks: List[Dict[str, Any]]       # raw liked tracks from Spotify
    mode: str                           # Balanced | Mood | Energy | Discovery | Background
    target_playlists: int               # user-chosen playlist count

    # After enrichment
    enriched: List[Dict[str, Any]]      # tracks with audio_features, artist_genres, gpt_description

    # After pre-clustering
    pre_groups: List[Dict[str, Any]]    # audio-based cluster summaries

    # After GPT curation
    playlists: List[Dict[str, Any]]     # final playlist definitions


# ==========================
# Node 1: Enrich tracks
# ==========================

def enrich_tracks(state: PipelineState) -> PipelineState:
    """
    Fetch audio features and artist genres from Spotify,
    then batch-request GPT artist descriptions.
    Replaces the old Wikipedia enrichment entirely.
    """
    from src.spotify_tools import (
        get_user_spotify_client,
        fetch_audio_features,
        fetch_all_artist_genres,
    )

    tracks = state.get("tracks", [])
    if not tracks:
        state["enriched"] = []
        return state

    sp = get_user_spotify_client()

    # 1) Fetch audio features for all tracks
    track_ids = [t["id"] for t in tracks if t.get("id")]
    audio_map = fetch_audio_features(sp, track_ids)

    # 2) Collect unique artist IDs and fetch genres
    artist_id_to_name: Dict[str, str] = {}
    track_artist_map: Dict[str, str] = {}  # track_id -> primary artist_id

    for t in tracks:
        artists = t.get("artists", [])
        if not artists:
            continue
        primary = artists[0]
        aid = primary.get("id", "")
        if aid:
            artist_id_to_name[aid] = primary.get("name", "")
            track_artist_map[t["id"]] = aid

    genre_map = fetch_all_artist_genres(sp, list(artist_id_to_name.keys()))

    # 3) GPT batch artist descriptions (replaces Wikipedia)
    unique_artists = list(set(artist_id_to_name.values()))
    description_map = _gpt_describe_artists(unique_artists)

    # 4) Assemble enriched tracks
    enriched: List[Dict[str, Any]] = []
    for t in tracks:
        tid = t.get("id")
        if not tid:
            continue

        artists = t.get("artists", [])
        artist_name = artists[0].get("name", "") if artists else ""
        artist_id = track_artist_map.get(tid, "")

        enriched.append({
            "id": tid,
            "name": t.get("name", ""),
            "artist": artist_name,
            "album": (t.get("album") or {}).get("name", ""),
            "popularity": t.get("popularity", 0),
            "audio_features": audio_map.get(tid, {}),
            "artist_genres": genre_map.get(artist_id, []),
            "artist_description": description_map.get(artist_name, ""),
        })

    state["enriched"] = enriched
    return state


def _gpt_describe_artists(artists: List[str]) -> Dict[str, str]:
    """
    Ask GPT to describe each artist in one line (genre, era, vibe).
    Batches artists to minimize API calls.
    """
    if not artists:
        return {}

    client = OpenAI()
    descriptions: Dict[str, str] = {}
    batch_size = 50

    for i in range(0, len(artists), batch_size):
        batch = artists[i : i + batch_size]
        numbered = "\n".join(f"{j+1}. {name}" for j, name in enumerate(batch))

        prompt = f"""For each music artist below, write ONE short description (under 15 words).
Include: primary genre, era/decade, and vibe/style.
If you don't know the artist, write "unknown".

Artists:
{numbered}

Reply as a JSON object mapping artist name to description. Example:
{{"Radiohead": "90s-2000s British alt-rock, experimental and melancholic"}}
"""

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            descriptions.update(data)
        except Exception as e:
            print(f"GPT artist description error (batch {i}): {e}")

    return descriptions


# ==========================
# Node 2: Pre-cluster
# ==========================

def pre_cluster(state: PipelineState) -> PipelineState:
    """
    Local k-means clustering on audio feature vectors.
    Produces sonic neighborhoods for GPT to merge.
    """
    enriched = state.get("enriched", [])
    if not enriched:
        state["pre_groups"] = []
        return state

    target = state.get("target_playlists", 7)
    # Aim for ~2-3x the target playlist count in pre-groups
    n_pre = max(target * 2, 8)
    n_pre = min(n_pre, len(enriched))

    groups = pre_cluster_tracks(enriched, n_clusters=n_pre)
    state["pre_groups"] = groups
    return state


# ==========================
# Node 3: GPT semantic curation
# ==========================

MODE_INSTRUCTIONS = {
    "mood": (
        "Group by emotional character. Use valence (<0.3 = melancholy, >0.7 = joyful), "
        "acousticness, and artist descriptions to identify emotional threads. "
        "Each playlist should evoke a distinct feeling."
    ),
    "energy": (
        "Create an energy gradient. Split into distinct tiers using energy and tempo values. "
        "Each playlist should have a narrow energy band. Order from chill to intense."
    ),
    "discovery": (
        "Maximize genre diversity within each playlist. Each playlist should bridge "
        "at least 2-3 different genres that share a sonic quality (similar energy, tempo, or valence)."
    ),
    "background": (
        "Select only neighborhoods with avg energy <0.5 and acousticness >0.3. "
        "Create smooth, unobtrusive flows. Merge neighborhoods that share calm, ambient qualities."
    ),
    "balanced": (
        "Create well-rounded playlists that balance genre coherence with variety. "
        "Each playlist should have a clear identity but not be one-dimensional. "
        "Consider genre, era, and audio profile when merging neighborhoods."
    ),
}


def gpt_curate(state: PipelineState) -> PipelineState:
    """
    GPT merges/splits pre-groups into final playlists with names and descriptions.
    """
    pre_groups = state.get("pre_groups", [])
    if not pre_groups:
        state["playlists"] = []
        return state

    mode = (state.get("mode") or "Balanced").lower()
    target_n = state.get("target_playlists", 7)
    total_tracks = sum(g["size"] for g in pre_groups)

    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["balanced"])

    # Build the group summary table for GPT
    group_lines = []
    for g in pre_groups:
        af = g.get("avg_features", {})
        line = (
            f"Neighborhood {g['group_id']}:\n"
            f"  Tracks: {g['size']}\n"
            f"  Avg energy: {af.get('energy', 0):.2f}, "
            f"danceability: {af.get('danceability', 0):.2f}, "
            f"valence: {af.get('valence', 0):.2f}, "
            f"tempo: {af.get('tempo', 0):.0f} BPM, "
            f"acousticness: {af.get('acousticness', 0):.2f}\n"
            f"  Top genres: {', '.join(g.get('top_genres', [])[:5]) or 'unknown'}\n"
            f"  Top artists: {', '.join(g.get('top_artists', [])[:5])}"
        )
        group_lines.append(line)

    neighborhoods = "\n\n".join(group_lines)

    system = f"""You are a world-class music curator building playlists from a listener's library.

Below are {len(pre_groups)} sonic neighborhoods from a library of {total_tracks} tracks.
Each neighborhood is a cluster of sonically similar tracks with their average audio profile.

Create exactly {target_n} playlists by merging or splitting these neighborhoods.

Mode: {mode.title()}
{mode_instruction}

For each playlist, return:
- "name": an evocative 2-5 word title (not generic like "Rock Mix" or "Chill Vibes")
- "description": 1-2 sentences capturing the specific mood/vibe
- "group_ids": array of neighborhood IDs to include in this playlist
- "track_order": one of "ascending_energy", "descending_energy", "valence_arc", "tempo_smooth", "shuffle"

Rules:
- Every neighborhood must be assigned to exactly one playlist
- No neighborhood should appear in multiple playlists
- Playlists should be roughly balanced in size (avoid one giant playlist and several tiny ones)
- Names should be evocative and specific, not generic

Return STRICT JSON only:
{{"playlists": [{{...}}]}}"""

    client = OpenAI()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": neighborhoods},
            ],
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        playlists = data.get("playlists", [])

        if not isinstance(playlists, list):
            state["playlists"] = []
            return state

        # Resolve group_ids to actual track_ids
        group_lookup = {g["group_id"]: g["track_ids"] for g in pre_groups}

        for pl in playlists:
            resolved_ids: List[str] = []
            for gid in pl.get("group_ids", []):
                resolved_ids.extend(group_lookup.get(gid, []))
            pl["track_ids"] = resolved_ids
            pl["size"] = len(resolved_ids)

        state["playlists"] = playlists
        return state

    except Exception as e:
        print(f"GPT curation error: {e}")
        state["playlists"] = []
        return state


# ==========================
# Node 4: Order tracks within playlists
# ==========================

def order_tracks(state: PipelineState) -> PipelineState:
    """
    Deterministically order tracks within each playlist based on
    GPT's suggested strategy and the actual audio features.
    """
    playlists = state.get("playlists", [])
    enriched = state.get("enriched", [])

    if not playlists or not enriched:
        return state

    track_lookup = {t["id"]: t for t in enriched}

    for pl in playlists:
        strategy = pl.get("track_order", "shuffle")
        ids = pl.get("track_ids", [])

        if strategy == "ascending_energy":
            ids.sort(key=lambda tid: track_lookup.get(tid, {}).get("audio_features", {}).get("energy", 0.5))
        elif strategy == "descending_energy":
            ids.sort(key=lambda tid: track_lookup.get(tid, {}).get("audio_features", {}).get("energy", 0.5), reverse=True)
        elif strategy == "valence_arc":
            # Start low, peak in middle, end low (emotional arc)
            ids.sort(key=lambda tid: track_lookup.get(tid, {}).get("audio_features", {}).get("valence", 0.5))
            mid = len(ids) // 2
            # Interleave: ascending first half, descending second half
            ascending = ids[:mid]
            descending = list(reversed(ids[mid:]))
            ids = []
            for i in range(max(len(ascending), len(descending))):
                if i < len(ascending):
                    ids.append(ascending[i])
                if i < len(descending):
                    ids.append(descending[i])
        elif strategy == "tempo_smooth":
            ids.sort(key=lambda tid: track_lookup.get(tid, {}).get("audio_features", {}).get("tempo", 120))

        pl["track_ids"] = ids

    state["playlists"] = playlists
    return state


# ==========================
# Build the LangGraph
# ==========================

graph = StateGraph(PipelineState)
graph.add_node("enrich", enrich_tracks)
graph.add_node("pre_cluster", pre_cluster)
graph.add_node("gpt_curate", gpt_curate)
graph.add_node("order_tracks", order_tracks)

graph.add_edge("enrich", "pre_cluster")
graph.add_edge("pre_cluster", "gpt_curate")
graph.add_edge("gpt_curate", "order_tracks")
graph.set_entry_point("enrich")
graph.set_finish_point("order_tracks")

app = graph.compile()
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from src.baseline_graph import app; print('Graph compiled:', app)"`
Expected: No import errors, prints graph object.

- [ ] **Step 3: Commit**

```bash
git add src/baseline_graph.py
git commit -m "feat: rewrite baseline_graph with audio-aware 4-node pipeline"
```

---

### Task 5: Rewrite gpt_curator.py entry point

**Files:**
- Rewrite: `src/gpt_curator.py` (483 lines → ~120 lines)

The old file had duplicate clustering logic, broken `responses.create` calls, and Wikipedia-dependent code. The new version is a thin orchestrator that calls the LangGraph pipeline and writes playlists to Spotify.

- [ ] **Step 1: Replace gpt_curator.py entirely**

```python
# src/gpt_curator.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import spotipy

from src.spotify_tools import get_user_spotify_client, _fetch_liked_tracks
from src.baseline_graph import app as pipeline


def _valid_spotify_id(tid: str) -> bool:
    """Basic sanity check for a Spotify track ID."""
    return isinstance(tid, str) and len(tid) == 22 and tid.isalnum()


def _create_or_replace_playlist(
    sp: spotipy.Spotify,
    user_id: str,
    name: str,
    desc: str,
    track_ids: List[str],
) -> Optional[str]:
    """Find or create a playlist by name and replace its contents."""
    if not track_ids:
        return None

    playlist_id: Optional[str] = None
    offset = 0

    while True:
        existing = sp.current_user_playlists(limit=50, offset=offset)
        for p in existing.get("items", []):
            if p.get("name") == name:
                playlist_id = p.get("id")
                break
        if playlist_id or not existing.get("next"):
            break
        offset += 50

    if playlist_id is None:
        p = sp.user_playlist_create(
            user_id, name, public=False,
            description=desc or "Generated by Vignity",
        )
        playlist_id = p["id"]
    else:
        sp.playlist_replace_items(playlist_id, [])
        sp.playlist_change_details(playlist_id, description=desc)

    for i in range(0, len(track_ids), 100):
        chunk = track_ids[i : i + 100]
        if chunk:
            sp.playlist_add_items(playlist_id, chunk)

    return playlist_id


def gpt_cluster_and_create_playlists(
    limit: int = 200,
    mode: str = "Balanced",
    target_playlists: int = 7,
    on_progress=None,
) -> List[str]:
    """
    Main entry point for the Vignity smart playlist pipeline.

    1. Fetch liked tracks
    2. Run the LangGraph pipeline (enrich → pre-cluster → GPT curate → order)
    3. Create/update playlists on Spotify
    4. Return playlist summaries
    """
    def log(msg, level="info"):
        if on_progress:
            on_progress(msg, level)
        print(msg)

    sp = get_user_spotify_client()

    log(f"Fetching up to {limit} liked tracks...", "running")
    liked_items = _fetch_liked_tracks(sp, limit)
    tracks = [item.get("track") for item in liked_items if item.get("track")]

    if not tracks:
        log("No tracks found.", "error")
        return []

    log(f"Found {len(tracks)} tracks. Starting enrichment pipeline...", "running")

    # Build input state — pass raw Spotify track objects
    valid_tracks = [
        t for t in tracks
        if isinstance(t, dict) and t.get("id") and _valid_spotify_id(t["id"])
    ]

    state = {
        "tracks": valid_tracks,
        "mode": mode,
        "target_playlists": target_playlists,
    }

    # Run the LangGraph pipeline
    result = pipeline.invoke(state)

    playlists = result.get("playlists", [])
    if not playlists:
        log("Pipeline returned no playlists.", "error")
        return []

    log(f"Creating {len(playlists)} playlists on Spotify...", "running")

    user = sp.current_user()
    user_id = user["id"]
    summaries: List[str] = []

    for pl in playlists:
        name = f"Vignity \u2013 {pl.get('name', 'Smart Set')}"
        desc = pl.get("description", f"Curated by Vignity ({mode} mode)")
        track_ids = [tid for tid in pl.get("track_ids", []) if _valid_spotify_id(tid)]

        if not track_ids:
            continue

        playlist_id = _create_or_replace_playlist(sp, user_id, name, desc, track_ids)

        if playlist_id:
            summary = f"{name} ({len(track_ids)} tracks) \u2014 {desc}"
            summaries.append(summary)
            log(f"\u2714 {name} \u2014 {len(track_ids)} tracks", "success")

    return summaries
```

- [ ] **Step 2: Verify import**

Run: `python -c "from src.gpt_curator import gpt_cluster_and_create_playlists; print('OK')"`
Expected: Prints "OK" with no errors.

- [ ] **Step 3: Commit**

```bash
git add src/gpt_curator.py
git commit -m "feat: rewrite gpt_curator as thin orchestrator using new LangGraph pipeline"
```

---

### Task 6: Update main.py — playlist count slider and progress logging

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Add playlist count slider to the left panel**

In `MainWindow.__init__`, after the GPT mode combobox section (after line 385 `left.addLayout(row)`), add:

```python
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
```

- [ ] **Step 2: Update the `action_gpt_smart_sets` method**

Replace the current `action_gpt_smart_sets` method (lines 493-504) with:

```python
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
```

- [ ] **Step 3: Verify the app launches**

Run: `python -m src.main`
Expected: App window opens with the new "PLAYLIST COUNT" spinner visible below the GPT mode selector.

- [ ] **Step 4: Commit**

```bash
git add src/main.py
git commit -m "feat: add playlist count control and progress logging to GPT Smart Sets"
```

---

### Task 7: Branding cleanup — SpotiSet → Vignity

**Files:**
- Modify: `src/spotify_tools.py` (lines 58, 66-67, 307, 334)
- Modify: `src/spotify_settings_page.py`
- Modify: `src/main.py` (line 83)

- [ ] **Step 1: Fix spotify_tools.py branding**

In `src/spotify_tools.py`:

Replace `"SpotiSet"` with `"Vignity"` in `get_env_path()` (line 67):
```python
        / "Vignity"
```

Replace the playlist description on line 307:
```python
            description="Generated by Vignity",
```

Replace the playlist name prefix on line 334:
```python
        playlist_name = f"Vignity – {genre}"
```

- [ ] **Step 2: Fix main.py logo path**

In `src/main.py` line 83, update the logo path:
```python
LOGO_PATH = resource_path("assets", "vignity_logo.png")
```

Also rename the actual asset file:
```bash
mv assets/spotiset_logo.PNG assets/vignity_logo.png
```

- [ ] **Step 3: Fix spotify_settings_page.py**

Read the file and replace any remaining "SpotiSet" references in the dialog title and labels with "Vignity".

- [ ] **Step 4: Verify branding**

Run: `grep -ri "spotiset" src/ --include="*.py" -l`
Expected: No files returned (all references cleaned).

- [ ] **Step 5: Commit**

```bash
git add src/spotify_tools.py src/main.py src/spotify_settings_page.py assets/
git commit -m "chore: complete SpotiSet → Vignity branding rename"
```

---

### Task 8: Remove wikipedia dependency

**Files:**
- Modify: `requirements.txt` (remove wikipedia if listed)

- [ ] **Step 1: Check and clean requirements**

The `wikipedia` package is imported in the old `baseline_graph.py` but is no longer used. Verify it's not in requirements.txt (it may have been installed separately). If it is, remove it.

Also verify the old `baseline_graph.py` no longer imports wikipedia (it shouldn't after the rewrite).

- [ ] **Step 2: Verify no remaining wikipedia imports**

Run: `grep -r "import wikipedia" src/`
Expected: No results.

- [ ] **Step 3: Commit if any changes**

```bash
git add requirements.txt
git commit -m "chore: remove wikipedia dependency (replaced by GPT artist descriptions)"
```

---

### Task 9: Integration test — full pipeline smoke test

- [ ] **Step 1: Verify all imports work**

Run:
```bash
python -c "
from src.spotify_tools import fetch_audio_features, fetch_all_artist_genres, get_user_spotify_client
from src.audio_clustering import pre_cluster_tracks, build_feature_matrix
from src.baseline_graph import app
from src.gpt_curator import gpt_cluster_and_create_playlists
print('All imports OK')
"
```
Expected: "All imports OK"

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 3: Launch the app and verify UI**

Run: `python -m src.main`

Verify:
- App window opens
- "PLAYLIST COUNT" spinner appears below GPT mode selector, range 3-20, default 7
- "VIGNITY" branding everywhere, no "SpotiSet" visible
- All buttons clickable
- GPT Smart Sets button passes limit, mode, and playlist count to the pipeline

- [ ] **Step 4: End-to-end test with Spotify (manual)**

1. Set scope to 50 (small for testing)
2. Select "Balanced" mode, 5 playlists
3. Click "RUN GPT SMART SETS"
4. Watch console for progress:
   - "Fetching up to 50 liked tracks..."
   - "Found X tracks. Starting enrichment pipeline..."
   - "Creating 5 playlists on Spotify..."
   - Individual playlist confirmations
5. Check Spotify — 5 new "Vignity – ..." playlists should exist

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete playlist generation overhaul — audio-aware two-stage clustering"
```

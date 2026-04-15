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

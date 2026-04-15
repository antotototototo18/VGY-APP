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
        tracks = [
            _make_track(f"id{i}", energy=i / 100, dance=i / 100, valence=(100 - i) / 100, tempo=60 + i * 1.2)
            for i in range(100)
        ]
        groups = pre_cluster_tracks(tracks)
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

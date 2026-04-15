# src/baseline_graph.py
from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from src.audio_clustering import pre_cluster_tracks
from src.utils.resource_loader import load_app_env, get_env_var

load_app_env()


# ==========================
# State definition
# ==========================

class PipelineState(TypedDict):
    # Inputs
    tracks: List[Dict[str, Any]]
    mode: str
    target_playlists: int

    # After enrichment
    enriched: List[Dict[str, Any]]

    # After pre-clustering
    pre_groups: List[Dict[str, Any]]

    # After GPT curation
    playlists: List[Dict[str, Any]]


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

    # 1) Fetch audio features for all tracks (may fail with 403 if endpoint is restricted)
    track_ids = [t["id"] for t in tracks if t.get("id")]
    audio_map = fetch_audio_features(sp, track_ids)
    has_audio = len(audio_map) > 0
    if not has_audio:
        print("Audio features unavailable (Spotify API restriction). Falling back to genre+description clustering.")

    # 2) Collect unique artist IDs and fetch genres
    artist_id_to_name: Dict[str, str] = {}
    track_artist_map: Dict[str, str] = {}

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

    api_key = get_env_var("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set — skipping artist descriptions.")
        return {}
    client = OpenAI(api_key=api_key)
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
    Pre-cluster tracks into neighborhoods.
    Uses k-means on audio features when available, falls back to genre-based grouping.
    """
    enriched = state.get("enriched", [])
    if not enriched:
        state["pre_groups"] = []
        return state

    target = state.get("target_playlists", 7)
    n_pre = max(target * 2, 8)
    n_pre = min(n_pre, len(enriched))

    # Check if we have audio features
    has_audio = any(t.get("audio_features") for t in enriched)

    if has_audio:
        groups = pre_cluster_tracks(enriched, n_clusters=n_pre)
    else:
        # Fallback: group by primary genre
        groups = _genre_based_grouping(enriched, n_pre)

    state["pre_groups"] = groups
    return state


def _genre_based_grouping(
    tracks: List[Dict[str, Any]], target_groups: int
) -> List[Dict[str, Any]]:
    """
    Fallback grouping when audio features are unavailable.
    Groups tracks by their primary artist genre.
    """
    from collections import Counter

    genre_buckets: Dict[str, List[str]] = {}
    for t in tracks:
        genres = t.get("artist_genres", [])
        primary = genres[0] if genres else "unknown"
        genre_buckets.setdefault(primary, []).append(t["id"])

    # Merge tiny genres into "other" if we have too many groups
    if len(genre_buckets) > target_groups * 2:
        sorted_genres = sorted(genre_buckets.items(), key=lambda x: len(x[1]), reverse=True)
        kept = dict(sorted_genres[:target_groups * 2 - 1])
        other_ids = []
        for genre, ids in sorted_genres[target_groups * 2 - 1:]:
            other_ids.extend(ids)
        if other_ids:
            kept["other"] = other_ids
        genre_buckets = kept

    # Build group summaries
    from src.audio_clustering import summarize_group
    groups = []
    for i, (genre, track_ids) in enumerate(sorted(genre_buckets.items(), key=lambda x: -len(x[1]))):
        summary = summarize_group(tracks, track_ids, group_id=i)
        groups.append(summary)

    return groups


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

    # Check if audio features are available
    has_audio = any(
        any(v != 0 for v in g.get("avg_features", {}).values())
        for g in pre_groups
    )

    group_lines = []
    for g in pre_groups:
        af = g.get("avg_features", {})
        if has_audio:
            audio_line = (
                f"  Avg energy: {af.get('energy', 0):.2f}, "
                f"danceability: {af.get('danceability', 0):.2f}, "
                f"valence: {af.get('valence', 0):.2f}, "
                f"tempo: {af.get('tempo', 0):.0f} BPM, "
                f"acousticness: {af.get('acousticness', 0):.2f}\n"
            )
        else:
            audio_line = ""
        line = (
            f"Neighborhood {g['group_id']}:\n"
            f"  Tracks: {g['size']}\n"
            f"{audio_line}"
            f"  Top genres: {', '.join(g.get('top_genres', [])[:5]) or 'unknown'}\n"
            f"  Top artists: {', '.join(g.get('top_artists', [])[:5])}"
        )
        group_lines.append(line)

    neighborhoods = "\n\n".join(group_lines)

    audio_context = (
        "Each neighborhood is a cluster of sonically similar tracks with their average audio profile."
        if has_audio else
        "Each neighborhood is a cluster of tracks grouped by genre and artist similarity."
    )

    system = f"""You are a world-class music curator building playlists from a listener's library.

Below are {len(pre_groups)} neighborhoods from a library of {total_tracks} tracks.
{audio_context}

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

    api_key = get_env_var("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set — cannot curate playlists.")
        state["playlists"] = []
        return state
    client = OpenAI(api_key=api_key)

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
            ids.sort(key=lambda tid: track_lookup.get(tid, {}).get("audio_features", {}).get("valence", 0.5))
            mid = len(ids) // 2
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

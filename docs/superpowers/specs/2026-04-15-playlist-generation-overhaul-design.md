# Playlist Generation Overhaul — Design Spec

**Date:** 2026-04-15
**Goal:** Transform Vignity's playlist generation from shallow name-based guessing to rich audio-aware curation that feels like a human DJ.

## Problem Statement

Current playlist quality is underwhelming across the board:
- GPT clusters with only artist names + fragile Wikipedia bios — no audio data
- 80-track hard cap silently discards 96% of a large library
- Mode selection (Mood/Energy/etc.) barely changes output — no real data backing the modes
- Two diverging code paths (`gpt_curator.py` vs `baseline_graph.py`), unclear which runs
- Wikipedia enrichment fails silently for obscure/international artists
- No user control over playlist count
- Discovery mode uses naive name search instead of audio similarity

**Target user:** Large, eclectic library (1500+ tracks, many genres). Wants flexible playlist count with each playlist feeling coherently curated.

## Design

### 1. Data Pipeline

**Current:** `Liked tracks (80 cap) → Wikipedia bios → GPT clusters → playlists`

**New:** `All liked tracks → audio features → artist genres → GPT artist descriptions → rich clustering → playlists`

Steps:
1. **Fetch all liked tracks** — remove 80-track cap, paginate full library
2. **Fetch audio features** — Spotify `audio_features` endpoint, batches of 100. Per track: energy, danceability, valence, tempo, acousticness, instrumentalness, liveness, loudness, speechiness
3. **Fetch artist genres** — batch-fetch via Spotify `artists` endpoint for all unique artists (batches of 50)
4. **GPT artist descriptions** — send batches of unique artist names to GPT, get 1-line style description (genre, era, vibe, influences). Replaces Wikipedia entirely. For unknowns, returns "unknown" — we fall back to audio features alone
5. **Assemble rich track profiles** — each track: name, artist, album, 9 audio feature dimensions, artist genres, artist description, popularity

No hard track limit. Full library processed.

### 2. Clustering Strategy — Two-Stage

**Stage 1: Local audio pre-grouping**
- K-means or agglomerative clustering on normalized audio feature vectors (energy, danceability, valence, tempo, acousticness)
- Runs locally with scikit-learn — no API call, fast, deterministic
- Target ~15-25 pre-groups depending on library size
- Each pre-group is sonically coherent

**Stage 2: GPT semantic merging & curation**
- GPT receives pre-group summaries: track count, avg audio profile, top genres, top artists, style descriptions
- GPT merges/splits/renames into N playlists (user-controlled count)
- GPT assigns evocative names and descriptions
- Track ordering within playlists uses audio features deterministically (energy arcs, mood progression)

**Mode behavior with real data:**
- **Balanced** — merge pre-groups by genre/era affinity
- **Mood** — merge by valence + acousticness + energy (actual mood dimensions)
- **Energy** — sort and split by energy/tempo bands (chill → intense)
- **Discovery** — maximize genre diversity within each playlist
- **Background** — filter to low-energy/high-acousticness, smooth flow

### 3. GPT Prompting & Output

**System prompt:** GPT acts as world-class music curator. Receives:
- Summary table of pre-groups (ID, size, avg energy/valence/tempo, top genres, top artists, description)
- User's target playlist count (N)
- Active mode with specific audio criteria

**Prompt template:**
```
You are a world-class music curator. Below are {X} sonic neighborhoods
from a listener's library of {total} tracks.

Each neighborhood has: track count, average audio profile,
top genres, top artists, and style descriptions.

Create exactly {N} playlists by merging/splitting these neighborhoods.
Mode: {mode} — {detailed mode instruction with specific audio criteria}

For each playlist:
- name: evocative 2-5 word title (not generic like "Rock Mix")
- description: 1-2 sentences capturing the vibe
- groups: which neighborhood IDs to include
- track_order: ordering strategy (e.g., "ascending energy", "valence arc")
```

**Mode-specific instructions:**
- **Mood:** "Group by emotional character. Use valence (<0.3 = melancholy, >0.7 = joyful), acousticness, and artist descriptions to identify emotional threads."
- **Energy:** "Create an energy gradient. Split into distinct tiers using energy and tempo values. Each playlist should have a narrow energy band."
- **Discovery:** "Maximize genre diversity within each playlist. Each playlist should bridge at least 2-3 different genres that share a sonic quality."
- **Background:** "Select only tracks with energy <0.5 and acousticness >0.3. Create smooth, unobtrusive flows. Order by minimal tempo variation."
- **Balanced:** "Create well-rounded playlists that balance genre coherence with variety. Each playlist should have a clear identity but not be one-dimensional."

**Output format:** Enforced via `response_format` JSON schema — no regex extraction.

**Track ordering:** Deterministic post-processing using audio features per GPT's suggested strategy (ascending energy, valence arc, etc.).

### 4. Code Architecture

**Consolidate into one pipeline. Remove diverging paths.**

**LangGraph flow (new nodes):**
```
enrich_tracks (audio features + genres + GPT artist descriptions)
  → pre_cluster (local k-means on audio vectors)
  → gpt_curate (semantic merging with full context)
  → create_playlists (write to Spotify)
```

**Files changed:**
- `spotify_tools.py` — add `fetch_audio_features()`, improve `get_artist_genres()` to handle all unique artists
- `baseline_graph.py` — rewrite with new nodes, new state schema including audio data
- `gpt_curator.py` — remove (all GPT logic moves into graph nodes)
- `curator_tools.py` — wire preferences into clustering input
- `main.py` — add playlist count slider (3-20, default 7), update action handler
- `requirements.txt` — add numpy, scikit-learn
- `library_organizer.py` — unchanged (simple artist-based fallback)

**OpenAI API:** Use `chat.completions.create` consistently, standardize on `gpt-4o` (or user-configurable).

### 5. User Controls

- **Playlist count slider** — 3 to 20, default 7
- **Mode selector** — unchanged (Balanced/Mood/Energy/Discovery/Background), now data-driven
- **Scope spinner** — unchanged, but all selected tracks are processed (no silent cap)

**Console feedback during generation:**
- "Fetching audio features... 400/1800 tracks"
- "Found 18 sonic neighborhoods"
- "Merging into 7 playlists..."
- Per-playlist summary: name, track count, vibe

**Playlist naming:** "Vignity – {evocative name}" (e.g., "Vignity – Midnight Drift")

### 6. Branding & Rename

- Repo: `spotiset_clean_demo` → `vignity`
- Website repo updated
- All "SpotiSet" strings in code/UI → "Vignity"
- `spotiset_logo.PNG` → `vignity_logo.png`
- Settings dialog title: "SpotiSet" → "Vignity"
- PyInstaller spec: consolidate to `Vignity.spec` only

### 7. New Dependencies

- `numpy` — array operations for audio feature vectors
- `scikit-learn` — k-means clustering

Both lightweight, well-established, no compatibility concerns with existing stack.

### 8. What's NOT Changing

- Scan Liked Songs button — still fetches and learns from library
- Build Genre Playlists — simple genre-based grouping
- Organize By Artist — simple artist buckets
- CosmosConsole UI — typing animation, waveform overlay
- Memory system in curator_tools — stays, gets wired in
- PyInstaller packaging — stays compatible

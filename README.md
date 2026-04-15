# Vignity – GPT-Powered Spotify Library Curator

Vignity is an **AI-assisted Spotify curator** that scans your liked songs, learns your tastes, and builds smart playlists using:

- A **PySide6 desktop UI**
- A **LangGraph agent** that orchestrates tools
- **Spotify Web API** for library access
- **Wikipedia enrichment** for artists
- **OpenAI GPT models** for clustering + curation

The goal is to act like a **small autonomous music assistant** that understands your library and proposes structured, meaningful playlists.

---

## Core Features

### 1. Library scan & telemetry
- Fetches up to **2,000 liked tracks** from your Spotify account.
- Builds a **telemetry view** of:
  - Most listened artists
  - Genre distribution
  - Candidate tracks for curation
- Stores user preferences in `memory/user_prefs.json` for re-use.

### 2. Curation actions (left panel)

From the main UI you can trigger several one-click workflows:

1. **SCAN LIKED SONGS**  
   - Downloads liked tracks via Spotify API.  
   - Updates internal memory and preferences.

2. **BUILD GENRE PLAYLISTS**  
   - Groups tracks by genre using telemetry.  
   - Builds themed playlists (e.g. "Vignity – Shoegaze").

3. **ORGANIZE BY ARTIST**  
   - Implemented in `src/library_organizer.py`.  
   - Creates one playlist per artist with at least _N_ liked songs  
     → playlists like **"Vignity – Radiohead"**.

4. **VIGNITY PICKS**  
   - Uses your play history/preferences from `user_prefs.json`.  
   - Picks a smaller, curated set of tracks as recommendations.

### 3. GPT "Smart Sets" (agentic behaviour)

In the **SMART SET: GPT POWERED** section you can choose a **GPT mode**:

- `Balanced`
- `Mood`
- `Energy`
- `Discovery`
- `Background`

When you click **RUN GPT SMART SETS**:

1. `src/spotify_tools.py` fetches a slice of your liked tracks.
2. `src/baseline_graph.py` builds a **LangGraph StateGraph** that:
   - Enriches tracks with **Wikipedia bios** for their artists.
   - Calls **OpenAI** to cluster tracks into coherent groups
     (e.g. "High-energy indie rock", "Dreamy late-night electronic").
3. `src/gpt_curator.py` interprets those clusters and suggests
   one or more playlists per "theme".
4. The app writes these playlists back to your Spotify account.

The agent decides **what to group with what**, and **what to create**, given high-level parameters (mode + track scope) rather than manual playlist editing.

---

## Architecture Overview

### UI Layer (PySide6)
- `src/main.py`
  - Main window: **"Vignity – AI Spotify Curator"**
  - Left panel: actions + GPT mode selector + scan limits.
  - Right panel: **log console**, status messages, and telemetry headers.
  - Animated panels and a custom loader overlay.
- `src/spotify_settings_page.py`
  - Settings dialog.
  - Lets a user enter their own Spotify credentials.
  - "TEST CONNECTION" button to verify OAuth works.

### Spotify Integration
- `src/spotify_tools.py`
  - Handles **OAuth** using environment variables:
    - `SPOTIPY_CLIENT_ID`
    - `SPOTIPY_CLIENT_SECRET`
    - `SPOTIPY_REDIRECT_URI`
  - Creates an authenticated `spotipy.Spotify` client.
  - Implements `_fetch_liked_tracks(...)` and helpers to create or update playlists.
  - Wraps some calls with **memory strengthening** (`load_memory`, `strengthen_memory`).

### Memory & Preferences
- `src/curator_tools.py`
  - Reads/writes `memory/user_prefs.json`.
  - Tracks:
    - per-artist play counts
    - liked genres
    - moods (if any)
  - Provides `Preferences` dataclass and helper functions to:
    - Explain **why** a track is recommended.
    - Look up new tracks by artist/genre via Spotify search.

- `memory/user_prefs.json`
  - JSON file that accumulates:
    - Artists you listen to the most
    - Genres you tend to like
    - Example tracks representing your taste

### Agentic Graph & GPT Curation
- `src/baseline_graph.py`
  - Defines a `TrackState` TypedDict with fields like `tracks`, `enriched`, `clusters`.
  - Creates a **LangGraph `StateGraph`** to orchestrate:
    1. Fetch/enrich tracks with Wikipedia bios.
    2. Call OpenAI (e.g. `gpt-4.1`) with a structured prompt to:
       - Detect themes
       - Cluster tracks
       - Output JSON clusters
    3. Return structured cluster data for playlist creation.

- `src/gpt_curator.py`
  - High-level GPT-based curator.
  - Calls OpenAI to:
    - Group tracks into "sets" with descriptions.
    - Output JSON that the app turns into actual Spotify playlists.

### Utility / Orchestration
- `src/library_organizer.py`
  - Implements "ORGANIZE BY ARTIST" feature.
- `assets/vignity_logo.png`
  - Logo for UI / README usage.

---

## Installation & Run

### 1) Clone the repository

```bash
git clone https://github.com/<your-username>/vignity.git
cd vignity
```

### 2) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

Create a `.env` file at the repo root:

```bash
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
OPENAI_API_KEY=your_openai_api_key
```

### 5) Run the app

```bash
python -m src.main
```

On first run, a browser window opens to complete Spotify OAuth.

### First run notes

- OAuth stores a local cache at `token.cache` in the project root.
- Memory is persisted in `memory/user_prefs.json` after you scan your library.

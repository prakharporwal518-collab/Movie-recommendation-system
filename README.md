# 🎬 CineMatch — Movie Recommendation System

Type in a movie you like and get similar movies, matched on **plot, genre, cast & director, or keywords**.
It's a **content-based filtering** recommender built with **TF-IDF** and **cosine similarity** (scikit-learn), and it has a scroll-animated 3D web UI.

## ML concepts

| Concept | Where it's used |
|---|---|
| **Content-based filtering** | Movies are recommended from their own metadata, not from other users' ratings. |
| **Feature vectorization (TF-IDF)** | `TfidfVectorizer` turns each movie's text into a sparse vector. Words that are common in one movie but rare across all movies get the highest weight. |
| **Cosine similarity** | `cosine_similarity` measures the angle between two vectors: 1.0 means the same direction and 0 means nothing in common. |
| **Weighted feature blending** | "Everything" mode combines four similarity scores: plot 35%, keywords 25%, cast 20%, genre 20%. |

### Pipeline

```
movies_metadata + credits + keywords
        │  pandas: merge on id, parse JSON-like columns, drop movies with < 10 votes
        ▼
4 text documents per movie
  plot      = overview + tagline                  (stop-words removed, 1–2-grams)
  genre     = "action crime drama"
  cast      = "christianbale heathledger … christophernolan christophernolan"
  keywords  = "joker vigilante dc comics …"
        │  TfidfVectorizer (one per feature)
        ▼
sparse TF-IDF matrices (≈23k movies)
        │  cosine_similarity(selected movie, all movies)
        ▼
weighted score → top-N recommendations
```

Each multi-word name is merged into one token (`Tom Hanks` → `tomhanks`), so "Tom Hanks" and "Tom Cruise" don't count as a match just because they share "Tom". The director is counted twice so it gets more weight.

Similarity is computed **for one movie at a time** (1 × N) instead of building a full N × N matrix. That full matrix would need about 4 GB of memory for 23k movies.

## Dataset

[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (TMDB metadata, 45k movies). It's included in this repo:

- `movies_metadata.zip`: title, overview, genres, rating, poster
- `credits_part1-4.zip`: cast and crew (split into four parts to stay under GitHub's file-size limit)
- `keywords.csv`: TMDB plot keywords

The code reads the zips directly, so you don't need to unzip anything.

## Run it

```bash
pip install -r requirements.txt

# Command line
python recommender.py "The Dark Knight"
python recommender.py "Toy Story" --mode genre -n 5      # modes: all, plot, genre, cast, keywords

# Web app → http://127.0.0.1:5000
python app.py

# Tests
pytest -q
```

The first run builds the model in about 30 seconds and caches it to `model/recommender.pkl` (25 MB, git-ignored). Runs after that start instantly. Use `--rebuild` to rebuild the model after you change the features.

### Example

```
Because you liked: The Dark Knight (2008)  [mode: all]
 1. The Dark Knight Rises (2012)  sim=0.569
 2. Batman Begins (2005)          sim=0.453
 3. Following (1998)              sim=0.264   ← another Christopher Nolan film
```

## Web UI

`static/index.html` has no dependencies: plain HTML, CSS and JavaScript.

- **Scroll-driven 3D hero**: a wall of posters tilts, rotates and moves back in 3D space as you scroll.
- The search panel rises up out of the 3D floor. Recommendation cards fly in along the Z-axis as they scroll into view.
- Cards and the selected movie's poster tilt in 3D as your mouse moves over them.
- Autocomplete search (also works with ↑ ↓ Enter), mode chips, similarity bars, and click-a-card to keep exploring.
- Works on phones and respects `prefers-reduced-motion`.

## API

| Endpoint | Description |
|---|---|
| `GET /api/search?q=godfather` | Title autocomplete (substring matches first, then fuzzy matches) |
| `GET /api/recommend?title=Inception&mode=cast&n=12` | Recommendations by title (or pass `index=` to pick an exact movie) |
| `GET /api/popular` | Most-voted movies (used for the hero poster wall) |

## Project structure

```
recommender.py        data loading, TF-IDF features, cosine similarity, CLI
app.py                Flask server + JSON API
static/index.html     3D scroll-animated frontend
tests/                pytest unit tests (synthetic data, run in about 1 second)
```

## Deploy on Render

The repo includes a `render.yaml`. In Render, choose **New → Blueprint** and select this repo. Or create a **Web Service** by hand with these settings:

| Setting | Value |
|---|---|
| Runtime | Python |
| Build command | `pip install -r requirements.txt && python recommender.py "Avatar" --rebuild -n 1` |
| Start command | `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT` |
| Env var | `PYTHON_VERSION` = `3.11.9` |

You don't need any secrets or API keys. Render sets `PORT` automatically. The model is built in the **build step** because building it uses about 640 MB of memory, while serving it uses only about 200 MB. That fits within the free tier's 512 MB limit.

## Ideas to extend

- Add **collaborative filtering** using `ratings_small.csv` (already in the repo) to build a hybrid recommender.
- Swap TF-IDF for sentence embeddings (for example `sentence-transformers`) to match plots by meaning.

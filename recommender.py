"""
Content-based movie recommender.

Each movie is turned into four text "documents":
    plot     -> overview + tagline
    genre    -> genre names
    cast     -> top 5 actors + director
    keywords -> TMDB plot keywords

Each document set is vectorized with TF-IDF, and similarity between two
movies is the cosine similarity of their vectors. The "all" mode blends the
four similarities with weights.

Usage (CLI):
    python recommender.py "The Dark Knight"
    python recommender.py "Toy Story" --mode genre -n 5
"""

import argparse
import ast
import difflib
import glob
import os
import pickle
import zipfile

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "model", "recommender.pkl")
MIN_VOTES = 10  # drop obscure titles with too little metadata

# How much each feature contributes in the blended "all" mode.
WEIGHTS = {"plot": 0.35, "keywords": 0.25, "cast": 0.2, "genre": 0.2}
MODES = ["all", *WEIGHTS]


# ---------------------------------------------------------------- loading ---

def _read_csv(name):
    """Read `name`.csv directly, or from `name`.zip if only the zip exists."""
    path = os.path.join(BASE_DIR, name)
    if os.path.exists(path + ".csv"):
        return pd.read_csv(path + ".csv", low_memory=False)
    with zipfile.ZipFile(path + ".zip") as zf:
        with zf.open(zf.namelist()[0]) as f:
            return pd.read_csv(f, low_memory=False)


def _read_credits():
    """Credits are split into credits_part*.zip because of GitHub's size limit."""
    if os.path.exists(os.path.join(BASE_DIR, "credits.csv")):
        return _read_csv("credits")
    parts = sorted(glob.glob(os.path.join(BASE_DIR, "credits_part*.zip")))
    return pd.concat(
        [_read_csv(os.path.splitext(os.path.basename(p))[0]) for p in parts],
        ignore_index=True,
    )


def _parse(value):
    """The dataset stores lists of dicts as Python-literal strings."""
    try:
        return ast.literal_eval(value) if isinstance(value, str) else []
    except (ValueError, SyntaxError):
        return []


def _names(value, limit=None):
    return [d["name"] for d in _parse(value)[:limit] if "name" in d]


def _director(crew):
    return [d["name"] for d in _parse(crew) if d.get("job") == "Director"][:1]


def _token(name):
    """'Tom Hanks' -> 'tomhanks' so a full name is one TF-IDF term."""
    return name.replace(" ", "").lower()


def load_movies():
    movies = _read_csv("movies_metadata")
    # A few rows are corrupted (dates in the id column); keep numeric ids only.
    movies = movies[pd.to_numeric(movies["id"], errors="coerce").notna()].copy()
    movies["id"] = movies["id"].astype(int)
    for col in ("vote_count", "vote_average", "popularity"):
        movies[col] = pd.to_numeric(movies[col], errors="coerce").fillna(0)
    movies = movies[movies["vote_count"] >= MIN_VOTES]

    credits = _read_credits()
    keywords = _read_csv("keywords")
    for df in (credits, keywords):
        df["id"] = df["id"].astype(int)

    movies = (
        movies.drop_duplicates("id")
        .merge(credits.drop_duplicates("id"), on="id", how="left")
        .merge(keywords.drop_duplicates("id"), on="id", how="left")
        .reset_index(drop=True)
    )

    movies["genre_list"] = movies["genres"].apply(_names)
    movies["cast_list"] = movies["cast"].apply(lambda c: _names(c, 5))
    movies["director"] = movies["crew"].apply(_director)
    movies["keyword_list"] = movies["keywords"].apply(_names)
    movies["overview"] = movies["overview"].fillna("")
    movies["year"] = movies["release_date"].astype(str).str[:4].where(
        movies["release_date"].notna(), ""
    )

    # Text documents fed to the vectorizers.
    movies["doc_plot"] = movies["overview"] + " " + movies["tagline"].fillna("")
    movies["doc_genre"] = movies["genre_list"].apply(lambda g: " ".join(map(_token, g)))
    movies["doc_cast"] = (movies["cast_list"] + movies["director"] * 2).apply(
        lambda people: " ".join(map(_token, people))  # director counted twice
    )
    movies["doc_keywords"] = movies["keyword_list"].apply(
        lambda k: " ".join(map(_token, k))
    )
    return movies


# ------------------------------------------------------------ recommender ---

class Recommender:
    def __init__(self, movies):
        self.movies = movies
        self.matrices = {
            "plot": TfidfVectorizer(stop_words="english", max_features=50000,
                                    ngram_range=(1, 2), min_df=2)
            .fit_transform(movies["doc_plot"]),
            "genre": TfidfVectorizer().fit_transform(movies["doc_genre"]),
            "cast": TfidfVectorizer().fit_transform(movies["doc_cast"]),
            "keywords": TfidfVectorizer(min_df=2).fit_transform(movies["doc_keywords"]),
        }
        # Keep only what the UI needs so the cached model stays small.
        self.movies = movies[[
            "id", "title", "year", "genre_list", "cast_list", "director", "overview",
            "vote_average", "vote_count", "poster_path",
        ]].copy()
        self._title_lower = self.movies["title"].fillna("").str.lower()

    # ---- persistence -----------------------------------------------------
    @classmethod
    def load(cls, rebuild=False):
        """Load the cached model, building (and caching) it on first run."""
        if not rebuild and os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        model = cls(load_movies())
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(model, f)
        return model

    # ---- lookup ----------------------------------------------------------
    def find(self, title):
        """Index of the best match for `title` (most-voted on duplicates)."""
        exact = np.flatnonzero(self._title_lower == title.strip().lower())
        if len(exact):
            return int(exact[self.movies["vote_count"].values[exact].argmax()])
        close = self.search(title, limit=1)
        return close[0]["index"] if close else None

    def search(self, query, limit=8):
        """Title autocomplete: substring matches first, then fuzzy matches."""
        q = query.strip().lower()
        if not q:
            return []
        hits = self.movies[self._title_lower.str.contains(q, regex=False)]
        hits = hits.sort_values("vote_count", ascending=False).head(limit)
        idx = list(hits.index)
        if len(idx) < limit:
            for t in difflib.get_close_matches(q, self._title_lower.tolist(), n=limit, cutoff=0.6):
                i = self.find_exact(t)
                if i not in idx:
                    idx.append(i)
        return [self.card(i) for i in idx[:limit]]

    def find_exact(self, lower_title):
        return int(np.flatnonzero(self._title_lower == lower_title)[0])

    # ---- core ML ---------------------------------------------------------
    def similarity(self, idx, mode="all"):
        """Cosine similarity of movie `idx` against every movie."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        modes = WEIGHTS if mode == "all" else {mode: 1.0}
        scores = np.zeros(len(self.movies))
        for name, weight in modes.items():
            m = self.matrices[name]
            scores += weight * cosine_similarity(m[idx], m).ravel()
        return scores

    def recommend(self, title=None, idx=None, mode="all", n=10):
        if idx is None:
            idx = self.find(title)
        if idx is None:
            return None, []
        scores = self.similarity(idx, mode)
        scores[idx] = -1  # never recommend the movie itself
        # Slight nudge toward well-rated films to break near-ties.
        quality = np.log1p(self.movies["vote_count"].values) / 10
        ranked = np.argsort(-(scores + 0.02 * quality))[:n]
        return self.card(idx), [
            {**self.card(i), "score": round(float(scores[i]), 4)} for i in ranked
        ]

    # ---- presentation ----------------------------------------------------
    def card(self, i):
        row = self.movies.iloc[i]
        poster = row["poster_path"] if isinstance(row["poster_path"], str) else None
        return {
            "index": int(i),
            "id": int(row["id"]),
            "title": row["title"],
            "year": row["year"],
            "genres": row["genre_list"],
            "cast": row["cast_list"][:3],
            "director": (row["director"] or [""])[0],
            "overview": row["overview"],
            "rating": float(row["vote_average"]),
            "votes": int(row["vote_count"]),
            "poster": f"https://image.tmdb.org/t/p/w342{poster}" if poster else None,
        }


def main():
    parser = argparse.ArgumentParser(description="Content-based movie recommender")
    parser.add_argument("title", help="a movie you like")
    parser.add_argument("--mode", choices=MODES, default="all")
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--rebuild", action="store_true", help="ignore cached model")
    args = parser.parse_args()

    model = Recommender.load(rebuild=args.rebuild)
    movie, recs = model.recommend(args.title, mode=args.mode, n=args.n)
    if movie is None:
        print(f"No movie found matching '{args.title}'.")
        return
    print(f"\nBecause you liked: {movie['title']} ({movie['year']})  [mode: {args.mode}]\n")
    for rank, r in enumerate(recs, 1):
        print(f"{rank:>2}. {r['title']} ({r['year']})  "
              f"sim={r['score']:.3f}  ★{r['rating']}  {', '.join(r['genres'][:3])}")


if __name__ == "__main__":
    main()

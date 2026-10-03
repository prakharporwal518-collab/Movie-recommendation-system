"""Flask web app: serves the 3D UI and a small JSON API over the recommender.

    python app.py   ->  http://127.0.0.1:5000
"""

from flask import Flask, jsonify, request, send_from_directory

from recommender import MODES, Recommender

app = Flask(__name__, static_folder="static")
print("Loading recommender (first run builds the model, ~30s)...")
model = Recommender.load()
print(f"Ready: {len(model.movies)} movies.")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/search")
def search():
    return jsonify(model.search(request.args.get("q", ""), limit=8))


@app.get("/api/popular")
def popular():
    top = model.movies[model.movies["poster_path"].notna()].nlargest(30, "vote_count")
    return jsonify([model.card(i) for i in top.index])


@app.get("/api/recommend")
def recommend():
    mode = request.args.get("mode", "all")
    if mode not in MODES:
        return jsonify(error=f"mode must be one of {MODES}"), 400
    n = min(max(request.args.get("n", 12, type=int), 1), 30)
    idx = request.args.get("index", type=int)
    if idx is not None and not 0 <= idx < len(model.movies):
        return jsonify(error="index out of range"), 400
    movie, recs = model.recommend(request.args.get("title", ""), idx=idx, mode=mode, n=n)
    if movie is None:
        return jsonify(error="Movie not found"), 404
    return jsonify(movie=movie, recommendations=recs, mode=mode)


if __name__ == "__main__":
    app.run(debug=False)

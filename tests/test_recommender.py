import pandas as pd
import pytest

from recommender import Recommender


def make(id_, title, overview, genres, cast, director, keywords, votes=100):
    return dict(id=id_, title=title, year="2000", overview=overview, poster_path=None,
                genre_list=genres, cast_list=cast, director=[director], keyword_list=keywords,
                vote_average=7.0, vote_count=votes,
                doc_plot=overview, doc_genre=" ".join(g.lower() for g in genres),
                doc_cast=" ".join(p.replace(" ", "").lower() for p in cast + [director] * 2),
                doc_keywords=" ".join(k.replace(" ", "").lower() for k in keywords))


@pytest.fixture(scope="module")
def model():
    movies = pd.DataFrame([
        make(1, "Space Quest", "astronauts travel through a wormhole to save humanity",
             ["Science Fiction"], ["Ann Lee"], "Joe Bloggs", ["space", "wormhole"]),
        make(2, "Space Quest II", "astronauts return through the wormhole to a new galaxy",
             ["Science Fiction"], ["Ann Lee"], "Joe Bloggs", ["space", "galaxy"]),
        make(3, "Love in Paris", "two strangers fall in love in paris",
             ["Romance"], ["Bob Ray"], "Kim Park", ["paris", "love"]),
        make(4, "Paris Hearts", "a romance blooms between strangers in paris",
             ["Romance"], ["Cara Dee"], "Kim Park", ["paris", "romance"]),
    ])
    return Recommender(movies)


def test_most_similar_is_the_sequel(model):
    movie, recs = model.recommend("space quest", n=3)
    assert movie["title"] == "Space Quest"
    assert recs[0]["title"] == "Space Quest II"
    assert all(r["title"] != "Space Quest" for r in recs)


@pytest.mark.parametrize("mode", ["plot", "genre", "cast", "keywords"])
def test_each_mode_finds_romance_match(model, mode):
    _, recs = model.recommend("Love in Paris", mode=mode, n=1)
    assert recs[0]["title"] == "Paris Hearts"


def test_scores_are_sorted_and_bounded(model):
    _, recs = model.recommend("Space Quest", n=3)
    scores = [r["score"] for r in recs]
    assert all(0 <= s <= 1.0001 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_search_and_unknown_title(model):
    assert model.search("paris")[0]["title"] in {"Love in Paris", "Paris Hearts"}
    assert model.search("") == []
    assert model.recommend("zzzz qqqq xxxx") == (None, [])


def test_invalid_mode(model):
    with pytest.raises(ValueError):
        model.similarity(0, mode="bogus")

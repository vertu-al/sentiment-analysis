from __future__ import annotations

import nltk
import pandas as pd
from nltk.corpus import movie_reviews


def ensure_movie_reviews_available() -> None:
    """Download the NLTK movie review corpus if it is unavailable."""
    try:
        nltk.data.find("corpora/movie_reviews")
    except LookupError:
        nltk.download("movie_reviews")


def load_movie_reviews() -> pd.DataFrame:
    """
    Load the NLTK Movie Reviews corpus into a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with the columns:
        - review
        - sentiment
    """
    ensure_movie_reviews_available()

    documents: list[dict[str, str]] = []

    for sentiment in movie_reviews.categories():
        for file_id in movie_reviews.fileids(sentiment):
            documents.append(
                {
                    "review": movie_reviews.raw(file_id),
                    "sentiment": sentiment,
                }
            )

    df = pd.DataFrame(documents)

    if df.empty:
        raise RuntimeError("The movie review corpus could not be loaded.")

    expected_columns = {"review", "sentiment"}

    if set(df.columns) != expected_columns:
        raise RuntimeError(
            f"Unexpected dataset columns: {list(df.columns)}"
        )

    return df
from src.data_loader import load_movie_reviews


def test_dataframe_exists():
    df = load_movie_reviews()

    assert df is not None


def test_dataframe_size():
    df = load_movie_reviews()

    assert len(df) == 2000


def test_columns():
    df = load_movie_reviews()

    assert "review" in df.columns
    assert "sentiment" in df.columns


def test_balanced_classes():
    df = load_movie_reviews()

    counts = df["sentiment"].value_counts()

    assert counts["pos"] == 1000
    assert counts["neg"] == 1000
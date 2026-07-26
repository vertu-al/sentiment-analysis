import pytest

from src.predict import predict_sentiment


def test_positive_prediction():
    prediction = predict_sentiment(
        "The movie was absolutely wonderful."
    )

    assert prediction in ["Positive", "Negative"]


def test_negative_prediction():
    prediction = predict_sentiment(
        "The movie was horrible."
    )

    assert prediction in ["Positive", "Negative"]


def test_empty_review():
    with pytest.raises(ValueError):
        predict_sentiment("")
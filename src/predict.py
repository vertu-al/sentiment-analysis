from __future__ import annotations

from pathlib import Path

import joblib

from src.preprocessing import preprocess_text

MODEL_PATH = Path("models/sentiment_model.joblib")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")


def load_artifacts():
    """
    Load the trained model and TF-IDF vectorizer.

    Returns
    -------
    tuple
        Trained classifier and vectorizer.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run `python -m src.train` first."
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer file not found: {VECTORIZER_PATH}. "
            "Run `python -m src.train` first."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def predict_sentiment(review: str) -> str:
    """
    Predict the sentiment of a movie review.

    Parameters
    ----------
    review:
        Raw movie review.

    Returns
    -------
    str
        Positive or Negative.
    """
    if not isinstance(review, str):
        raise TypeError("review must be a string")

    if not review.strip():
        raise ValueError("review must not be empty")

    model, vectorizer = load_artifacts()

    processed_review = preprocess_text(review)
    review_vector = vectorizer.transform([processed_review])
    prediction = model.predict(review_vector)[0]

    return "Positive" if prediction == "pos" else "Negative"


def main() -> None:
    """Run an interactive prediction session."""
    print("Movie Review Sentiment Analysis")
    print("Press Ctrl+C to exit.")
    print()

    while True:
        try:
            review = input("Enter a movie review:\n> ")
            prediction = predict_sentiment(review)

            print(f"\nPredicted sentiment: {prediction}\n")

        except ValueError as error:
            print(f"\nError: {error}\n")

        except KeyboardInterrupt:
            print("\n\nExiting.")
            break


if __name__ == "__main__":
    main()
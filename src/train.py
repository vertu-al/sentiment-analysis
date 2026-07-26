from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_loader import load_movie_reviews
from src.preprocessing import preprocess_text

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"


def train_model() -> tuple[
    LogisticRegression,
    TfidfVectorizer,
    float,
]:
    """
    Train a TF-IDF and Logistic Regression sentiment classifier.

    Returns
    -------
    tuple
        Trained model, vectorizer, and test accuracy.
    """
    df = load_movie_reviews()

    df["processed_review"] = df["review"].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_review"],
        df["sentiment"],
        test_size=0.20,
        random_state=42,
        stratify=df["sentiment"],
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train_tfidf, y_train)

    accuracy = model.score(X_test_tfidf, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, vectorizer, accuracy


if __name__ == "__main__":
    _, _, accuracy = train_model()
    print("Model trained successfully.")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
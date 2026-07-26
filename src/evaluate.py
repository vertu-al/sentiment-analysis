from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.data_loader import load_movie_reviews
from src.preprocessing import preprocess_text

MODEL_PATH = Path("models/sentiment_model.joblib")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")
FIGURE_DIR = Path("figures")
CONFUSION_MATRIX_PATH = FIGURE_DIR / "confusion_matrix.png"


def load_artifacts():
    """Load the trained model and TF-IDF vectorizer."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run `python -m src.train` first."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def evaluate_model() -> None:
    """Evaluate the trained model and save a confusion matrix figure."""
    df = load_movie_reviews()
    df["processed_review"] = df["review"].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_review"],
        df["sentiment"],
        test_size=0.20,
        random_state=42,
        stratify=df["sentiment"],
    )

    model, vectorizer = load_artifacts()

    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
        )
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train_tfidf, y_train)
    baseline_predictions = baseline.predict(X_test_tfidf)

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    print(f"Baseline accuracy: {baseline_accuracy:.4f}")

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=["neg", "pos"],
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Negative", "Positive"],
    )

    display.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
    )
    plt.close()

    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    evaluate_model()
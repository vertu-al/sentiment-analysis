import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if they are unavailable."""
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for resource_path, download_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name)


ensure_nltk_resources()

STOP_WORDS = set(stopwords.words("english")) - {"no", "nor", "not"}
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Clean and normalize English text for sentiment classification.

    Parameters
    ----------
    text:
        Raw input text.

    Returns
    -------
    str
        Preprocessed text.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)

    processed_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS
    ]

    return " ".join(processed_tokens)
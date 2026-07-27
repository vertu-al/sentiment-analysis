# Sentiment Analysis Using Classical Natural Language Processing

This repository contains the implementation and evaluation of a classical Natural Language Processing pipeline for binary movie-review sentiment classification.

A Python-based Natural Language Processing system that classifies movie reviews as **positive** or **negative** using TF–IDF feature extraction and Logistic Regression.

The project implements a complete supervised machine-learning workflow, including data loading, text preprocessing, model training, evaluation, interactive prediction, automated testing, and software quality assurance.

---

## Project Overview

The objective of the project is to develop and evaluate a binary sentiment-classification system for movie reviews.

The system follows this workflow:

- Load the NLTK Movie Reviews dataset
- Clean and preprocess the review text
- Convert the text into numerical TF–IDF features
- Train a Logistic Regression classifier
- Evaluate the model on a separate test set
- Classify new movie reviews as positive or negative

The implementation uses classical NLP and machine-learning methods that are computationally efficient, interpretable, and suitable for binary text-classification tasks.

---

## Features

- Movie-review dataset loading with NLTK
- Text lowercasing and tokenisation
- Punctuation removal
- Stop-word removal
- Preservation of negation terms
- Lemmatisation
- TF–IDF vectorisation
- Logistic Regression classification
- Stratified train–test split
- Automated model evaluation
- Confusion-matrix generation
- Baseline comparison
- Interactive command-line prediction
- Unit testing with pytest
- Static type checking with mypy
- Code-quality checks with Ruff

---

## Project Structure

```text
sentiment-analysis/
│
├── .github/
│   └── workflows/
│       └── python.yml
│
├── data/
│   ├── processed/
│   └── raw/
│
├── figures/
│   └── confusion_matrix.png
│
├── models/
│   ├── sentiment_model.joblib
│   └── tfidf_vectorizer.joblib
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── report/
│   └── project_report.pdf
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_predict.py
│   └── test_preprocessing.py
│
├── .gitignore
├── mypy.ini
├── pytest.ini
├── README.md
└── requirements.txt
```

The trained model files are generated locally by running the training module. Depending on the repository configuration, they may be excluded from version control.

---

## NLP Pipeline

```text
Raw Movie Review
        │
        ▼
Text Preprocessing
- Lowercasing
- Tokenisation
- Punctuation Removal
- Stop-word Removal
- Lemmatisation
        │
        ▼
TF–IDF Feature Extraction
        │
        ▼
Logistic Regression Classifier
        │
        ▼
Sentiment Prediction
Positive / Negative
```

---

## Dataset

The project uses the **NLTK Movie Reviews corpus**, which contains:

- 2,000 movie reviews
- 1,000 positive reviews
- 1,000 negative reviews

The dataset is divided into:

- 80% training data
- 20% test data

A stratified train–test split is used to preserve the balanced class distribution.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vertu-al/sentiment-analysis.git
cd sentiment-analysis
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python -m src.train
```

This command:

- loads and preprocesses the dataset;
- trains the TF–IDF vectoriser;
- trains the Logistic Regression classifier;
- evaluates test accuracy;
- saves the model and vectoriser to the `models` directory.

---

### Evaluate the model

```bash
python -m src.evaluate
```

This command outputs:

- accuracy;
- precision;
- recall;
- F1-score;
- baseline accuracy;
- confusion matrix.

The confusion matrix is saved to:

```text
figures/confusion_matrix.png
```

---

### Predict sentiment

```bash
python -m src.predict
```

Example:

```text
Movie Review Sentiment Analysis

Enter a movie review:
> The acting was excellent, but the ending felt slightly rushed.

Predicted sentiment: Positive
```

Press `Ctrl+C` to exit the interactive prediction program.

---

## Testing and Code Quality

Run all tests:

```bash
python -m pytest
```

Current result:

```text
12 passed
```

Run Ruff:

```bash
ruff check .
```

Run mypy:

```bash
mypy src
```

---

## Results

The Logistic Regression classifier achieved an overall accuracy of **83.25%** on a held-out test set containing 400 movie reviews.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.8519 | 0.8050 | 0.8278 | 200 |
| Positive | 0.8152 | 0.8600 | 0.8370 | 200 |
| Macro average | 0.8335 | 0.8325 | 0.8324 | 400 |
| Weighted average | 0.8335 | 0.8325 | 0.8324 | 400 |

### Overall Performance

| Metric | Value |
|---|---:|
| Accuracy | **0.8325** |
| Baseline accuracy | 0.5000 |
| Weighted precision | 0.8335 |
| Weighted recall | 0.8325 |
| Weighted F1-score | 0.8324 |

### Confusion Matrix

```text
[[161, 39],
 [ 28, 172]]
```

This corresponds to:

- 161 correctly classified negative reviews
- 172 correctly classified positive reviews
- 39 negative reviews incorrectly classified as positive
- 28 positive reviews incorrectly classified as negative

The trained classifier substantially outperformed the 50% majority-class baseline.

---

## Technologies

- Python 3.12
- NLTK
- scikit-learn
- pandas
- NumPy
- Matplotlib
- joblib
- Jupyter Notebook
- pytest
- mypy
- Ruff
- Git
- GitHub Actions

---

## Limitations

The model uses a unigram TF–IDF representation and therefore has limited ability to capture:

- word order;
- compositional meaning;
- sarcasm and irony;
- mixed sentiment;
- long-distance dependencies;
- complex negation;
- contextual distinctions between topic vocabulary and evaluative vocabulary.

Despite these limitations, the model provides a strong and interpretable baseline for binary movie-review sentiment classification.

---

## Future Improvements

Potential extensions include:

- Support Vector Machine comparison
- Multinomial Naïve Bayes comparison
- N-gram feature experiments
- Cross-validation
- Hyperparameter optimisation
- Error analysis
- Model Interpretability
- Feature-importance analysis
- Streamlit web interface
- Docker deployment
- Transformer-based sentiment classification

---

## Author

**Verena Uyka**

B.Sc. Applied Artificial Intelligence (ongoing)

Interests:

- Natural Language Processing
- Machine Learning
- Software Engineering
- Retrieval-Augmented Generation
- Explainable Artificial Intelligence

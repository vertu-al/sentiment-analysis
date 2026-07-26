# Sentiment Analysis Using Classical Natural Language Processing

A Python-based sentiment analysis system that classifies movie reviews as **positive** or **negative** using classical Natural Language Processing (NLP) techniques and supervised machine learning.

The project was developed as part of an academic software engineering assignment and later extended into a reusable, modular Python package following modern software development practices.

---

## Project Overview

This project demonstrates the complete workflow of a classical NLP pipeline:

- Text preprocessing
- Feature engineering using TF–IDF
- Logistic Regression classification
- Model evaluation
- Interactive sentiment prediction
- Unit testing
- Static type checking
- Code quality analysis

Unlike modern Large Language Models (LLMs), this project focuses on interpretable and computationally efficient machine learning methods that remain widely used in industry.

---

## Features

- Natural Language Processing using NLTK
- Text cleaning and preprocessing
- Stop-word removal
- Lemmatisation
- TF–IDF vectorisation
- Logistic Regression classifier
- Command-line sentiment prediction
- Automated evaluation
- Unit tests with pytest
- Static type checking with mypy
- Code quality checks with Ruff

---

## Project Structure

```text
sentiment-analysis/
│
├── data/
├── figures/
├── models/
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
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
├── requirements.txt
├── README.md
└── LICENSE
```

---

## NLP Pipeline

```text
Raw Movie Review
        │
        ▼
Text Preprocessing
- Lowercasing
- Tokenisation
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
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

macOS/Linux

```bash
source .venv/bin/activate
```

Windows

```powershell
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python -m src.train
```

---

### Evaluate the model

```bash
python -m src.evaluate
```

---

### Predict sentiment

```bash
python -m src.predict
```

Example

```text
Enter a movie review:

The acting was excellent but the ending was disappointing.

Predicted sentiment:

Positive
```

---

## Testing

Run all tests

```bash
python -m pytest
```

Run Ruff

```bash
ruff check .
```

Run mypy

```bash
mypy src
```

---

## Results

| Metric | Value |
|---------|------:|
| Accuracy | **83.25%** |
| Precision | 0.83 |
| Recall | 0.83 |
| F1-score | 0.83 |

The classifier achieved an overall accuracy of **83.25%** on the NLTK Movie Reviews dataset using a TF–IDF representation and a Logistic Regression classifier.

---

## Technologies

- Python 3.12
- NLTK
- scikit-learn
- pandas
- Matplotlib
- pytest
- mypy
- Ruff
- Git

---

## Future Improvements

Planned extensions include:

- Support Vector Machines (SVM)
- Naïve Bayes comparison
- Hyperparameter optimisation
- Cross-validation
- Model persistence improvements
- Streamlit web interface
- Docker deployment
- Continuous Integration with GitHub Actions
- Transformer-based sentiment analysis (BERT)

---

## License

This project is released under the MIT License.

---

## Author

**Verena Uyka**

B.Sc. Applied Artificial Intelligence (ongoing)

Research interests:

- Natural Language Processing
- Machine Learning
- Active Inference
- Retrieval-Augmented Generation (RAG)
- Explainable AI
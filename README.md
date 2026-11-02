# 🎬 ReelFeel — Movie Review Sentiment Analyzer

A small NLP application that predicts whether a movie review expresses **positive** or **negative** sentiment. It uses TF-IDF features and a Logistic Regression classifier, with a Streamlit web interface.

## Live locally

### 1. Clone the repository

```bash
git clone https://github.com/wagishi/Movie_review_Model_Creation.git
cd Movie_review_Model_Creation
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies and launch

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (usually http://localhost:8501) that you can open in your browser. The first run may download the NLTK English stopword corpus.

## How it works

1. Enter or paste a movie review.
2. The app removes English stopwords using the same approach as the training notebook.
3. The saved TF-IDF vectorizer converts the cleaned text into numeric features.
4. The saved Logistic Regression model predicts positive (1) or negative (0).
5. If the estimator supports `predict_proba`, the app displays its class-probability estimates.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application and inference |
| `movie_review_model_creation.py` | Original training workflow |
| `Movie_Review.csv` | Movie review dataset |
| `model.pkl` | Saved Logistic Regression model |
| `scaler.pkl` | Saved TF-IDF vectorizer (historical filename) |
| `requirements.txt` | Runtime dependencies |

## Important notes

- The model is binary: positive or negative. It is not designed to detect neutral sentiment.
- Probability values are model estimates, not guarantees and not necessarily calibrated confidence.
- Sarcasm, mixed sentiment, slang, and out-of-domain reviews can reduce performance.
- The pickle artifacts must come from a trusted source. Loading an untrusted pickle file can execute arbitrary code.
- The app uses the same stopword-removal logic as the notebook. If you retrain the model with different preprocessing or vectorizer settings, update and version the artifacts together.

## Deploy

For Streamlit Community Cloud, select this repository, set the main file path to `app.py`, and use `requirements.txt` for dependencies. Ensure `model.pkl` and `scaler.pkl` are included in the repository or otherwise made available to the app.

## Model

- **Vectorization:** TF-IDF (`max_features=2500` in the training notebook)
- **Classifier:** Logistic Regression
- **Output classes:** Positive / Negative

# 🎬 ReelFeel — Movie Review Sentiment Analyzer

<p align="center">
  <strong>An NLP-powered web app that classifies movie reviews as Positive or Negative.</strong>
  <br />
  Built with Python, Scikit-learn, TF-IDF, Logistic Regression, and Streamlit.
  <br /><br />
  <a href="https://reelfeel-movie-reviews.streamlit.app/"><strong>🚀 Try the Live Demo</strong></a>
  &nbsp; · &nbsp;
  <a href="https://github.com/wagishi/Movie_review_Model_Creation"><strong>View Source Code</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn&logoColor=white" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/NLP-TF--IDF-purple" alt="NLP TF-IDF" />
  <img src="https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
</p>

---

## 🌐 Live Application

**[Open ReelFeel — Movie Review Sentiment Analyzer](https://reelfeel-movie-reviews.streamlit.app/)**

Paste a movie review or choose a sample review to see the model's sentiment prediction. The application is deployed using Streamlit Community Cloud.

## 📌 Project Overview

ReelFeel is an end-to-end machine learning project focused on **Natural Language Processing (NLP)** and binary sentiment classification. It takes a written movie review, converts the text into numerical features using TF-IDF, and uses a trained Logistic Regression classifier to predict whether the review expresses positive or negative sentiment.

The project covers the core workflow from text preprocessing and feature extraction to model training, serialized model artifacts, and a deployed interactive application.

## ✨ Key Features

- **Live sentiment analysis:** Classifies a review as Positive or Negative.
- **Interactive Streamlit interface:** Clean interface with sample reviews to try.
- **Consistent text preprocessing:** Removes English stopwords using the same approach as the training workflow.
- **TF-IDF feature extraction:** Converts review text into numerical features using the saved vectorizer.
- **Prediction probability estimates:** Displays positive and negative class estimates when supported by the model.
- **Input validation and error handling:** Handles empty, very short, stopword-only, and oversized reviews.
- **Efficient inference:** Loads model artifacts with Streamlit resource caching.

## 🧠 Machine Learning Workflow

```text
Movie Review Dataset
        │
        ▼
Data Cleaning
(Remove missing records)
        │
        ▼
Text Preprocessing
(English stopword removal)
        │
        ▼
TF-IDF Vectorization
(max_features = 2500)
        │
        ▼
Train / Test Split
(80% / 20%)
        │
        ▼
Logistic Regression
        │
        ▼
Saved Model + Vectorizer
        │
        ▼
Streamlit Web Application
        │
        ▼
Positive / Negative Prediction
```

### Model details

| Component | Implementation |
|---|---|
| Problem type | Binary sentiment classification |
| Language processing | NLTK stopword removal |
| Feature extraction | TF-IDF |
| TF-IDF vocabulary limit | 2,500 features |
| Classification algorithm | Logistic Regression |
| Training split | 80% training / 20% testing |
| Web framework | Streamlit |
| Model persistence | Python pickle |

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| Python | Core programming language |
| Pandas | Dataset loading and manipulation |
| NLTK | Text preprocessing |
| Scikit-learn | TF-IDF vectorization, model training and inference |
| Logistic Regression | Sentiment classification |
| Streamlit | Interactive web application |
| Pickle | Saving and loading trained artifacts |

## 🖥️ Run Locally

Follow these steps to run ReelFeel on your machine.

### 1. Clone the repository

```bash
git clone https://github.com/wagishi/Movie_review_Model_Creation.git
cd Movie_review_Model_Creation
```

### 2. Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (typically **http://localhost:8501**). On the first run, the application may download the NLTK English stopwords corpus.

## 📁 Repository Structure

```text
Movie_review_Model_Creation/
│
├── app.py                          # Streamlit application and prediction pipeline
├── movie_review_model_creation.py  # Model training and preprocessing workflow
├── Movie_Review.csv                # Movie review dataset
├── model.pkl                       # Trained Logistic Regression model
├── scaler.pkl                      # Saved TF-IDF vectorizer
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

> **Note:** `scaler.pkl` is the historical filename used for the TF-IDF vectorizer. It is not a feature scaler.

## 🔍 Example Usage

1. Open the [live application](https://reelfeel-movie-reviews.streamlit.app/).
2. Select a sample review or enter your own.
3. Click **Analyze sentiment**.
4. View the predicted sentiment and the model's class-probability estimates, when available.

The classifier supports two output classes: **Positive** and **Negative**.

## ⚠️ Limitations

- This is a binary classifier; it does not predict a neutral class.
- Sarcasm, irony, mixed opinions, slang, and unfamiliar vocabulary can affect predictions.
- Displayed probability values are model estimates and are **not necessarily calibrated confidence scores**.
- The training script currently uses a random train/test split without a fixed random seed. Repeated training runs may therefore produce different splits.
- The repository does not currently report a reproducible held-out accuracy or other evaluation metric; no performance score is claimed here.
- The deployed application depends on compatible versions of the saved Scikit-learn artifacts and runtime libraries.

## 🚀 Potential Improvements

- Add a reproducible evaluation pipeline with accuracy, precision, recall, F1-score, and a confusion matrix.
- Use stratified splitting and a fixed random seed for repeatable experiments.
- Compare Logistic Regression with models such as Linear SVM and Naive Bayes.
- Explore n-grams, hyperparameter tuning, and more advanced text representations.
- Add a neutral class and improve handling of sarcasm and mixed sentiment.
- Track model versions and automate testing before deployment.

## 👨‍💻 Author

**Wagishi Jagat**  
Electrical Engineering Undergraduate | NIT Patna

- **GitHub:** [@wagishi](https://github.com/wagishi)
- **LinkedIn:** [Wagishi Jagat](https://www.linkedin.com/in/wagishi-jagat-216498333/)

---

<p align="center">
  If you find this project interesting, feel free to explore the code and try the live demo.
  <br />
  <strong><a href="https://reelfeel-movie-reviews.streamlit.app/">Try ReelFeel 🚀</a></strong>
</p>

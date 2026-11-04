from __future__ import annotations

import pickle
from pathlib import Path

import nltk
import streamlit as st
from nltk.corpus import stopwords

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "scaler.pkl"
MAX_REVIEW_LENGTH = 5_000

st.set_page_config(
    page_title="ReelFeel | Movie Review Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(145deg, #101827 0%, #182338 55%, #241d36 100%); }
      [data-testid="stHeader"] { background: transparent; }
      .hero { padding: 1.4rem 0 0.6rem; }
      .eyebrow { color: #a5b4fc; letter-spacing: .18em; font-size: .76rem; font-weight: 700; }
      .hero h1 { color: #f8fafc; font-size: clamp(2.2rem, 7vw, 3.6rem); margin: .3rem 0; }
      .hero p { color: #cbd5e1; font-size: 1.05rem; }
      .panel { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
               border-radius: 18px; padding: 1rem 1.2rem; }
      .result { border-radius: 16px; padding: 1.1rem 1.25rem; margin: .7rem 0; }
      .positive { background: rgba(16,185,129,.13); border: 1px solid rgba(52,211,153,.45); }
      .negative { background: rgba(248,113,113,.13); border: 1px solid rgba(248,113,113,.45); }
      .muted { color: #cbd5e1; font-size: .9rem; }
      div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 10px; min-height: 2.8rem; font-weight: 700;
      }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_artifacts():
    """Load the trusted, repository-provided scikit-learn artifacts once."""
    missing = [path.name for path in (MODEL_PATH, VECTORIZER_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model artifact(s): " + ", ".join(missing))

    # Pickle files can execute code when loaded. Only load artifacts from a
    # trusted source; never accept user-uploaded pickle files.
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

    return model, vectorizer, stop_words


def clean_review(review: str, stop_words: set[str]) -> str:
    """Keep preprocessing consistent with the original training notebook."""
    return " ".join(word for word in review.split() if word.lower() not in stop_words)


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">NLP · MACHINE LEARNING · SENTIMENT ANALYSIS</div>
      <h1>ReelFeel 🎬</h1>
      <p>Does a movie review love the film or leave it disappointed?
      Paste a review and let the model estimate its sentiment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, vectorizer, stop_words = load_artifacts()
except Exception as exc:
    st.error("The model could not be loaded. Check that model.pkl and scaler.pkl are present and that dependencies are installed.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Analyze a review")
examples = {
    "Write or paste your own": "",
    "Positive example": "A beautifully made film with brilliant performances, a moving story, and a soundtrack that stays with you.",
    "Negative example": "The story was painfully slow, the characters felt flat, and the ending was a complete disappointment.",
}
choice = st.selectbox("Try an example", list(examples.keys()))
default_text = examples[choice]

with st.form("review_form"):
    review = st.text_area(
        "Your movie review",
        value=default_text,
        height=170,
        max_chars=MAX_REVIEW_LENGTH,
        placeholder="For example: The performances were excellent, but the plot was predictable...",
        help=f"Maximum {MAX_REVIEW_LENGTH:,} characters.",
    )
    submitted = st.form_submit_button("Analyze sentiment", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    normalized_review = review.strip()
    if not normalized_review:
        st.warning("Please enter a review before analyzing.")
    elif len(normalized_review) < 3:
        st.warning("Please enter a little more text so the model has something to analyze.")
    else:
        cleaned_review = clean_review(normalized_review, stop_words)
        if not cleaned_review.strip():
            st.warning("This review contains only common stopwords. Please add more descriptive words.")
        else:
            try:
                features = vectorizer.transform([cleaned_review])
                prediction = int(model.predict(features)[0])
                probabilities = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
                label = "Positive" if prediction == 1 else "Negative"
                emoji = "😊" if prediction == 1 else "😞"
                css_class = "positive" if prediction == 1 else "negative"

                st.markdown("---")
                st.subheader("Prediction")
                st.markdown(
                    f'<div class="result {css_class}"><h2>{emoji} {label} review</h2>'
                    f'<p class="muted">The model classified the sentiment as {label.lower()}.</p></div>',
                    unsafe_allow_html=True,
                )

                left, right = st.columns(2)
                left.metric("Words analyzed", len(normalized_review.split()))
                right.metric("Characters", len(normalized_review))

                if probabilities is not None:
                    class_to_probability = {
                        int(class_label): float(probability)
                        for class_label, probability in zip(model.classes_, probabilities)
                    }
                    positive_probability = class_to_probability.get(1, 0.0)
                    negative_probability = class_to_probability.get(0, 0.0)
                    st.write("Model probability estimates")
                    st.progress(positive_probability, text=f"Positive: {positive_probability:.1%}")
                    st.progress(negative_probability, text=f"Negative: {negative_probability:.1%}")
                    st.caption("These are model estimates, not guarantees or calibrated measures of how correct the prediction is.")

                with st.expander("How this prediction is made"):
                    st.write(
                        "The review is cleaned using the same stopword-removal approach as the training notebook, "
                        "converted into TF-IDF features, and passed to a Logistic Regression classifier."
                    )
            except Exception as exc:
                st.error("Prediction failed. Please check the saved model and vectorizer versions.")
                with st.expander("Technical details"):
                    st.code(str(exc))

st.markdown("---")
st.caption("ReelFeel · A learning project using TF-IDF and Logistic Regression. Predictions may be wrong, especially for sarcasm, mixed opinions, or reviews unlike the training data.")

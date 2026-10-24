import streamlit as st
import joblib
from pathlib import Path
import torch
from transformers import pipeline

# --- Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detector")
st.markdown("""
This application uses two Machine Learning models to detect if a news article is **Real** or **Fake**.
*   **Baseline:** TF-IDF + Logistic Regression (Fast, traditional ML)
*   **BERT:** Fine-tuned DistilBERT (Deep Learning, context-aware)
""")

# --- Load Models (Cached so it only loads once) ---
@st.cache_resource
def load_baseline():
    base_dir = Path("models/baseline")
    tfidf = joblib.load(base_dir / "tfidf.pkl")
    model = joblib.load(base_dir / "logreg.pkl")
    return tfidf, model

@st.cache_resource
def load_bert():
    bert_dir = str(Path("models/bert/final").resolve())
    # device=-1 forces CPU
    nlp = pipeline("text-classification", model=bert_dir, tokenizer=bert_dir, device=-1) 
    return nlp

try:
    tfidf, base_model = load_baseline()
    bert_model = load_bert()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models. Make sure your models are saved in the correct folders. Details: {e}")
    models_loaded = False

# --- UI Layout ---
text_input = st.text_area("Paste the news article here:", height=250, placeholder="Enter title and body text...")

if st.button("Analyze Article", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    elif models_loaded:
        with st.spinner("Analyzing..."):
            
            # 1. Baseline Prediction
            text_vec = tfidf.transform([text_input])
            base_pred = base_model.predict(text_vec)[0]
            base_proba = base_model.predict_proba(text_vec)[0]
            base_conf = base_proba[base_pred] * 100
            
            # 2. BERT Prediction
            bert_result = bert_model(text_input[:2000])[0] # Truncate to 2000 chars for safety
            # The pipeline returns LABEL_0 (Real) or LABEL_1 (Fake)
            bert_pred = 1 if bert_result['label'] == 'LABEL_1' else 0
            bert_conf = bert_result['score'] * 100

            # --- Display Results ---
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Traditional ML (Baseline)")
                if base_pred == 1:
                    st.error(f"🚨 **FAKE NEWS** (Confidence: {base_conf:.2f}%)")
                else:
                    st.success(f"✅ **REAL NEWS** (Confidence: {base_conf:.2f}%)")

            with col2:
                st.subheader("Deep Learning (DistilBERT)")
                if bert_pred == 1:
                    st.error(f"🚨 **FAKE NEWS** (Confidence: {bert_conf:.2f}%)")
                else:
                    st.success(f"✅ **REAL NEWS** (Confidence: {bert_conf:.2f}%)")
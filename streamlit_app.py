import streamlit as st
import joblib
import torch
import gdown
import zipfile
import os
from google import genai
from transformers import pipeline

# --- Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detector")
st.markdown("""
This application uses **three** models to detect if a news article is **Real** or **Fake**.
*   **Baseline:** TF-IDF + Logistic Regression (Fast, keyword-based)
*   **BERT:** Fine-tuned DistilBERT (Deep Learning, context-aware)
*   **Gemini AI:** Google's Large Language Model (Real-world knowledge)
""")

# --- Load Models ---
@st.cache_resource
def load_baseline():
    tfidf = joblib.load("tfidf.pkl")
    model = joblib.load("logreg.pkl")
    return tfidf, model

@st.cache_resource
def load_bert():
    file_id = "1cWQlcmudcCov59qN6u8CiIEOGjMYI45Y" 
    download_path = "bert_model.zip"
    
    if not os.path.exists("./final"):
        url = f'https://drive.google.com/uc?export=download&confirm=t&id={file_id}'
        gdown.download(url, download_path, quiet=False)
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(".")
            
    nlp = pipeline("text-classification", model="./final", tokenizer="./final", device=-1) 
    return nlp

try:
    tfidf, base_model = load_baseline()
    bert_model = load_bert()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading local models. Details: {e}")
    models_loaded = False

# --- UI Layout ---
text_input = st.text_area("Paste the news article here:", height=200, placeholder="Enter title and body text...")

if st.button("Analyze Article", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    elif models_loaded:
        with st.spinner("Analyzing with all 3 models..."):
            
            # 1. Baseline Prediction
            text_vec = tfidf.transform([text_input])
            base_pred = base_model.predict(text_vec)[0]
            base_proba = base_model.predict_proba(text_vec)[0]
            base_conf = base_proba[base_pred] * 100
            
            # 2. BERT Prediction
            bert_result = bert_model(text_input[:2000])[0] 
            bert_pred = 1 if bert_result['label'] == 'LABEL_1' else 0
            bert_conf = bert_result['score'] * 100

            # 3. Gemini API Prediction (New SDK)
            try:
                # The new SDK uses this syntax:
                client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                prompt = f"Analyze the following news article. Is it REAL news or FAKE news? Reply strictly with only the word 'REAL' or 'FAKE' and nothing else. Article: {text_input[:5000]}"
                response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                gemini_pred = response.text.strip().upper()
                gemini_is_fake = True if "FAKE" in gemini_pred else False
            except Exception as e:
                gemini_is_fake = None

            # --- Display Results ---
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Baseline (TF-IDF)")
                if base_pred == 1:
                    st.error(f"🚨 **FAKE NEWS**\nConfidence: {base_conf:.2f}%")
                else:
                    st.success(f"✅ **REAL NEWS**\nConfidence: {base_conf:.2f}%")

            with col2:
                st.subheader("Deep Learning (BERT)")
                if bert_pred == 1:
                    st.error(f"🚨 **FAKE NEWS**\nConfidence: {bert_conf:.2f}%")
                else:
                    st.success(f"✅ **REAL NEWS**\nConfidence: {bert_conf:.2f}%")

            with col3:
                st.subheader("Google Gemini AI")
                if gemini_is_fake is True:
                    st.error("🚨 **FAKE NEWS**\n(Based on world knowledge)")
                elif gemini_is_fake is False:
                    st.success("✅ **REAL NEWS**\n(Based on world knowledge)")
                else:
                    st.warning("API Key missing or invalid.")

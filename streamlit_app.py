import streamlit as st
import joblib
import torch
import gdown
import zipfile
import os
from openai import OpenAI
from transformers import pipeline

# --- Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detector")
st.markdown("""
This application uses **three** models to detect if a news article is **Real** or **Fake**.
*   **Baseline:** TF-IDF + Logistic Regression
*   **BERT:** Fine-tuned DistilBERT (Note: Trained on 2016 US politics, may struggle with modern/international news)
*   **OpenRouter (LLM):** Modern LLM with real-world knowledge
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

            # 3. OpenRouter Prediction
            try:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=st.secrets["OPENROUTER_API_KEY"],
                )
                prompt = f"Analyze the following news article. Is it REAL news or FAKE news? Reply strictly with only the word 'REAL' or 'FAKE' and nothing else. Article: {text_input[:5000]}"
                
                response = client.chat.completions.create(
                    model="nvidia/nemotron-3.5-lightning:free", 
                    messages=[{"role": "user", "content": prompt}],
                )
                llm_pred = response.choices[0].message.content.strip().upper()
                llm_is_fake = True if "FAKE" in llm_pred else False
                llm_error = None
            except Exception as e:
                llm_is_fake = None
                llm_error = str(e)

            # --- ENSEMBLE VOTING (NEW) ---
            # We have 3 models. If at least 2 say FAKE, final is FAKE.
            votes_for_fake = 0
            if base_pred == 1: votes_for_fake += 1
            if bert_pred == 1: votes_for_fake += 1
            if llm_is_fake is True: votes_for_fake += 1

            st.divider()
            
            # Display the Overall Verdict
            st.subheader("🏆 Final Ensemble Verdict")
            if votes_for_fake >= 2:
                st.error("🚨 **OVERALL VERDICT: FAKE NEWS** (Majority Vote)")
            else:
                st.success("✅ **OVERALL VERDICT: REAL NEWS** (Majority Vote)")
            
            st.write("---")
            
            # Display Individual Model Results
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
                st.subheader("OpenRouter (LLM)")
                if llm_is_fake is True:
                    st.error("🚨 **FAKE NEWS**\n(Based on world knowledge)")
                elif llm_is_fake is False:
                    st.success("✅ **REAL NEWS**\n(Based on world knowledge)")
                elif llm_error:
                    st.error(f"API Error: {llm_error}")
                else:
                    st.warning("Unknown error.")

import streamlit as st
import joblib
import os
import json
from openai import OpenAI
from huggingface_hub import InferenceClient

# --- Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detector & Cyber Threat Analyzer")
st.markdown("""
This application uses **three** models to detect if a news article is **Real** or **Fake**. 
*   **Baseline:** TF-IDF + Logistic Regression (Fast, keyword-based)
*   **Deep Learning:** Modern DeBERTa-v3 Transformer (API)
*   **OpenRouter (LLM):** Modern LLM (Real-world knowledge)
""")

# --- Load Local Baseline Model ---
@st.cache_resource
def load_baseline():
    tfidf = joblib.load("tfidf.pkl")
    model = joblib.load("logreg.pkl")
    return tfidf, model

try:
    tfidf, base_model = load_baseline()
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
        with st.spinner("Analyzing with all 3 models and generating report..."):
            
            # 1. Baseline Prediction
            text_vec = tfidf.transform([text_input])
            base_pred = base_model.predict(text_vec)[0]
            base_proba = base_model.predict_proba(text_vec)[0]
            base_conf = base_proba[base_pred] * 100

            # 2. Modern Deep Learning Prediction (Hugging Face API)
            bert_is_fake = None
            bert_conf = 0
            try:
                hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
                # Using a modern zero-shot classification model for logic-based detection
                result = hf_client.zero_shot_classification(
                    text_input[:1000], # Truncate for speed
                    candidate_labels=["fake news", "real news"],
                    model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
                )
                # The API returns a list of dicts sorted by score
                top_prediction = result[0]
                bert_is_fake = True if top_prediction['label'] == "fake news" else False
                bert_conf = top_prediction['score'] * 100
            except Exception as e:
                st.error(f"HuggingFace API Error: {e}")

            # 3. OpenRouter Prediction (JSON Mode for Reasoning & Action)
            llm_is_fake = None
            llm_reason = "No explanation provided."
            llm_action = "No action needed."
            llm_error = None
            
            try:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=st.secrets["OPENROUTER_API_KEY"],
                )
                
                prompt = f"""
                Analyze the following news article. 
                Return ONLY a valid JSON object with exactly these three keys:
                "verdict": "REAL" or "FAKE",
                "reason": "A short 1-2 sentence explanation of why this is real or fake based on tone, logic, and facts.",
                "action": "If FAKE, mention that the user should report this to the Indian Cyber Crime portal (cybercrime.gov.in) or PIB Fact Check. If REAL, say 'No action needed'."
                
                Article: {text_input[:4000]}
                """
                
                response = client.chat.completions.create(
                    model="nvidia/nemotron-3.5-lightning:free", 
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                
                raw_response = response.choices[0].message.content.strip()
                if raw_response.startswith("```json"):
                    raw_response = raw_response.replace("```json", "").replace("```", "").strip()
                
                parsed_json = json.loads(raw_response)
                verdict_str = parsed_json.get("verdict", "").upper()
                llm_is_fake = True if "FAKE" in verdict_str else False
                llm_reason = parsed_json.get("reason", "No reason provided.")
                llm_action = parsed_json.get("action", "No action provided.")
                
            except Exception as e:
                llm_error = str(e)

            # --- ENSEMBLE VOTING ---
            votes_for_fake = 0
            if base_pred == 1: votes_for_fake += 1
            if bert_is_fake is True: votes_for_fake += 1
            if llm_is_fake is True: votes_for_fake += 1

            st.divider()
            
            # --- FINAL VERDICT & ACTIONS ---
            st.subheader("🏆 Final Verdict")
            if votes_for_fake >= 2:
                st.error("🚨 **OVERALL VERDICT: FAKE NEWS** (Majority Vote)")
                st.info(f"**🧠 Reason:** {llm_reason}")
                st.warning(f"**🛡️ Recommended Action:** {llm_action}")
                
                st.markdown("""
                ### 📢 Report This Fake News to Authorities:
                Please take a screenshot of this article and report it immediately to the official Indian Government portals:
                *   🚔 **[National Cyber Crime Reporting Portal](https://cybercrime.gov.in/)** (Select 'Report Cyber Crime' -> 'Fake News')
                *   📰 **[PIB Fact Check](https://pib.gov.in/factcheck.aspx)** (WhatsApp: +91-8799717971)
                *   🐦 **Twitter/X:** Tag `@Cyberdost` (Ministry of Home Affairs)
                """)
            else:
                st.success("✅ **OVERALL VERDICT: REAL NEWS** (Majority Vote)")
                st.info(f"**🧠 Reason:** {llm_reason}")

            st.write("---")
            
            # --- INDIVIDUAL MODEL BREAKDOWN ---
            st.subheader("🔍 Model Breakdown")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Baseline (TF-IDF)**")
                if base_pred == 1:
                    st.error(f"🚨 FAKE\nConfidence: {base_conf:.2f}%")
                else:
                    st.success(f"✅ REAL\nConfidence: {base_conf:.2f}%")

            with col2:
                st.markdown("**Deep Learning (Modern API)**")
                if bert_is_fake is True:
                    st.error(f"🚨 FAKE\nConfidence: {bert_conf:.2f}%")
                elif bert_is_fake is False:
                    st.success(f"✅ REAL\nConfidence: {bert_conf:.2f}%")
                else:
                    st.warning("API Error")

            with col3:
                st.markdown("**OpenRouter (LLM)**")
                if llm_is_fake is True:
                    st.error("🚨 FAKE\n(Gave reasoning)")
                elif llm_is_fake is False:
                    st.success("✅ REAL\n(Gave reasoning)")
                else:
                    st.error("API Error")

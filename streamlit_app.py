import streamlit as st
import joblib
import os
import json
import re
from openai import OpenAI
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS

# --- Page Config ---
st.set_page_config(page_title="Global Fake News Detector", page_icon="🌍", layout="wide")

st.title("🌍 Global Fake News Detector & Threat Analyzer")
st.markdown("""
This app uses **3 AI Models + RAG + Reverse Engineering** to detect if a news article is **Real** or **Fake**. It supports global news sources and detects known satire sites.
""")

# --- GLOBAL TRUSTED SOURCES DATABASE ---
TRUSTED_SOURCES = [
    # Global / International
    'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'aljazeera.com', 'npr.org', 'bloomberg.com',
    # North America
    'nytimes.com', 'washingtonpost.com', 'wsj.com', 'cnn.com', 'foxnews.com', 'nbcnews.com', 'cbsnews.com', 'cbc.ca', 'theglobeandmail.com',
    # Europe
    'theguardian.com', 'telegraph.co.uk', 'independent.co.uk', 'ft.com', 'dw.com', 'france24.com', 'lemonde.fr', 'spiegel.de', 'euronews.com',
    # Asia & India
    'timesofindia.indiatimes.com', 'thehindu.com', 'indianexpress.com', 'ndtv.com', 'hindustantimes.com', 'pib.gov.in',
    'scmp.com', 'straitstimes.com', 'japantimes.co.jp', 'thejakartapost.com',
    # Middle East & Africa
    'haaretz.com', 'jpost.com', 'khaleejtimes.com', 'gulfnews.com', 'news24.com', 'allafrica.com'
]

SATIRE_SOURCES = [
    'theonion.com', 'fakingnews.com', 'worldnewsdailyreport.com', 'thefauxy.com', 
    'thebeaverton.com', 'clickhole.com', 'reductress.com', 'theborowitzreport.com'
]

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
text_input = st.text_area("Paste the news article here (or drop a URL):", height=200, placeholder="Enter title, body text, or URL...")

if st.button("Analyze Article", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    elif models_loaded:
        with st.spinner("Analyzing globally..."):
            
            # --- 0. CLEAN THE TEXT ---
            clean_text = re.sub(r'http\S+', '', text_input)
            clean_text = clean_text.replace('Read more at:', '').replace('..', '.').strip()
            
            # 1. Baseline Prediction
            text_vec = tfidf.transform([clean_text])
            base_pred = base_model.predict(text_vec)[0]
            base_proba = base_model.predict_proba(text_vec)[0]
            base_conf = base_proba[base_pred] * 100

            # 2. Modern Deep Learning (Lightweight, Fast Model)
            bert_is_fake = None
            bert_conf = 0
            bert_status = "success"
            try:
                hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
                # Using a fast RoBERTa model specifically trained for fake news
                result = hf_client.text_classification(
                    clean_text[:1000],
                    model="hamzab/roberta-fake-news-classification"
                )
                top_prediction = result[0]
                # Model returns 'FAKE' or 'TRUE'
                bert_is_fake = True if top_prediction['label'].upper() in ['FAKE', 'LABEL_0'] else False
                bert_conf = top_prediction['score'] * 100
            except Exception:
                bert_status = "offline" # Handle gracefully without a scary yellow bar

            # --- 3. REVERSE ENGINEERING (GLOBAL DATABASE) ---
            re_origin = "Unknown"
            re_trust = "Medium (Unverified)"
            is_satire = False
            
            url_match = re.search(r'https?://[^\s]+', text_input)
            if url_match:
                domain = url_match.group(0).split('/')[2].replace('www.', '')
                re_origin = f"URL Provided: {domain}"
            else:
                try:
                    search_query = clean_text[:80].replace('"', '')
                    exact_search = DDGS().text(f'"{search_query}"', max_results=1)
                    if exact_search:
                        domain = exact_search[0]['href'].split('/')[2].replace('www.', '')
                        re_origin = domain
                    else:
                        re_origin = "No exact match found online"
                        domain = ""
                except:
                    domain = ""

            if domain:
                if any(satire in domain for satire in SATIRE_SOURCES):
                    re_trust = "Low (Known Satire/Fake Site)"
                    is_satire = True
                elif any(trusted in domain for trusted in TRUSTED_SOURCES):
                    re_trust = "High (Verified Global News Source)"
                else:
                    re_trust = "Medium (Unverified/Unknown Source)"

            # 4. OpenRouter Prediction (RAG + LLM)
            llm_is_fake = None
            llm_reason = "No explanation provided."
            llm_action = "No action needed."
            
            try:
                search_results = DDGS().text(clean_text[:150], max_results=3)
                context_text = "\n".join([f"- {r['title']}: {r['body']}" for r in search_results])
                
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=st.secrets["OPENROUTER_API_KEY"],
                )
                
                prompt = f"""
                Analyze the following news article.
                Web Context: {context_text}
                REVERSE ENGINEERING DATA:
                - Traced Origin: '{re_origin}'
                - Trust Level: '{re_trust}'
                
                Rules:
                1. If the Trust Level is 'High', assume the article is REAL unless the text is clearly a joke.
                2. If the Trust Level is 'Low (Known Satire)', it is FAKE.
                3. Rely on your own global knowledge to verify the facts.
                
                Return ONLY a valid JSON object with exactly these keys:
                "verdict": "REAL" or "FAKE",
                "reason": "1-2 sentences explaining why based on facts and the traced origin.",
                "action": "If FAKE, advise reporting to cybercrime.gov.in. If REAL, say 'No action needed'."
                
                Article: {clean_text[:4000]}
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
                llm_is_fake = True if "FAKE" in parsed_json.get("verdict", "").upper() else False
                llm_reason = parsed_json.get("reason", "No reason provided.")
                llm_action = parsed_json.get("action", "No action provided.")
                
            except Exception as e:
                pass # Suppress LLM errors if it fails

            # --- ENSEMBLE VOTING ---
            final_is_fake = False
            if is_satire:
                final_is_fake = True 
            elif "High" in re_trust:
                final_is_fake = False 
            else:
                final_is_fake = llm_is_fake if llm_is_fake is not None else (base_pred == 1)

            st.divider()
            
            # --- FINAL VERDICT ---
            st.subheader("🏆 Final Verdict")
            if final_is_fake:
                st.error("🚨 **OVERALL VERDICT: FAKE NEWS**")
                st.info(f"**🧠 Reason:** {llm_reason}")
                st.warning(f"**🛡️ Recommended Action:** {llm_action}")
                st.markdown("""
                ### 📢 Report This Fake News to Authorities:
                *   🚔 **[National Cyber Crime Reporting Portal](https://cybercrime.gov.in/)**
                *   📰 **[PIB Fact Check](https://pib.gov.in/factcheck.aspx)**
                """)
            else:
                st.success("✅ **OVERALL VERDICT: REAL NEWS**")
                st.info(f"**🧠 Reason:** {llm_reason}")

            st.write("---")
            
            # --- REVERSE ENGINEERING RESULTS ---
            st.subheader("🕵️ Reverse Engineering (Source Tracing)")
            st.markdown(f"**Traced Origin:** `{re_origin}`")
            if "High" in re_trust:
                st.success(f"**Trust Level:** {re_trust}")
            elif "Low" in re_trust:
                st.error(f"**Trust Level:** {re_trust}")
            else:
                st.warning(f"**Trust Level:** {re_trust}")

            st.write("---")
            
            # --- INDIVIDUAL MODEL BREAKDOWN ---
            st.subheader("🔍 Model Breakdown")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Baseline (TF-IDF)**")
                if base_pred == 1: st.error(f"🚨 FAKE ({base_conf:.1f}%)")
                else: st.success(f"✅ REAL ({base_conf:.1f}%)")

            with col2:
                st.markdown("**Deep Learning (RoBERTa)**")
                if bert_status == "offline":
                    st.markdown("⚪ *Model Offline (API Timeout)*") # Clean gray text instead of yellow bar
                elif bert_is_fake is True:
                    st.error(f"🚨 FAKE ({bert_conf:.1f}%)")
                elif bert_is_fake is False:
                    st.success(f"✅ REAL ({bert_conf:.1f}%)")

            with col3:
                st.markdown("**OpenRouter (LLM)**")
                if llm_is_fake is True: st.error("🚨 FAKE")
                elif llm_is_fake is False: st.success("✅ REAL")
                else: st.markdown("⚪ *Model Offline*")

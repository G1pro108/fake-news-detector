import streamlit as st
import joblib
import os
import json
import re
from openai import OpenAI
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS

# --- Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detector & Threat Analyzer")
st.markdown("""
This app uses **3 AI Models + RAG + Reverse Engineering** to detect if a news article is **Real** or **Fake**.
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
        with st.spinner("Analyzing..."):
            
            # --- 0. CLEAN THE TEXT ---
            # Remove URLs and "Read more at" for cleaner analysis
            clean_text = re.sub(r'http\S+', '', text_input)
            clean_text = clean_text.replace('Read more at:', '').replace('..', '.').strip()
            
            # 1. Baseline Prediction
            text_vec = tfidf.transform([clean_text])
            base_pred = base_model.predict(text_vec)[0]
            base_proba = base_model.predict_proba(text_vec)[0]
            base_conf = base_proba[base_pred] * 100

            # 2. Modern Deep Learning (Using a lighter, reliable model)
            bert_is_fake = None
            bert_conf = 0
            try:
                hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
                result = hf_client.zero_shot_classification(
                    clean_text[:1000],
                    candidate_labels=["fake news", "real news"],
                    model="cross-encoder/nli-distilroberta-base" # Lighter model to prevent timeouts
                )
                top_prediction = result[0]
                bert_is_fake = True if top_prediction['label'] == "fake news" else False
                bert_conf = top_prediction['score'] * 100
            except Exception as e:
                st.warning(f"Deep Learning API bypassed due to timeout.")

            # --- 3. REVERSE ENGINEERING (RAE) ---
            re_origin = "Unknown"
            re_trust = "Medium"
            
            # Check if user pasted a URL
            url_match = re.search(r'https?://[^\s]+', text_input)
            if url_match:
                domain = url_match.group(0).split('/')[2].replace('www.', '')
                re_origin = f"URL Provided: {domain}"
                
                # Check against trusted Indian & Global sources
                trusted_sources = ['timesofindia.indiatimes.com', 'thehindu.com', 'indianexpress.com', 
                                   'ndtv.com', 'hindustantimes.com', 'reuters.com', 'bbc.com', 
                                   'pib.gov.in', 'apnews.com']
                
                if any(trusted in domain for trusted in trusted_sources):
                    re_trust = "High (Credible Source)"
                else:
                    re_trust = "Low (Unknown Source)"
            else:
                # If no URL, search the web for the exact text
                try:
                    search_query = clean_text[:80].replace('"', '')
                    exact_search = DDGS().text(f'"{search_query}"', max_results=1)
                    if exact_search:
                        domain = exact_search[0]['href'].split('/')[2].replace('www.', '')
                        re_origin = domain
                        trusted_sources = ['timesofindia.indiatimes.com', 'thehindu.com', 'indianexpress.com', 'ndtv.com', 'hindustantimes.com', 'reuters.com', 'bbc.com']
                        if any(trusted in domain for trusted in trusted_sources):
                            re_trust = "High (Credible Source)"
                        else:
                            re_trust = "Medium (Unknown Source)"
                    else:
                        re_origin = "No exact match found online"
                        re_trust = "Low (Unverified)"
                except:
                    pass

            # 4. OpenRouter Prediction (RAG + LLM)
            llm_is_fake = None
            llm_reason = "No explanation provided."
            llm_action = "No action needed."
            
            try:
                # RAG Step
                search_results = DDGS().text(clean_text[:150], max_results=3)
                context_text = "\n".join([f"- {r['title']}: {r['body']}" for r in search_results])
                
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=st.secrets["OPENROUTER_API_KEY"],
                )
                
                # We add the Reverse Engineering findings directly into the prompt!
                prompt = f"""
                Analyze the following news article based on this live web context.
                Web Context: {context_text}
                
                IMPORTANT: Reverse engineering traced the original source to: '{re_origin}' with a Trust Level of '{re_trust}'.
                If the Trust Level is 'High', you should strongly lean towards REAL, unless the text is obviously satire.
                
                Return ONLY a valid JSON object with exactly these three keys:
                "verdict": "REAL" or "FAKE",
                "reason": "A short 1-2 sentence explanation.",
                "action": "If FAKE, mention reporting to cybercrime.gov.in. If REAL, say 'No action needed'."
                
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
                st.error(f"LLM API Error: {e}")

            # --- ENSEMBLE VOTING (Rule-Based) ---
            # If Reverse Engineering finds a "High Trust" source, the final verdict is REAL.
            # Otherwise, rely on the LLM's verdict.
            final_is_fake = False
            
            if "High" in re_trust:
                final_is_fake = False # Trusted source overrides everything
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
                st.markdown("**Deep Learning (DeBERTa)**")
                if bert_is_fake is True: st.error(f"🚨 FAKE ({bert_conf:.1f}%)")
                elif bert_is_fake is False: st.success(f"✅ REAL ({bert_conf:.1f}%)")
                else: st.warning("API Timeout")

            with col3:
                st.markdown("**OpenRouter (LLM)**")
                if llm_is_fake is True: st.error("🚨 FAKE")
                elif llm_is_fake is False: st.success("✅ REAL")
                else: st.error("API Error")

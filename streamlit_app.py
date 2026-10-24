import streamlit as st
import joblib
import os
import json
from openai import OpenAI
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS

# --- Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detector & Threat Analyzer")
st.markdown("""
This application uses **3 AI Models + RAG + Reverse Engineering** to detect if a news article is **Real** or **Fake**. 
*   **Baseline:** TF-IDF
*   **Deep Learning:** DeBERTa-v3
*   **OpenRouter (LLM + RAG):** Live web fact-check
*   **Reverse Engineering:** Traces the origin of the text online!
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
        with st.spinner("Analyzing with AI, searching live web, and reverse-engineering source..."):
            
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
                result = hf_client.zero_shot_classification(
                    text_input[:1000],
                    candidate_labels=["fake news", "real news"],
                    model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
                )
                top_prediction = result[0]
                bert_is_fake = True if top_prediction['label'] == "fake news" else False
                bert_conf = top_prediction['score'] * 100
            except Exception as e:
                pass

            # --- REVERSE ENGINEERING: TRACE THE SOURCE ---
            re_origin = "Unknown"
            re_trust = "Unknown"
            try:
                # Extract a unique chunk of text to search
                search_query = text_input[:100].replace('"', '')
                # Search for an EXACT phrase match
                exact_search = DDGS().text(f'"{search_query}"', max_results=2)
                
                if exact_search:
                    # If we find a result, we trace the domain
                    first_url = exact_search[0]['href']
                    domain = first_url.split('/')[2].replace('www.', '')
                    re_origin = domain
                    
                    # Check if it's a known credible source
                    credible_domains = ['reuters.com', 'bbc.com', 'nytimes.com', 'thehindu.com', 'indiatoday.in', 'apnews.com']
                    satire_domains = ['theonion.com', 'fakingnews.com', 'worldnewsdailyreport.com', 'thefauxy.com']
                    
                    if any(cred in domain for cred in credible_domains):
                        re_trust = "High (Credible Source)"
                    elif any(sat in domain for sat in satire_domains):
                        re_trust = "Low (Satire/Fake Site)"
                    else:
                        re_trust = "Medium (Unknown Source)"
                else:
                    re_origin = "No exact match found online (Highly Suspicious)"
                    re_trust = "Low (Unverified)"
            except Exception as e:
                re_origin = "Search failed."
                re_trust = "Unknown"

            # 3. OpenRouter Prediction (RAG + LLM)
            llm_is_fake = None
            llm_reason = "No explanation provided."
            llm_action = "No action needed."
            
            try:
                # RAG Step
                search_results = DDGS().text(text_input[:150], max_results=3)
                context_text = "\n".join([f"- {r['title']}: {r['body']}" for r in search_results])
                
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=st.secrets["OPENROUTER_API_KEY"],
                )
                
                prompt = f"""
                Analyze the following news article based on this live web context.
                Context: {context_text}
                Also note: Reverse engineering found the original source to be '{re_origin}' which has a trust level of '{re_trust}'.
                
                Return ONLY a valid JSON object with exactly these three keys:
                "verdict": "REAL" or "FAKE",
                "reason": "A short 1-2 sentence explanation.",
                "action": "If FAKE, mention reporting to cybercrime.gov.in. If REAL, say 'No action needed'."
                
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
                llm_is_fake = True if "FAKE" in parsed_json.get("verdict", "").upper() else False
                llm_reason = parsed_json.get("reason", "No reason provided.")
                llm_action = parsed_json.get("action", "No action provided.")
                
            except Exception as e:
                pass

            # --- ENSEMBLE VOTING ---
            final_is_fake = False
            if llm_is_fake is not None:
                if llm_is_fake is False:
                    final_is_fake = False
                else:
                    if base_pred == 1 or bert_is_fake is True:
                        final_is_fake = True
            
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
            st.markdown(f"**Original Source Traced:** `{re_origin}`")
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
                else: st.warning("API Error")

            with col3:
                st.markdown("**OpenRouter (LLM)**")
                if llm_is_fake is True: st.error("🚨 FAKE")
                elif llm_is_fake is False: st.success("✅ REAL")
                else: st.error("API Error")

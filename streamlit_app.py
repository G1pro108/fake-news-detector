import streamlit as st
import joblib
import os
import json
import re
from openai import OpenAI
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS
from PIL import Image
import pytesseract

# --- Page Config ---
st.set_page_config(page_title="Multimodal Fake News Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ Multimodal Fake News & AI Detector")
st.markdown("""
This system detects **Fake News, AI-Generated Text, and AI-Generated Images**. 
It supports pasting text, links, or uploading cropped screenshots.
""")

# --- GLOBAL TRUSTED SOURCES DATABASE ---
TRUSTED_SOURCES = [
    'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'aljazeera.com', 'npr.org', 'bloomberg.com',
    'nytimes.com', 'washingtonpost.com', 'wsj.com', 'cnn.com', 'foxnews.com', 'nbcnews.com', 'cbsnews.com', 'cbc.ca',
    'theguardian.com', 'telegraph.co.uk', 'independent.co.uk', 'ft.com', 'dw.com', 'france24.com', 'lemonde.fr',
    'timesofindia.indiatimes.com', 'thehindu.com', 'indianexpress.com', 'ndtv.com', 'hindustantimes.com', 'pib.gov.in',
    'scmp.com', 'straitstimes.com', 'japantimes.co.jp', 'haaretz.com', 'jpost.com', 'khaleejtimes.com', 'gulfnews.com'
]
SATIRE_SOURCES = ['theonion.com', 'fakingnews.com', 'worldnewsdailyreport.com', 'thefauxy.com', 'thebeaverton.com']

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

# --- UI Layout: TABS FOR TEXT VS IMAGE ---
tab1, tab2 = st.tabs(["📝 Paste Text / URL", "🖼️ Upload Cropped Image"])

text_input = ""
image_input = None

with tab1:
    text_input = st.text_area("Paste the news article here (or URL):", height=200, placeholder="Enter title, body text, or URL...")

with tab2:
    image_input = st.file_uploader("Upload a screenshot of a news article or tweet (JPG/PNG):", type=["jpg", "jpeg", "png"])

if st.button("Analyze Now", type="primary"):
    if not text_input.strip() and not image_input:
        st.warning("Please paste text or upload an image.")
    elif models_loaded:
        with st.spinner("Running Multimodal AI Analysis..."):
            
            # --- 0. EXTRACT TEXT FROM IMAGE (OCR) ---
            extracted_text = ""
            ai_image_verdict = None
            
            if image_input:
                img = Image.open(image_input)
                # Extract text using Tesseract OCR
                extracted_text = pytesseract.image_to_string(img).strip()
                if extracted_text:
                    st.info(f"**Extracted Text from Image:**\n\n{extracted_text}")
                    text_input = extracted_text # Overwrite text_input for the rest of the pipeline
                else:
                    st.warning("No text found in image. Checking if the image itself is AI-generated...")

                # --- AI IMAGE DETECTION ---
                try:
                    hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
                    # Using a specialized model to detect AI-generated images
                    img_res = hf_client.image_classification(image_input, model="umm-maybe/AI-image-detector")
                    top_img = img_res[0]
                    if top_img['label'].lower() in ['artificial', 'fake', 'ai']:
                        ai_image_verdict = True
                        st.error(f"🚨 **This image is likely AI-Generated** (Confidence: {top_img['score']*100:.1f}%)")
                    else:
                        ai_image_verdict = False
                        st.success(f"✅ **This image looks Real/Photographic** (Confidence: {top_img['score']*100:.1f}%)")
                except Exception as e:
                    pass # Ignore if API fails

            if not text_input.strip() and ai_image_verdict is None:
                st.stop() # Nothing to analyze

            # --- 1. AI TEXT DETECTION ---
            is_ai_text = False
            try:
                hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
                ai_text_res = hf_client.text_classification(text_input[:1000], model="openai-community/roberta-base-openai-detector")
                if ai_text_res[0]['label'].lower() in ['fake', 'chatgpt', 'ai'] and ai_text_res[0]['score'] > 0.6:
                    is_ai_text = True
            except:
                pass

            # --- 2. BASELINE PREDICTION ---
            clean_text = re.sub(r'http\S+', '', text_input).replace('Read more at:', '').strip()
            text_vec = tfidf.transform([clean_text])
            base_pred = base_model.predict(text_vec)[0]
            base_conf = base_model.predict_proba(text_vec)[0][base_pred] * 100

            # --- 3. REVERSE ENGINEERING (RAE) ---
            re_origin = "Unknown"
            re_trust = "Medium (Unverified)"
            domain = ""
            
            url_match = re.search(r'https?://[^\s]+', text_input)
            if url_match:
                domain = url_match.group(0).split('/')[2].replace('www.', '')
                re_origin = f"URL Provided: {domain}"
            else:
                try:
                    exact_search = DDGS().text(f'"{clean_text[:80]}"', max_results=1)
                    if exact_search:
                        domain = exact_search[0]['href'].split('/')[2].replace('www.', '')
                        re_origin = domain
                except: pass

            if domain:
                if any(satire in domain for satire in SATIRE_SOURCES): re_trust = "Low (Known Satire/Fake Site)"
                elif any(trusted in domain for trusted in TRUSTED_SOURCES): re_trust = "High (Verified Global News Source)"

            # --- 4. LLM + RAG PREDICTION ---
            llm_is_fake = None
            llm_reason = "No explanation provided."
            try:
                search_results = DDGS().text(clean_text[:150], max_results=2)
                context_text = "\n".join([f"- {r['title']}: {r['body']}" for r in search_results])
                
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
                prompt = f"""
                Analyze this news. Web Context: {context_text}
                Traced Origin: '{re_origin}' | Trust: '{re_trust}'
                AI Generated Text Flag: {is_ai_text}
                
                Return JSON: "verdict": "REAL" or "FAKE", "reason": "1-2 sentences.", "action": "report or no action"
                Article: {clean_text[:3000]}
                """
                response = client.chat.completions.create(model="nvidia/nemotron-3.5-lightning:free", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                parsed_json = json.loads(response.choices[0].message.content.strip().replace("```json", "").replace("```", ""))
                llm_is_fake = True if "FAKE" in parsed_json.get("verdict", "").upper() else False
                llm_reason = parsed_json.get("reason", "")
            except: pass

            # --- FINAL VERDICT LOGIC ---
            final_is_fake = llm_is_fake if llm_is_fake is not None else (base_pred == 1)
            if is_ai_text: final_is_fake = True # AI text is flagged fake
            if ai_image_verdict is True: final_is_fake = True # AI image overrides
            if "High" in re_trust: final_is_fake = False # Trusted source overrides
            if "Satire" in re_trust: final_is_fake = True

            st.divider()
            st.subheader("🏆 Final Verdict")
            if final_is_fake:
                st.error("🚨 **OVERALL VERDICT: FAKE / AI-GENERATED**")
                st.write(f"**Reason:** {llm_reason}")
                if is_ai_text: st.warning("⚠️ **Warning:** The text was flagged as likely written by AI (ChatGPT/Bot).")
                if ai_image_verdict: st.warning("⚠️ **Warning:** The uploaded image was flagged as AI-generated.")
                st.markdown("### 📢 Report to Authorities:\n* 🚔 [Cyber Crime Portal](https://cybercrime.gov.in/) | 📰 [PIB Fact Check](https://pib.gov.in/)")
            else:
                st.success("✅ **OVERALL VERDICT: REAL NEWS**")
                st.write(f"**Reason:** {llm_reason}")

            st.write("---")
            st.subheader("🕵️ Source Tracing & Analysis")
            st.markdown(f"**Traced Origin:** `{re_origin}` | **Trust Level:** {re_trust}")
            
            col1, col2 = st.columns(2)
            col1.metric("AI Text Detection", "AI-Generated" if is_ai_text else "Human-Written")
            col2.metric("Baseline Model", f"FAKE {base_conf:.1f}%" if base_pred == 1 else f"REAL {base_conf:.1f}%")

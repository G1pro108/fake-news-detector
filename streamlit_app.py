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
import cv2
import numpy as np
import tempfile
import time
import c2pa

# --- Page Config ---
st.set_page_config(page_title="Multimodal Fake News & Deepfake Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ Multimodal Fake News & Deepfake Detector")
st.markdown("""
Protect yourself from misinformation. You can **paste text, paste a URL, or upload images, text files, and video clips (MP4)**.
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

# --- C2PA PROVENANCE CHECK FUNCTION ---
def analyze_c2pa_metadata(uploaded_file):
    if uploaded_file is None or uploaded_file.name.endswith('.txt'):
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        reader = c2pa.Reader(tmp_path)
        manifest_store = json.loads(reader.json())
        os.remove(tmp_path)
        active_manifest_id = manifest_store.get('active_manifest')
        if not active_manifest_id:
            return {"status": "No Manifest", "summary": "No C2PA manifest found."}
        manifest = manifest_store.get('manifests', {}).get(active_manifest_id, {})
        signer = manifest.get('signature_info', {}).get('issuer', 'Unknown')
        is_ai_generated = False
        for assertion in manifest.get('assertions', []):
            if 'c2pa.ai_generative_training' in assertion.get('label', '') or 'trainedAlgorithmicMedia' in str(assertion):
                is_ai_generated = True
        return {"status": "Valid", "signer": signer, "is_ai_generated_claim": is_ai_generated, "summary": f"Valid C2PA manifest. Signed by: {signer}."}
    except Exception:
        return {"status": "No Manifest", "summary": "No C2PA manifest found or format unsupported."}

# --- LOAD BASELINE MODEL ---
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

# --- HELPER: EXTRACT FRAMES FROM VIDEO ---
def extract_video_frames(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    vidcap = cv2.VideoCapture(tfile.name)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames > 0:
        frame_indices = [int(total_frames * 0.2), int(total_frames * 0.5), int(total_frames * 0.8)]
        for idx in frame_indices:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, image = vidcap.read()
            if success:
                frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    vidcap.release()
    os.remove(tfile.name)
    return frames

# --- UNIFIED UI LAYOUT ---
text_input = st.text_area("📝 Copy & Paste text or URL here:", height=120)
st.markdown("<h4 style='text-align: center; color: gray;'>— OR —</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📁 Upload an Image, Text file, or Video:", type=["jpg", "jpeg", "png", "txt", "mp4", "mov"])

if st.button("Analyze Now", type="primary", use_container_width=True):
    
    input_text = ""
    is_image = is_video = False
    ai_media_verdict = None
    media_api_failed = False
    c2pa_result = None 
    
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        c2pa_result = analyze_c2pa_metadata(uploaded_file)
        
        if filename.endswith('.txt'):
            input_text = uploaded_file.read().decode("utf-8")
            st.info("📄 **Text file loaded.**")
        elif filename.endswith(('.jpg', '.jpeg', '.png')):
            is_image = True
            img = Image.open(uploaded_file)
            input_text = pytesseract.image_to_string(img).strip()
            if input_text: st.info(f"📸 **Text extracted from image:**\n\n{input_text[:200]}...")
            try:
                hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
                img_res = hf_client.image_classification(img, model="dima806/deepfake_vs_real_image_detection")
                if img_res[0]['label'].lower() in ['deepfake', 'fake', 'artificial']:
                    ai_media_verdict = True
                    st.error(f"🚨 **Image is AI-Generated/Deepfake** (Confidence: {img_res[0]['score']*100:.1f}%)")
                else:
                    ai_media_verdict = False
            except Exception: media_api_failed = True
        elif filename.endswith(('.mp4', '.mov')):
            is_video = True
            with st.spinner("Extracting frames from video..."):
                frames = extract_video_frames(uploaded_file)
                if not frames: st.error("Could not extract frames.")
                else:
                    st.info(f"🎞️ **Video processed. Extracting {len(frames)} frames.**")
                    ai_votes = 0
                    try:
                        hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
                        for frame in frames:
                            for attempt in range(3):
                                try:
                                    res = hf_client.image_classification(frame, model="dima806/deepfake_vs_real_image_detection")
                                    if res[0]['label'].lower() in ['deepfake', 'fake', 'artificial']: ai_votes += 1
                                    break 
                                except Exception: time.sleep(1.5) 
                        if ai_votes >= 2:
                            ai_media_verdict = True
                            st.error(f"🚨 **Video is likely AI-Generated!** ({ai_votes}/{len(frames)} frames flagged).")
                        else:
                            ai_media_verdict = False
                            st.success(f"✅ **Video appears real.** ({ai_votes}/{len(frames)} frames flagged).")
                    except Exception: media_api_failed = True
    elif text_input.strip():
        input_text = text_input.strip()
        
    if not input_text and not is_image and not is_video:
        st.warning("⚠️ Please paste some text or upload a file.")
    else:
        with st.spinner("Running Multimodal AI Analysis..."):
            
            is_ai_text = False
            base_pred = 0
            base_conf = 0
            re_origin = "Not found"
            re_trust = "Unknown (No source verified)"
            llm_is_fake = None
            llm_reason = "No text to analyze."

            clean_text = re.sub(r'http\S+', '', input_text).replace('Read more at:', '').strip()
            
            if clean_text:
                # A. AI TEXT DETECTION
                try:
                    hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
                    ai_text_res = hf_client.text_classification(clean_text[:1000], model="openai-community/roberta-base-openai-detector")
                    if ai_text_res[0]['label'].lower() in ['fake', 'chatgpt', 'ai'] and ai_text_res[0]['score'] > 0.6: is_ai_text = True
                except: pass

                # B. BASELINE PREDICTION
                text_vec = tfidf.transform([clean_text])
                base_pred = base_model.predict(text_vec)[0]
                base_conf = base_model.predict_proba(text_vec)[0][base_pred] * 100

                # C. UPGRADED REVERSE ENGINEERING (RAE)
                domain = ""
                url_match = re.search(r'https?://[^\s]+', input_text)
                if url_match:
                    domain = url_match.group(0).split('/')[2].replace('www.', '')
                    re_origin = f"URL Provided: {domain}"
                else:
                    # Clean text specifically for searching (remove punctuation and line breaks)
                    search_text = re.sub(r'[^\w\s]', '', clean_text)[:100].strip()
                    if search_text:
                        try:
                            # 1. Try exact phrase match (most accurate)
                            exact_search = DDGS().text(f'"{search_text}"', max_results=1)
                            if exact_search and exact_search[0].get('href'):
                                domain = exact_search[0]['href'].split('/')[2].replace('www.', '')
                                re_origin = domain
                            else:
                                # 2. Fallback to keyword search if exact match fails
                                keyword_search = DDGS().text(search_text, max_results=1)
                                if keyword_search and keyword_search[0].get('href'):
                                    domain = keyword_search[0]['href'].split('/')[2].replace('www.', '')
                                    re_origin = f"{domain} (Approximate)"
                                else:
                                    re_origin = "No online match found (Highly Suspicious)"
                        except Exception as e:
                            re_origin = "Search API Error"

                if domain:
                    if any(satire in domain for satire in SATIRE_SOURCES): 
                        re_trust = "Low (Known Satire)"
                    elif any(trusted in domain for trusted in TRUSTED_SOURCES): 
                        re_trust = "High (Verified Global Source)"
                    else:
                        re_trust = "Medium (Unknown Source)"

                # D. LLM PREDICTION (RAG)
                try:
                    search_results = DDGS().text(clean_text[:100], max_results=2)
                    context_text = "\n".join([f"- {r['title']}: {r['body']}" for r in search_results])
                    
                    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
                    prompt = f"""
                    Analyze this news. Web Context: {context_text}
                    Traced Origin: '{re_origin}' | Trust: '{re_trust}'
                    Return JSON: "verdict": "REAL" or "FAKE", "reason": "1-2 sentences."
                    Article: {clean_text[:3000]}
                    """
                    response = client.chat.completions.create(model="nvidia/nemotron-3.5-lightning:free", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                    parsed_json = json.loads(response.choices[0].message.content.strip().replace("```json", "").replace("```", ""))
                    llm_is_fake = True if "FAKE" in parsed_json.get("verdict", "").upper() else False
                    llm_reason = parsed_json.get("reason", "")
                except: pass

            # --- FINAL ENSEMBLE VERDICT ---
            final_is_fake = False
            
            if clean_text:
                final_is_fake = llm_is_fake if llm_is_fake is not None else (base_pred == 1)
                if is_ai_text: final_is_fake = True 
                if "High" in re_trust: final_is_fake = False 
            
            if ai_media_verdict is True: final_is_fake = True 
            if c2pa_result and c2pa_result["status"] == "Valid" and c2pa_result.get("is_ai_generated_claim"): final_is_fake = True 

            st.divider()
            st.subheader("🏆 Final Verdict")
            
            if media_api_failed and (is_video or is_image):
                st.warning("⚠️ **AI Media Detection API timed out.**")
                if not clean_text: st.stop()

            if final_is_fake:
                st.error("🚨 **OVERALL VERDICT: FAKE / MISLEADING / AI-GENERATED**")
                if llm_reason: st.write(f"**Reason:** {llm_reason}")
                if is_ai_text: st.warning("⚠️ **Warning:** Text flagged as AI-generated.")
                if ai_media_verdict: st.warning("⚠️ **Warning:** Media flagged as Deepfake.")
                st.markdown("### 📢 Report to Authorities:\n* 🚔 [Cyber Crime Portal](https://cybercrime.gov.in/) | 📰 [PIB Fact Check](https://pib.gov.in/)")
            else:
                st.success("✅ **OVERALL VERDICT: REAL / AUTHENTIC**")
                if llm_reason: st.write(f"**Reason:** {llm_reason}")

            if c2pa_result:
                st.write("---")
                st.subheader("📜 C2PA Provenance Check")
                if c2pa_result["status"] == "Valid":
                    st.success(c2pa_result["summary"])
                else:
                    st.warning(c2pa_result["summary"])

            if clean_text:
                st.write("---")
                st.subheader("🕵️ Reverse Engineering (Source Tracing)")
                
                # Color code the trust level for better UI
                if "High" in re_trust:
                    st.success(f"**Traced Origin:** `{re_origin}` | **Trust Level:** {re_trust}")
                elif "Low" in re_trust or "Suspicious" in re_origin:
                    st.error(f"**Traced Origin:** `{re_origin}` | **Trust Level:** {re_trust}")
                else:
                    st.warning(f"**Traced Origin:** `{re_origin}` | **Trust Level:** {re_trust}")
                
                col1, col2 = st.columns(2)
                col1.metric("AI Text Detection", "AI-Generated" if is_ai_text else "Human-Written")
                col2.metric("Baseline Model", f"FAKE {base_conf:.1f}%" if base_pred == 1 else f"REAL {base_conf:.1f}%")

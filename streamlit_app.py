import streamlit as st
import joblib
import os
import json
import re
import concurrent.futures
from openai import OpenAI
from duckduckgo_search import DDGS
from deep_translator import GoogleTranslator
from transformers import pipeline
from PIL import Image
import pytesseract
import cv2
import numpy as np
import tempfile
import time
import c2pa

# --- Page Config ---
st.set_page_config(page_title="Global Multimodal Fact-Checker", page_icon="🛡️", layout="wide")
st.title("🛡️ Global Multimodal Fact-Checker")
st.markdown("Supports **Text, URL, Images, and Videos**. Analyzes using 350 Swarm Agents & a 3-Head AI Panel.")

# --- LOAD LOCAL MODELS ---
@st.cache_resource
def load_baseline():
    tfidf = joblib.load("tfidf.pkl")
    model = joblib.load("logreg.pkl")
    return tfidf, model

@st.cache_resource
def load_local_vision():
    return pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection", device=-1)

try:
    tfidf, base_model = load_baseline()
    local_vision = load_local_vision()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models. Details: {e}")
    models_loaded = False

# --- C2PA METADATA CHECK ---
def analyze_c2pa(uploaded_file):
    if uploaded_file is None or uploaded_file.name.endswith('.txt'): return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        reader = c2pa.Reader(tmp_path)
        manifest_store = json.loads(reader.json())
        os.remove(tmp_path)
        active_manifest_id = manifest_store.get('active_manifest')
        if not active_manifest_id: return {"status": "No Manifest", "summary": "No C2PA manifest found. (Normal for web files)."}
        manifest = manifest_store.get('manifests', {}).get(active_manifest_id, {})
        signer = manifest.get('signature_info', {}).get('issuer', 'Unknown')
        is_ai = any('c2pa.ai_generative_training' in a.get('label', '') for a in manifest.get('assertions', []))
        return {"status": "Valid", "signer": signer, "is_ai": is_ai, "summary": f"Signed by: {signer}."}
    except:
        return {"status": "No Manifest", "summary": "Metadata stripped (Normal for most web files)."}

# --- VIDEO FRAME EXTRACTION ---
def extract_video_frames(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    vidcap = cv2.VideoCapture(tfile.name)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames > 0:
        for idx in [int(total_frames * 0.2), int(total_frames * 0.5), int(total_frames * 0.8)]:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, image = vidcap.read()
            if success: frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    vidcap.release()
    os.remove(tfile.name)
    return frames

# --- SESSION STATE (For Refresh Button) ---
if 'form_key' not in st.session_state:
    st.session_state.form_key = 0

# --- UI LAYOUT ---
text_input = st.text_area("📝 Copy & Paste text or URL here:", height=120, key=f"text_{st.session_state.form_key}")
st.markdown("<h4 style='text-align: center; color: gray;'>— OR —</h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📁 Upload or Paste an Image, Text file, or Video:", 
    type=["jpg", "jpeg", "png", "txt", "mp4", "mov"], 
    key=f"file_{st.session_state.form_key}",
    help="💡 Tip: Click inside the upload box and press Ctrl+V (or Cmd+V) to paste a screenshot directly!"
)
st.caption("💡 **Tip:** Click the upload box above and press **Ctrl+V** to paste a screenshot directly.")

col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    analyze_clicked = st.button("Deploy Swarm & Analyze", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🔄 Start Over / Refresh", use_container_width=True):
        st.session_state.form_key += 1
        st.rerun()

if analyze_clicked:
    input_text = ""
    is_image = is_video = False
    media_verdict = None
    c2pa_result = None
    re_origin = "Not found"
    re_trust = "Unknown"
    
    # ==========================================
    # PHASE 1: FILE PROCESSING & OCR
    # ==========================================
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        c2pa_result = analyze_c2pa(uploaded_file)
        
        if filename.endswith('.txt'):
            input_text = uploaded_file.read().decode("utf-8")
            st.info("📄 **Text file loaded.**")
        elif filename.endswith(('.jpg', '.jpeg', '.png')):
            is_image = True
            img = Image.open(uploaded_file)
            input_text = pytesseract.image_to_string(img).strip()
            if input_text: st.info(f"📸 **Text extracted from image.**")
            try:
                res = local_vision(img)
                if 'fake' in res[0]['label'].lower() or 'artificial' in res[0]['label'].lower():
                    media_verdict = True
                    st.error("🚨 **Local Vision: Image is AI-Generated!**")
                else:
                    media_verdict = False
                    st.success("✅ **Local Vision: Image appears real.**")
            except Exception as e: st.warning(f"Image vision analysis failed: {e}")
        elif filename.endswith(('.mp4', '.mov')):
            is_video = True
            with st.spinner("Extracting video frames..."):
                frames = extract_video_frames(uploaded_file)
                if frames:
                    ai_votes = sum(1 for f in frames if 'fake' in local_vision(f)[0]['label'].lower())
                    if ai_votes >= 2:
                        media_verdict = True
                        st.error("🚨 **Local Vision: Video is AI-Generated!**")
                    else:
                        media_verdict = False
                        st.success("✅ **Local Vision: Video appears real.**")
    elif text_input.strip():
        input_text = text_input.strip()

    if not input_text and not is_image and not is_video:
        st.warning("⚠️ Please paste text or upload a file.")
    else:
        clean_text = re.sub(r'http\S+', '', input_text).strip()
        
        if clean_text:
            # ==========================================
            # PHASE 2: TRANSLATION & REVERSE ENGINEERING
            # ==========================================
            st.subheader("🌐 Phase 1: Language & Source Processing")
            try:
                translator = GoogleTranslator(source='auto', target='en')
                translated_text = translator.translate(clean_text)
                if translated_text.lower() != clean_text.lower():
                    st.info(f"📝 **Translated to English:** {translated_text[:150]}...")
            except:
                translated_text = clean_text

            domain = ""
            try:
                search_text = re.sub(r'[^\w\s]', '', translated_text)[:80].strip()
                exact_search = DDGS().text(f'"{search_text}"', max_results=1)
                if exact_search:
                    domain = exact_search[0]['href'].split('/')[2].replace('www.', '')
                    re_origin = domain
            except: pass

            TRUSTED = ['reuters.com', 'apnews.com', 'bbc.com', 'thehindu.com', 'indianexpress.com', 'pib.gov.in']
            SATIRE = ['theonion.com', 'fakingnews.com']
            
            if domain:
                if any(s in domain for s in SATIRE): re_trust = "Low (Known Satire)"
                elif any(t in domain for t in TRUSTED): re_trust = "High (Verified Source)"
                else: re_trust = "Medium (Unknown)"
            else:
                re_trust = "Unknown (No exact match online)"

            st.markdown(f"**Traced Origin:** `{re_origin}` | **Trust Level:** {re_trust}")

            # ==========================================
            # PHASE 3: THE 350-AGENT SWARM
            # ==========================================
            st.subheader("🐝 Phase 2: 350-Agent Swarm Search")
            base_query = " ".join(translated_text.split()[:12])
            domain_pools = {
                "Politics": ["site:reuters.com", "site:thehindu.com", "election"],
                "International": ["site:bbc.com", "site:un.org", "diplomacy"],
                "Business": ["site:bloomberg.com", "site:livemint.com", "market"],
                "Health": ["site:who.int", "site:nih.gov", "medical"],
                "Science": ["site:nature.com", "study", "research"],
                "Environment": ["site:nasa.gov", "climate", "disaster"],
                "Crime": ["site:ndtv.com", "police", "court"],
                "Sports": ["site:espn.com", "tournament", "match"],
                "Entertainment": ["site:variety.com", "movie", "celebrity"],
                "FactCheck": ["site:snopes.com", "site:altnews.in", "site:pib.gov.in", "hoax"]
            }
            agent_votes = []
            progress_bar = st.progress(0)
            
            def agent_task(query, site):
                try:
                    time.sleep(0.1) 
                    results = DDGS().text(f"{query} {site}", max_results=1)
                    if not results: return 0
                    txt = " ".join([r['body'] for r in results]).lower()
                    if any(w in txt for w in ["fake", "hoax", "false", "misleading"]): return -1
                    if any(w in txt for w in ["true", "confirmed", "real", "official"]): return 1
                    return 0
                except: return 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(agent_task, base_query, sites[i % len(sites)]) 
                           for domain, sites in domain_pools.items() for i in range(35)]
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    agent_votes.append(future.result())
                    progress_bar.progress((i + 1) / 350)
            
            fake_signals = agent_votes.count(-1)
            real_signals = agent_votes.count(1)
            st.success(f"✅ 350 Agents deployed. Found {fake_signals} Fake / {real_signals} Real signals.")
            progress_bar.empty()

            # ==========================================
            # PHASE 4: THE 3-JUDGE PANEL (FIXED)
            # ==========================================
            st.subheader("🏛️ Phase 3: Consulting the 3-Judge Panel...")
            judge_votes = []
            text_for_judges = translated_text[:1500]

            # Judge 1: Google Gemini (Updated Model & Error Display)
            with st.spinner("Judge 1 (Google Gemini) is analyzing..."):
                try:
                    gemini_client = OpenAI(
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
                        api_key=st.secrets["GEMINI_API_KEY"]
                    )
                    res = gemini_client.chat.completions.create(
                        model="gemini-2.0-flash", # ✅ Updated model
                        messages=[{"role": "user", "content": f"Is this text real or fake? Return ONLY 'REAL' or 'FAKE'. Text: {text_for_judges}"}]
                    )
                    verdict = res.choices[0].message.content.strip().upper()
                    if "FAKE" in verdict: judge_votes.append(-1); st.error("Judge 1 (Gemini): Voted FAKE")
                    elif "REAL" in verdict: judge_votes.append(1); st.success("Judge 1 (Gemini): Voted REAL")
                except Exception as e: 
                    st.warning(f"Judge 1 (Gemini) Error: {e}") # ✅ Shows exact error

            # Judge 2: OpenRouter (Updated Free Model & Error Display)
            with st.spinner("Judge 2 (OpenRouter) is analyzing..."):
                try:
                    or_client = OpenAI(
                        base_url="https://openrouter.ai/api/v1", 
                        api_key=st.secrets["OPENROUTER_API_KEY"]
                    )
                    res = or_client.chat.completions.create(
                        model="meta-llama/llama-3.1-8b-instruct:free", # ✅ Highly reliable free model
                        messages=[{"role": "user", "content": f"Is this text real or fake? Return ONLY 'REAL' or 'FAKE'. Text: {text_for_judges}"}]
                    )
                    verdict = res.choices[0].message.content.strip().upper()
                    if "FAKE" in verdict: judge_votes.append(-1); st.error("Judge 2 (OpenRouter): Voted FAKE")
                    elif "REAL" in verdict: judge_votes.append(1); st.success("Judge 2 (OpenRouter): Voted REAL")
                except Exception as e: 
                    st.warning(f"Judge 2 (OpenRouter) Error: {e}") # ✅ Shows exact error

            # Judge 3: Sarvam AI (Error Display)
            with st.spinner("Judge 3 (Sarvam AI) is analyzing..."):
                try:
                    sarvam_client = OpenAI(
                        base_url="https://api.sarvam.ai/v1", 
                        api_key=st.secrets["SARVAM_API_KEY"]
                    )
                    res = sarvam_client.chat.completions.create(
                        model="sarvam-105b", 
                        messages=[{"role": "user", "content": f"Analyze for Indian context. Is this REAL or FAKE? Return ONLY 'REAL' or 'FAKE'. Text: {text_for_judges}"}]
                    )
                    verdict = res.choices[0].message.content.strip().upper()
                    if "FAKE" in verdict: judge_votes.append(-1); st.error("Judge 3 (Sarvam): Voted FAKE")
                    elif "REAL" in verdict: judge_votes.append(1); st.success("Judge 3 (Sarvam): Voted REAL")
                except Exception as e: 
                    st.warning(f"Judge 3 (Sarvam) Error: {e}") # ✅ Shows exact error

            # ==========================================
            # PHASE 5: FINAL CONSENSUS (With Swarm Fallback)
            # ==========================================
            st.divider()
            st.subheader("🏆 Final Consensus Verdict")
            
            # CHECK IF ALL JUDGES FAILED
            if len(judge_votes) == 0:
                st.warning("⚠️ **All 3 Head Judges are currently unavailable (API limits reached or offline).**")
                st.info("🔄 **Swarm Fallback Activated:** The system is relying entirely on the 350-Agent majority voting. *Please refresh tomorrow when the AI Judge limits reset for full analysis.*")
                
                # Decision based purely on Swarm
                final_is_fake = (fake_signals > real_signals)
                confidence_text = f"Based solely on 350 Swarm Agents: {fake_signals} Fake vs {real_signals} Real signals."
                
            else:
                # Decision based on Weighted Voting (Swarm + Judges)
                total_fake = (fake_signals * 1) + (judge_votes.count(-1) * 50)
                total_real = (real_signals * 1) + (judge_votes.count(1) * 50)
                final_is_fake = total_fake > total_real
                confidence_text = f"(Swarm: {fake_signals} F / {real_signals} R | Judges: {judge_votes.count(-1)} F / {judge_votes.count(1)} R)"

            # Media & C2PA Overrides
            if media_verdict is True: final_is_fake = True
            if c2pa_result and c2pa_result["status"] == "Valid" and c2pa_result.get("is_ai"): final_is_fake = True

            # Display Final Result
            if final_is_fake:
                st.error(f"🚨 **OVERALL VERDICT: FAKE / MISLEADING**\n{confidence_text}")
            else:
                st.success(f"✅ **OVERALL VERDICT: REAL / AUTHENTIC**\n{confidence_text}")

        # Display C2PA Results
        if c2pa_result:
            st.write("---")
            st.subheader("📜 C2PA Provenance Check")
            if c2pa_result["status"] == "Valid": st.success(c2pa_result["summary"])
            else: st.warning(c2pa_result["summary"])

# --- PERMANENT REPORTING PORTAL ---
st.write("---")
st.subheader("📢 Report Misinformation to Authorities")
st.markdown("""
If you suspect any content you checked is malicious, defamatory, or spreading panic, please report it immediately to the official Indian Government portals:
*   🚔 **[National Cyber Crime Reporting Portal](https://cybercrime.gov.in/)** (Select 'Report Cyber Crime' -> 'Fake News')
*   📰 **[PIB Fact Check](https://pib.gov.in/factcheck.aspx)** (WhatsApp: +91-8799717971)
*   🐦 **Twitter/X:** Tag `@Cyberdost` (Ministry of Home Affairs)
""")

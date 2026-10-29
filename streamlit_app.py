# ============================================================
# Global Multimodal Fact-Checker — Comprehensive LLM Registry
# ============================================================

import streamlit as st
import joblib
import os
import io
import re
import time
import tempfile
import numpy as np
import cv2
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageChops
from transformers import pipeline
from openai import OpenAI
from deep_translator import GoogleTranslator

# ---- Optional C2PA (graceful fallback if package missing) ----
try:
    from fast_c2pa_python import read_c2pa_from_file
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Global Multimodal Fact-Checker",
    page_icon="🛡️",
    layout="wide",
)
st.title("🛡️ Global Multimodal Fact-Checker")
st.markdown(
    "Supports **Text, URL, Images, Videos, and Audio**. "
    "Local-first AI with direct web scraping fallback."
)

# ============================================================
# COMPREHENSIVE LLM PROVIDER REGISTRY
# ============================================================
PROVIDER_DEFAULTS = {
    "OpenAI": {
        "url": "https://api.openai.com/v1",
        "models": [
            "gpt-6-astra", "gpt-6", "gpt-5.6-sol", "gpt-5.5", 
            "gpt-5.4", "gpt-5.3-codex", "o3-mini", "o1-mini", 
            "gpt-4o", "gpt-4o-mini"
        ]
    },
    "Anthropic": {
        "url": "https://api.anthropic.com/v1",
        "models": [
            "claude-opus-5", "claude-opus-4.7", "claude-sonnet-5", 
            "claude-haiku-5.5", "claude-3-5-sonnet"
        ]
    },
    "Google Gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": [
            "gemini-4-argon", "gemini-3.6-flash", "gemini-3-pro-preview", 
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"
        ]
    },
    "Meta (Llama)": {
        "url": "https://api.llama-api.com/v1",
        "models": [
            "llama-4-maverick", "llama-4-scout", "llama-3.3-70b", 
            "llama-3.1-70b", "llama-3.1-8b"
        ]
    },
    "Mistral AI": {
        "url": "https://api.mistral.ai/v1",
        "models": [
            "mistral-large-latest", "mistral-small-latest", "ministral-3-14b",
            "ministral-3-8b", "open-mistral-nemo", "mixtral-8x7b"
        ]
    },
    "xAI (Grok)": {
        "url": "https://api.x.ai/v1",
        "models": ["grok-4.5", "grok-4", "grok-3"]
    },
    "DeepSeek": {
        "url": "https://api.deepseek.com",
        "models": ["deepseek-v4.1-flash", "deepseek-v4", "deepseek-chat", "deepseek-reasoner"]
    },
    "Alibaba Cloud (Qwen)": {
        "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen3.8-max", "qwen3.7-plus", "qwen-max", "qwen-plus", "qwen-turbo"]
    },
    "Z.ai (GLM)": {
        "url": "https://api.z.ai/api/paas/v4/",
        "models": ["glm-5.3", "glm-5.2", "glm-4.6", "glm-4"]
    },
    "Cohere": {
        "url": "https://api.cohere.ai/v1",
        "models": ["command-r-plus", "command-r", "command-light"]
    },
    "Groq (Ultra-Fast)": {
        "url": "https://api.groq.com/openai/v1",
        "models": [
            "llama-3.3-70b-versatile", "llama-3.1-8b-instant", 
            "mixtral-8x7b-32768", "gemma2-9b-it"
        ]
    },
    "OpenRouter (Aggregator)": {
        "url": "https://openrouter.ai/api/v1",
        "models": [
            "openai/gpt-6-astra", "anthropic/claude-opus-5", 
            "meta-llama/llama-4-maverick", "google/gemini-3.6-flash",
            "deepseek/deepseek-v4.1-flash"
        ]
    },
    "Together AI": {
        "url": "https://api.together.xyz/v1",
        "models": [
            "meta-llama/Llama-4-Maverick", 
            "Qwen/Qwen3.8-Max", 
            "deepseek-ai/DeepSeek-V4.1"
        ]
    },
    "Nvidia NIM": {
        "url": "https://integrate.api.nvidia.com/v1",
        "models": [
            "meta/llama-4-maverick", "meta/llama-3.1-70b-instruct", 
            "nvidia/nemotron-4-340b-instruct", "nvidia/nemotron-nano-12b"
        ]
    },
    "Cerebras": {
        "url": "https://api.cerebras.ai/v1",
        "models": ["llama3.1-8b", "llama3.1-70b"]
    },
    "Sarvam AI": {
        "url": "https://api.sarvam.ai/v1",
        "models": ["sarvam-105b"]
    },
    "AI21 Labs": {
        "url": "https://api.ai21.com/studio/v1",
        "models": ["jamba-1-5-mini", "jamba-1-5-large"]
    },
    "MiniMax": {
        "url": "https://api.minimax.chat/v1",
        "models": ["minimax-m3", "minimax-m2.1"]
    },
    "Moonshot AI (Kimi)": {
        "url": "https://api.moonshot.cn/v1",
        "models": ["kimi-k3.1-pro", "moonshotai/kimi-k2.5"]
    },
    "Amazon (Bedrock)": {
        "url": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "models": ["nova-pro", "nova-lite", "nova-micro", "nova-premier"]
    },
    "Custom (OpenAI-compatible)": {
        "url": "",
        "models": ["Custom..."]
    }
}

with st.sidebar:
    st.header("⚙️ LLM Judge Settings")
    st.markdown("Configure up to **3** judges simultaneously.")
    
    tabs = st.tabs(["Judge 1", "Judge 2", "Judge 3"])
    
    for i, tab in enumerate(tabs):
        slot_num = i + 1
        state_key = f"saved_judge_{slot_num}"
        input_key = f"api_key_input_{slot_num}"
        
        with tab:
            if state_key in st.session_state:
                saved_key = st.session_state[state_key]['api_key']
                masked_key = f"{saved_key[:4]}...{saved_key[-4:]}" if len(saved_key) > 8 else "***"
                
                st.success(f"✅ Active: {st.session_state[state_key]['name']}")
                st.caption(f"Key: {masked_key}")
                
                if st.button(f"Remove Judge {slot_num}", key=f"rem_{slot_num}", use_container_width=True):
                    del st.session_state[state_key]
                    st.rerun()
            else:
                provider_choice = st.selectbox(f"Provider", list(PROVIDER_DEFAULTS.keys()), key=f"prov_{slot_num}")
                default_url = PROVIDER_DEFAULTS[provider_choice]["url"]
                base_url = st.text_input(f"Base URL", value=default_url, key=f"url_{slot_num}")
                
                model_options = PROVIDER_DEFAULTS[provider_choice]["models"] + ["Custom..."]
                model_choice = st.selectbox(f"Model", model_options, key=f"model_sel_{slot_num}")
                
                if model_choice == "Custom...":
                    model_name = st.text_input(f"Enter Custom Model Name", value="", key=f"custom_{slot_num}")
                else:
                    model_name = model_choice

                api_key = st.text_input(f"API Key", type="password", key=input_key)
                
                if st.button(f"Activate Judge {slot_num}", key=f"btn_{slot_num}", use_container_width=True):
                    if api_key and base_url and model_name:
                        st.session_state[state_key] = {
                            "name": f"{provider_choice} ({model_name})",
                            "api_key": api_key,
                            "base_url": base_url,
                            "model": model_name
                        }
                        del st.session_state[input_key]
                        st.rerun()
                    else:
                        st.error("Please fill all fields.")

# ============================================================
# MODEL LOADERS
# ============================================================
@st.cache_resource
def load_baseline():
    try:
        tfidf = joblib.load("models/baseline/tfidf.pkl")
        model = joblib.load("models/baseline/logreg.pkl")
        return tfidf, model
    except Exception:
        return None, None

@st.cache_resource
def load_hf_fallback():
    try:
        return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news")
    except Exception:
        return None

@st.cache_resource
def load_local_vision():
    try:
        return pipeline("image-classification", model="prithivMLmods/deepfake-detector-model-v1", device=-1)
    except Exception:
        return None

@st.cache_resource
def load_audio_detector():
    try:
        from jabberjay import Jabberjay
        return Jabberjay()
    except Exception:
        return None

tfidf, base_model = load_baseline()
hf_fallback = load_hf_fallback()

# ============================================================
# OPTIMIZED AUDIO & IMAGE DETECTORS
# ============================================================
def analyze_audio(uploaded_file):
    detector = load_audio_detector()
    if not detector:
        return None, 0.0, "Jabberjay audio detector not installed."

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        result = detector.detect(tmp_path)
        label = result.label.lower()
        confidence = float(result.confidence)
        is_fake = "spoof" in label or not getattr(result, "is_bonafide", False)
        return is_fake, confidence, f"Jabberjay → {result.label} ({confidence:.2%})"
    except Exception as e:
        return None, 0.0, f"Audio analysis failed: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def analyze_image_deepfake(img):
    vision_model = load_local_vision()
    if not vision_model:
        return False, "Vision model not loaded."
    
    img_fast = img.copy()
    img_fast.thumbnail((512, 512))
    
    results = vision_model(img_fast)
    top = results[0]
    is_fake = any(k in top["label"].lower() for k in ("fake", "artificial", "ai", "synthetic"))
    return is_fake, f"{top['label']} ({top['score']:.2%})"

def ela_manipulation_score(img, quality=90):
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)
    diff = ImageChops.difference(img.convert("RGB"), recompressed.convert("RGB"))
    return max(e[1] for e in diff.getextrema())

# ============================================================
# C2PA PROVENANCE
# ============================================================
def analyze_c2pa(uploaded_file):
    if not C2PA_AVAILABLE: return None
    if uploaded_file is None or uploaded_file.name.lower().endswith((".txt", ".wav", ".mp3", ".m4a", ".flac")): return None
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        metadata = read_c2pa_from_file(tmp_path)
        os.remove(tmp_path)

        if not metadata: return {"status": "No Manifest", "summary": "No C2PA manifest found."}
        signer = metadata.get("generator", "Unknown") if isinstance(metadata, dict) else "Unknown"
        is_ai = "ai" in str(metadata).lower() or "generative" in str(metadata).lower()
        return {"status": "Valid", "signer": signer, "is_ai": is_ai, "summary": f"Signed by: {signer}. AI: {is_ai}"}
    except Exception:
        return None

# ============================================================
# FAST VIDEO FRAME EXTRACTION
# ============================================================
def extract_video_frames(video_file):
    video_file.seek(0)
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close()

    vidcap = cv2.VideoCapture(tfile.name)
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total > 0:
        # Grab only 1 frame at 50% for speed
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.5))
        ok, frame = vidcap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    vidcap.release()
    if os.path.exists(tfile.name):
        os.remove(tfile.name)
    return frames

# ============================================================
# DIRECT WEB SCRAPER (No API Needed)
# ============================================================
def search_web_via_url(query):
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.post(url, data={"q": query}, headers=headers, timeout=10)
        response.raise_for_status()
        
        if response.status_code == 202:
            return [], "Rate-limited by DuckDuckGo."
            
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        
        for result in soup.find_all("div", class_="result__body"):
            title_elem = result.find("a", class_="result__a")
            snippet_elem = result.find("a", class_="result__snippet")
            
            if title_elem:
                title = title_elem.get_text(strip=True)
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                results.append({"title": title, "body": snippet})
                
            if len(results) >= 3: break
                
        if not results:
            return [], "No results parsed."
            
        return results, "Success"
    except Exception as e:
        return [], f"Network error: {e}"

def analyze_with_search(text):
    base_query = " ".join(text.split()[:12])
    results, status_msg = search_web_via_url(f'"{base_query}" fact check')
    
    if not results:
        return None, status_msg, ""

    fake_words = {"fake", "hoax", "false", "misleading", "debunked"}
    real_words = {"true", "confirmed", "official", "verified", "authentic"}
    score = 0
    live_context = ""
    
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        combined = f"{title} {body}".lower()
        live_context += f"- **{title}**: {body}\n"
        if any(w in combined for w in fake_words): score -= 1
        if any(w in combined for w in real_words): score += 1

    return score, f"Scraped {len(results)} live web results.", live_context

# ============================================================
# UNIVERSAL DYNAMIC LLM SCANNER
# ============================================================
def get_dynamic_llm_providers():
    providers = []
    for i in range(1, 4):
        saved = st.session_state.get(f"saved_judge_{i}")
        if saved:
            providers.append(saved)

    try:
        for key_name in st.secrets:
            if key_name.endswith("_API_KEY") and st.secrets[key_name].strip():
                prefix = key_name.replace("_API_KEY", "").upper()
                url = st.secrets.get(f"{prefix}_BASE_URL")
                model = st.secrets.get(f"{prefix}_MODEL")
                
                if not url or not model:
                    for prov, details in PROVIDER_DEFAULTS.items():
                        if prefix in prov.upper().replace(" ", ""):
                            if not url: url = details["url"]
                            if not model and details["models"]: model = details["models"][0]
                            break
                            
                if url and model:
                    providers.append({
                        "name": f"{prefix.capitalize()} (Secret)",
                        "base_url": url,
                        "model": model,
                        "api_key": st.secrets[key_name]
                    })
    except Exception:
        pass
    return providers

def ask_llm(provider, text):
    try:
        headers = {"HTTP-Referer": "http://localhost:8501"} if "openrouter" in provider["base_url"] else None
        client = OpenAI(base_url=provider["base_url"], api_key=provider["api_key"], default_headers=headers)
        res = client.chat.completions.create(
            model=provider["model"],
            messages=[{"role": "user", "content": f"Is this REAL or FAKE? Return ONLY one word. Text: {text}"}],
        )
        verdict = res.choices[0].message.content.strip().upper()
        return -1 if "FAKE" in verdict else (1 if "REAL" in verdict else None)
    except Exception as e:
        return f"ERROR: {str(e)}" 

# ============================================================
# STATE RESET & UI
# ============================================================
def reset_app():
    for k in list(st.session_state.keys()):
        if k.startswith(("text_", "file_", "form_")):
            del st.session_state[k]
    st.session_state.form_key = 0
    st.rerun()

if "form_key" not in st.session_state:
    st.session_state.form_key = 0

text_input = st.text_area("📝 Copy & Paste text or URL here:", height=120, key=f"text_{st.session_state.form_key}")
st.markdown("<h4 style='text-align: center; color: gray;'>— OR —</h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📎 Upload or Paste an Image, Text, Audio, or Video:",
    type=["jpg", "jpeg", "png", "txt", "mp4", "mov", "wav", "mp3", "m4a", "flac"],
    key=f"file_{st.session_state.form_key}",
    help="Tip: Click inside the upload box and press Ctrl+V to paste a screenshot directly!",
)
st.caption("💡 **Tip:** Click the upload box above and press **Ctrl+V** to paste a screenshot directly.")

col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1: analyze_clicked = st.button("🚀 Deploy Analysis", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🔄 Start Over / Refresh", use_container_width=True): reset_app()

# ============================================================
# MAIN PROCESSING
# ============================================================
if analyze_clicked:
    input_text = ""
    media_verdict = None
    media_analyzed = False
    c2pa_result = None
    is_image = is_video = is_audio = False

    # ---------- PHASE 1: FILE PROCESSING ----------
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        c2pa_result = analyze_c2pa(uploaded_file)

        if filename.endswith(".txt"):
            input_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.info("📄 **Text file loaded.**")

        elif filename.endswith((".jpg", ".jpeg", ".png")):
            is_image, media_analyzed = True, True
            img = Image.open(uploaded_file)
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text: 
                input_text += "\n" + ocr_text 
                st.info("📸 **Text extracted from image.**")

            vision_fake, vision_detail = analyze_image_deepfake(img)
            ela_score = ela_manipulation_score(img)

            if vision_fake:
                media_verdict = True
                st.error(f"🖼️ **AI-Generated Image Detected** — {vision_detail}")
            elif ela_score > 40:
                media_verdict = True
                st.warning(f"🖼️ **Possible Manipulation Detected** — ELA {ela_score}")
            else:
                media_verdict = False
                st.success(f"✅ **Image appears authentic** — {vision_detail}, ELA {ela_score}")

        elif filename.endswith((".mp4", ".mov")):
            is_video, media_analyzed = True, True
            with st.spinner("Extracting video frame (Fast Mode)..."):
                frames = extract_video_frames(uploaded_file)
            if not frames:
                st.warning("Could not extract frames.")
            else:
                v_fake, v_detail = analyze_image_deepfake(frames[0])
                ela = ela_manipulation_score(frames[0])
                
                if v_fake or ela > 40:
                    media_verdict = True
                    st.error(f"🎥 **Video shows manipulation signals** — {v_detail}, ELA {ela}")
                else:
                    media_verdict = False
                    st.success(f"✅ **Sampled video frame appears authentic.**")

        elif filename.endswith((".wav", ".mp3", ".m4a", ".flac")):
            is_audio, media_analyzed = True, True
            with st.spinner("Analyzing audio for cloning..."):
                audio_is_fake, _, audio_detail = analyze_audio(uploaded_file)
                if audio_is_fake is True:
                    media_verdict = True
                    st.error(f"🔊 **Audio Deepfake Detected** — {audio_detail}")
                elif audio_is_fake is False:
                    media_verdict = False
                    st.success(f"✅ **Audio appears authentic** — {audio_detail}")
                else:
                    st.warning(audio_detail)

    if text_input.strip():
        input_text = text_input.strip() + "\n" + input_text

    if not input_text.strip() and not is_image and not is_video and not is_audio:
        st.warning("⚠️ Please paste text or upload a file.")
    else:
        clean_text = re.sub(r"http\S+", "", input_text).strip()

        if clean_text:
            # ---------- PHASE 1: LANGUAGE & LOCAL MODELS ----------
            st.subheader("🌐 Phase 1: Language & Local Models")
            try:
                translated_text = GoogleTranslator(source="auto", target="en").translate(clean_text)
                if translated_text.lower() != clean_text.lower():
                    st.info(f"📝 **Translated to English:** {translated_text[:150]}...")
            except Exception:
                translated_text = clean_text

            tfidf_vote = None
            hf_vote = None

            if tfidf is not None and base_model is not None:
                try:
                    vec = tfidf.transform([translated_text])
                    pred = base_model.predict(vec)[0]
                    prob = base_model.predict_proba(vec)[0][pred] * 100
                    if pred == 1:
                        st.error(f"🚨 **Local TF-IDF Model:** FAKE (Confidence: {prob:.2f}%)")
                        tfidf_vote = -1
                    else:
                        st.success(f"✅ **Local TF-IDF Model:** REAL (Confidence: {prob:.2f}%)")
                        tfidf_vote = 1
                except Exception: pass
            
            if hf_fallback is not None:
                try:
                    hf_res = hf_fallback(translated_text[:512])[0]
                    hf_pred = 1 if hf_res['label'] == 'LABEL_1' else 0
                    hf_prob = hf_res['score'] * 100
                    if hf_pred == 0:
                        st.error(f"🚨 **HuggingFace AI Model:** FAKE (Confidence: {hf_prob:.2f}%)")
                        hf_vote = -1
                    else:
                        st.success(f"✅ **HuggingFace AI Model:** REAL (Confidence: {hf_prob:.2f}%)")
                        hf_vote = 1
                except Exception: pass

            text_votes = [v for v in [tfidf_vote, hf_vote] if v is not None]
            final_text_vote = None
            if text_votes:
                final_text_vote = -1 if text_votes.count(-1) > text_votes.count(1) else 1
                text_fake = (final_text_vote == -1)
            else: 
                text_fake = False

            # ---------- PHASE 2: DIRECT WEB URL SEARCH ----------
            st.subheader("🔎 Phase 2: Live Web Search (No API)")
            search_score, search_detail, live_context = analyze_with_search(translated_text)
            
            if search_score is None:
                st.warning(f"Web search skipped: {search_detail}")
            else:
                st.info(search_detail)
                if live_context:
                    with st.expander("🌐 View Scraped Web Context"):
                        st.markdown(live_context)

            # ---------- PHASE 3: UNIVERSAL AI JUDGES ----------
            st.subheader("🏛️ Phase 3: Consulting AI Judges...")
            judge_votes = []
            text_for_judges = f"Web Context:\n{live_context}\n\nText to evaluate:\n{translated_text[:1500]}"
            api_ran = False

            active_providers = get_dynamic_llm_providers()

            for provider in active_providers:
                vote = ask_llm(provider, text_for_judges)
                
                if isinstance(vote, str) and vote.startswith("ERROR"):
                    st.error(f"Judge {provider['name']}: {vote}")
                elif vote is not None:
                    api_ran = True
                    judge_votes.append(vote)
                    if vote == -1: st.error(f"Judge {provider['name']}: Voted FAKE")
                    else: st.success(f"Judge {provider['name']}: Voted REAL")
                else: st.warning(f"Judge {provider['name']}: Failed.")
            
            if not api_ran:
                st.info("ℹ️ No working API keys found. Relying entirely on Local AI + Live Web Context.")

            # ---------- PHASE 4: CONSENSUS (FIXED UNKNOWN BUG) ----------
            st.divider()
            st.subheader("🏆 Final Consensus Verdict")
            
            local_fake = media_verdict is True
            search_fake = (search_score is not None and search_score < 0)
            judge_fake = (judge_votes.count(-1) > judge_votes.count(1)) if judge_votes else False

            media_weight = 2 if media_analyzed else 0
            # FIX: If there is text, we ALWAYS give it weight now, even if local models fail.
            # This prevents the total_weight from being 0.
            text_weight = 2 if clean_text else 0
            search_weight = 1 if search_score is not None else 0
            judge_weight = 3 if judge_votes else 0

            fake_score = ((media_weight if local_fake else 0) + (text_weight if text_fake else 0) + (search_weight if search_fake else 0) + (judge_weight if judge_votes and judge_fake else 0))
            total_weight = media_weight + text_weight + search_weight + judge_weight

            if total_weight == 0:
                st.warning("⚠️ **UNKNOWN:** The models could not confidently analyze this content. Try providing a longer piece of text.")
            else:
                fake_ratio = fake_score / total_weight
                
                confidence = f"Media: {'FAKE' if local_fake else ('REAL' if media_analyzed else 'n/a')} | Text: {'FAKE' if text_fake else 'REAL'} | Web: {'FAKE' if search_fake else ('REAL' if search_score is not None else 'n/a')} | Judges: {judge_votes.count(-1)}F / {judge_votes.count(1)}R"

                if c2pa_result and c2pa_result.get("status") == "Valid" and c2pa_result.get("is_ai"):
                    fake_ratio = 1.0
                    confidence += " | C2PA: AI flag → forced FAKE"

                if fake_ratio > 0.6:
                    st.error(f"🚨 **VERDICT: LIKELY FAKE / MISLEADING**\n\n{confidence}")
                elif fake_ratio < 0.4:
                    st.success(f"✅ **VERDICT: LIKELY REAL / AUTHENTIC**\n\n{confidence}")
                else:
                    st.warning(f"⚖️ **VERDICT: UNCERTAIN / MIXED SIGNALS**\n\n{confidence}")

        # ---------- C2PA REPORT ----------
        if c2pa_result:
            st.write("---")
            st.subheader("📜 C2PA Provenance Check")
            if c2pa_result["status"] == "Valid": st.success(c2pa_result["summary"])
            elif c2pa_result["status"] == "Unavailable": st.info(c2pa_result["summary"])
            else: st.warning(c2pa_result["summary"])

# ============================================================
# REPORTING PORTAL
# ============================================================
st.write("---")
st.subheader("📢 Report Misinformation to Authorities")
st.markdown("""
*   🚔 **[National Cyber Crime Reporting Portal](https://cybercrime.gov.in/)**
*   📰 **[PIB Fact Check](https://pib.gov.in/factcheck.aspx)**
*   🐦 **Twitter/X:** Tag `@Cyberdost`
""")

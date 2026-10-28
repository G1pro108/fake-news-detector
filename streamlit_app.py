# ============================================================
# Global Multimodal Fact-Checker — Fixed & Enhanced Version
# Supports: Text, URL, Image, Video, Audio (cloning detection),
#           C2PA provenance, deepfake + ELA tamper detection,
#           optional LLM judges, robust refresh, no-fail operation.
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
from PIL import Image, ImageChops
from transformers import pipeline
from duckduckgo_search import DDGS
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
    "Local-first AI with optional LLM judges. Works fully offline for media analysis."
)


# ============================================================
# MODEL LOADERS
# ============================================================
@st.cache_resource
def load_baseline():
    try:
        tfidf = joblib.load("models/baseline/tfidf.pkl")
        model = joblib.load("models/baseline/logreg.pkl")
        return tfidf, model
    except Exception as e:
        return None, None

@st.cache_resource
def load_local_vision():
    try:
        return pipeline(
            "image-classification",
            model="prithivMLmods/deepfake-detector-model-v1",
            device=-1,
        )
    except Exception:
        return None

@st.cache_resource
def load_audio_detector():
    try:
        from jabberjay import Jabberjay
        return Jabberjay()
    except Exception:
        return None

# Try to load baseline models; warn but continue if missing.
tfidf, base_model = load_baseline()
models_loaded = tfidf is not None


# ============================================================
# AUDIO CLONING DETECTION
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
        detail = f"Jabberjay → {result.label} (confidence {confidence:.2%})"
        return is_fake, confidence, detail
    except Exception as e:
        return None, 0.0, f"Audio analysis failed: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# IMAGE DEEPFAKE + ELA TAMPER DETECTION
# ============================================================
def analyze_image_deepfake(img):
    vision_model = load_local_vision()
    if not vision_model:
        return False, "Vision model not loaded."
    
    results = vision_model(img)
    top = results[0]
    label = top["label"].lower()
    score = top["score"]
    is_fake = any(k in label for k in ("fake", "artificial", "ai", "synthetic"))
    return is_fake, f"{label} ({score:.2%})"

def ela_manipulation_score(img, quality=90):
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)
    diff = ImageChops.difference(img.convert("RGB"), recompressed.convert("RGB"))
    extrema = diff.getextrema()
    return max(e[1] for e in extrema)


# ============================================================
# C2PA PROVENANCE
# ============================================================
def analyze_c2pa(uploaded_file):
    if not C2PA_AVAILABLE:
        return {"status": "Unavailable", "summary": "C2PA reader not installed."}
    if uploaded_file is None or uploaded_file.name.lower().endswith((".txt", ".wav", ".mp3", ".m4a", ".flac")):
        return None
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        metadata = read_c2pa_from_file(tmp_path)
        os.remove(tmp_path)

        if not metadata:
            return {"status": "No Manifest", "summary": "No C2PA manifest found."}

        signer = metadata.get("generator", "Unknown") if isinstance(metadata, dict) else "Unknown"
        is_ai = "ai" in str(metadata).lower() or "generative" in str(metadata).lower()
        return {"status": "Valid", "signer": signer, "is_ai": is_ai, "summary": f"Signed by: {signer}. AI: {is_ai}"}
    except Exception as e:
        return {"status": "No Manifest", "summary": f"Could not read C2PA ({e})."}


# ============================================================
# VIDEO FRAME EXTRACTION
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
        for frac in (0.2, 0.5, 0.8):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
            ok, frame = vidcap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    vidcap.release()
    if os.path.exists(tfile.name):
        os.remove(tfile.name)
    return frames


# ============================================================
# SAFE FACT-CHECK SEARCH
# ============================================================
def fact_check_search(query, max_results=5):
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                return results
        except Exception:
            time.sleep(2 ** attempt)
    return []

def analyze_with_search(text):
    base_query = " ".join(text.split()[:12])
    results = fact_check_search(f'"{base_query}" fact check')
    if not results:
        return None, "No search results."
    fake_words = {"fake", "hoax", "false", "misleading", "debunked"}
    real_words = {"true", "confirmed", "official", "verified", "authentic"}
    score = 0
    for r in results:
        body = (r.get("body", "") + " " + r.get("title", "")).lower()
        if any(w in body for w in fake_words): score -= 1
        if any(w in body for w in real_words): score += 1
    return score, f"Scanned {len(results)} fact-check results."


# ============================================================
# DYNAMIC LLM JUDGES (Auto-detects any key in Secrets)
# ============================================================
# If you add any of these keys to Streamlit Secrets, they will auto-activate.
LLM_PROVIDERS = [
    {"name": "Gemini", "key": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-1.5-flash"},
    {"name": "Groq", "key": "GROQ_API_KEY", "base_url": "https://api.groq.com/openai/v1", "model": "llama3-8b-8192"},
    {"name": "OpenRouter", "key": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.1-8b-instruct:free"},
    {"name": "Sarvam", "key": "SARVAM_API_KEY", "base_url": "https://api.sarvam.ai/v1", "model": "sarvam-105b"}
]

def _has_secret(key):
    try:
        return key in st.secrets and st.secrets[key].strip() != ""
    except Exception:
        return False

def ask_llm(provider, text):
    """Generic function to query any OpenAI-compatible API."""
    if not _has_secret(provider["key"]):
        return None # Silently skip if no key
        
    try:
        headers = {"HTTP-Referer": "http://localhost:8501", "X-Title": "Fact Checker"} if provider["name"] == "OpenRouter" else None
        
        client = OpenAI(
            base_url=provider["base_url"],
            api_key=st.secrets[provider["key"]],
            default_headers=headers
        )
        res = client.chat.completions.create(
            model=provider["model"],
            messages=[{
                "role": "user",
                "content": f"Is this REAL or FAKE? Return ONLY one word. Text: {text}",
            }],
        )
        verdict = res.choices[0].message.content.strip().upper()
        return -1 if "FAKE" in verdict else (1 if "REAL" in verdict else None)
    except Exception:
        return None # Silently fail (won't crash the app, won't show ugly errors)


# ============================================================
# TRUE STATE RESET
# ============================================================
def reset_app():
    for k in list(st.session_state.keys()):
        if k.startswith(("text_", "file_", "form_")):
            del st.session_state[k]
    st.session_state.form_key = 0
    st.rerun()

if "form_key" not in st.session_state:
    st.session_state.form_key = 0


# ============================================================
# UI LAYOUT
# ============================================================
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
with col_btn1:
    analyze_clicked = st.button("🚀 Deploy Analysis", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🔄 Start Over / Refresh", use_container_width=True):
        reset_app()


# ============================================================
# MAIN PROCESSING
# ============================================================
if analyze_clicked:
    input_text = ""
    media_verdict = None
    media_analyzed = False
    c2pa_result = None
    is_image = is_video = is_audio = False

    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        c2pa_result = analyze_c2pa(uploaded_file)

        if filename.endswith(".txt"):
            input_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.info("📄 **Text file loaded.**")

        elif filename.endswith((".jpg", ".jpeg", ".png")):
            is_image = True
            media_analyzed = True
            img = Image.open(uploaded_file)
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                input_text += "\n" + ocr_text # Combine OCR text with any typed text
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
            is_video = True
            media_analyzed = True
            with st.spinner("Extracting video frames..."):
                frames = extract_video_frames(uploaded_file)
            if not frames:
                st.warning("Could not extract frames.")
            else:
                fake_frames = 0
                for i, frame in enumerate(frames):
                    v_fake, v_detail = analyze_image_deepfake(frame)
                    ela = ela_manipulation_score(frame)
                    if v_fake or ela > 40:
                        fake_frames += 1
                        st.warning(f"Frame {i+1}: suspicious — {v_detail}, ELA {ela}")
                if fake_frames > 0:
                    media_verdict = True
                    st.error(f"🎥 **{fake_frames}/{len(frames)} frames show manipulation.**")
                else:
                    media_verdict = False
                    st.success(f"✅ All {len(frames)} sampled frames appear authentic.")

        elif filename.endswith((".wav", ".mp3", ".m4a", ".flac")):
            is_audio = True
            media_analyzed = True
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

    # Combine typed text with file OCR text (if both exist)
    if text_input.strip():
        input_text = text_input.strip() + "\n" + input_text

    # ---- NO INPUT CHECK ----
    if not input_text.strip() and not is_image and not is_video and not is_audio:
        st.warning("⚠️ Please paste text or upload a file.")
    else:
        clean_text = re.sub(r"http\S+", "", input_text).strip()

        if clean_text:
            st.subheader("🌐 Phase 1: Language & Source Processing")
            try:
                translated_text = GoogleTranslator(source="auto", target="en").translate(clean_text)
                if translated_text.lower() != clean_text.lower():
                    st.info(f"📝 **Translated to English:** {translated_text[:150]}...")
            except Exception:
                translated_text = clean_text

            st.subheader("🔎 Phase 2: Fact-Check Search")
            search_score, search_detail = analyze_with_search(translated_text)
            st.info(search_detail)

            # ---- AI JUDGES ----
            st.subheader("🏛️ Phase 3: Consulting AI Judges...")
            judge_votes = []
            text_for_judges = translated_text[:1500]
            api_ran = False

            for provider in LLM_PROVIDERS:
                vote = ask_llm(provider, text_for_judges)
                if vote is not None:
                    api_ran = True
                    judge_votes.append(vote)
                    if vote == -1: st.error(f"Judge {provider['name']}: Voted FAKE")
                    else: st.success(f"Judge {provider['name']}: Voted REAL")
            
            # REQUIREMENT 3: If no APIs are available, hide the section entirely
            if not api_ran:
                st.info("ℹ️ No API keys found in Secrets. Using only local models and web search for consensus.")

            # ---- CONSENSUS ----
            st.divider()
            st.subheader("🏆 Final Consensus Verdict")
            local_fake = media_verdict is True
            search_fake = (search_score is not None and search_score < 0)
            judge_fake = (judge_votes.count(-1) > judge_votes.count(1)) if judge_votes else False

            media_weight = 2 if media_analyzed else 0
            search_weight = 1 if search_score is not None else 0
            judge_weight = 3 if judge_votes else 0

            fake_score = ((media_weight if local_fake else 0) + (search_weight if search_fake else 0) + (judge_weight if judge_votes and judge_fake else 0))
            total_weight = media_weight + search_weight + judge_weight

            # THE BUG FIX: If total_weight is 0, we cannot give a verdict
            if total_weight == 0:
                st.warning("⚠️ **INCONCLUSIVE:** No models or search results could process this content. Please check your input or API keys.")
            else:
                final_is_fake = (fake_score > (total_weight / 2))
                confidence = f"Local: {'FAKE' if local_fake else ('REAL' if media_analyzed else 'n/a')} | Search: {'FAKE' if search_fake else ('REAL' if search_score is not None else 'n/a')} | Judges: {judge_votes.count(-1)}F / {judge_votes.count(1)}R"

                if c2pa_result and c2pa_result.get("status") == "Valid" and c2pa_result.get("is_ai"):
                    final_is_fake = True
                    confidence += " | C2PA: AI flag → forced FAKE"

                if final_is_fake:
                    st.error(f"🚨 **VERDICT: LIKELY FAKE / MISLEADING**\n\n{confidence}")
                else:
                    st.success(f"✅ **VERDICT: LIKELY REAL / AUTHENTIC**\n\n{confidence}")

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

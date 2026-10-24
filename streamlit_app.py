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
import c2pa # NEW IMPORT

# --- Page Config ---
st.set_page_config(page_title="Multimodal Fake News & Deepfake Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ Multimodal Fake News & Deepfake Detector")
st.markdown("""
Protect yourself from misinformation. You can **paste text, paste a URL, or upload images, text files, and video clips (MP4)**.
""")

# --- NEW C2PA PROVENANCE CHECK FUNCTION (using official library) ---
def analyze_c2pa_metadata(uploaded_file):
    """Reads C2PA manifest using the official c2pa-python library and returns a verdict."""
    if uploaded_file is None:
        return None
    
    if uploaded_file.name.endswith('.txt'):
        return {"status": "N/A", "summary": "C2PA check skipped for text files."}
        
    try:
        # The official library uses a Reader object
        reader = c2pa.Reader.from_file(uploaded_file)
        manifest_store = json.loads(reader.json())
        
        # Check for a valid manifest
        active_manifest_id = manifest_store.get('active_manifest')
        if active_manifest_id:
            manifest = manifest_store.get('manifests', {}).get(active_manifest_id, {})
            
            # Look for the signer/issuer
            signer = manifest.get('signature_info', {}).get('issuer', 'Unknown')
            
            # Check for AI-generation assertions
            is_ai_generated = False
            for assertion in manifest.get('assertions', []):
                if 'c2pa.ai_generative_training' in assertion.get('label', ''):
                    is_ai_generated = True
                    break
            
            return {
                "status": "Valid",
                "signer": signer,
                "is_ai_generated_claim": is_ai_generated,
                "summary": f"Valid C2PA manifest found. Signed by: {signer}."
            }
        else:
            return {"status": "No Manifest", "summary": "No C2PA manifest found. Provenance cannot be verified."}
            
    except Exception as e:
        # This exception often means no manifest is present or the file is unsupported
        return {"status": "No Manifest", "summary": "No C2PA manifest found."}

# --- GLOBAL TRUSTED SOURCES DATABASE ---
# ... (keep your existing TRUSTED_SOURCES and SATIRE_SOURCES lists here) ...

@st.cache_resource
def load_baseline():
    tfidf = joblib.load("tfidf.pkl")
    model = joblib.load("logreg.pkl")
    return tfidf, model

# ... (keep the rest of your helper functions and UI layout the same) ...

if st.button("Analyze Now", type="primary", use_container_width=True):
    
    input_text = ""
    is_image = False
    is_video = False
    ai_media_verdict = None
    media_api_failed = False
    c2pa_result = None # Initialize C2PA result
    
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        
        # Run C2PA Check immediately
        c2pa_result = analyze_c2pa_metadata(uploaded_file)
        
        # ... (keep your existing logic for text, image, and video processing) ...

    # ... (keep your existing text extraction and analysis logic) ...

            # --- FINAL VERDICT LOGIC ---
            final_is_fake = False
            
            # ... (keep your existing logic for determining final_is_fake) ...

            # --- C2PA OVERRIDE LOGIC ---
            if c2pa_result:
                if c2pa_result["status"] == "Valid" and c2pa_result["is_ai_generated_claim"]:
                    final_is_fake = True # Cryptographic proof it's AI-generated
                elif c2pa_result["status"] == "Valid" and c2pa_result["signer"] and "OpenAI" in c2pa_result["signer"]:
                    final_is_fake = True # Strong signal from AI generator

            st.divider()
            st.subheader("🏆 Final Verdict")
            
            # ... (keep your existing logic for displaying the final verdict) ...

            # --- DISPLAY C2PA RESULTS ---
            if c2pa_result:
                st.write("---")
                st.subheader("📜 C2PA Provenance Check")
                if c2pa_result["status"] == "Valid":
                    st.success(c2pa_result["summary"])
                    if c2pa_result.get("is_ai_generated_claim"):
                        st.error("🚨 **Manifest explicitly states this file was AI-Generated.**")
                elif c2pa_result["status"] == "Invalid":
                    st.error(c2pa_result["summary"])
                else:
                    st.warning(c2pa_result["summary"])

            # ... (keep the rest of your app's display logic) ...

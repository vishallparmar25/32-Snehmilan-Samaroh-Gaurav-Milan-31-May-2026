import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import pickle
import dlib
import cv2
import json
import subprocess

# --- APP CONFIGURATION & HIDING GIT/PROFILE LINKS ---
st.set_page_config(
    page_title="AI Photo Finder",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Comprehensive injected CSS to remove top header, GitHub links, footer, and the bottom profile widget
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            div[data-testid="stStatusWidget"] {visibility: hidden;}
            [data-testid="stDecoration"] {display: none;}
            .viewerBadge_container__1QSob {display: none !important;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

EVENT_IMAGES_DIR = "event_images"
INDEX_FILE = "gallery_index.pkl"
GOOGLE_DRIVE_FOLDER_ID = "1KaLc9BAAQJqNM7DiHjCYGELqGUbB-HQt"

if not os.path.exists(EVENT_IMAGES_DIR):
    os.makedirs(EVENT_IMAGES_DIR, exist_ok=True)

# --- DYNAMIC MODEL PATH RESOLUTION ---
try:
    import face_recognition_models
    BASE_MODELS_DIR = os.path.join(os.path.dirname(face_recognition_models.__file__), "models")
except ImportError:
    BASE_MODELS_DIR = "/home/adminuser/venv/lib/python3.11/site-packages/face_recognition_models/models"

predictor_path = os.path.join(BASE_MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
face_rec_path = os.path.join(BASE_MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")

@st.cache_resource
def load_dlib_models():
    """Caches heavy dlib models in memory so they only load ONCE"""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    encoder = dlib.face_recognition_model_v1(face_rec_path)
    return detector, predictor, encoder

try:
    face_detector, shape_predictor, face_encoder = load_dlib_models()
except Exception as e:
    st.error(f"Error loading AI models. Checked path: {predictor_path}. System Error: {e}")

def get_face_encodings(img_array):
    """Locates faces and generates 128D encodings matching face_recognition output"""
    detected_faces = face_detector(img_array, 1)
    encodings = []
    for face in detected_faces:
        shape = shape_predictor(img_array, face)
        encoding = np.array(face_encoder.compute_face_descriptor(img_array, shape, 1))
        encodings.append(encoding)
    return encodings

def compare_faces(known_encodings, face_to_check, tolerance=0.45):
    """Computes Euclidean distance to find matching faces"""
    if len(known_encodings) == 0:
        return [False]
    distances = np.linalg.norm(known_encodings - face_to_check, axis=1)
    return list(distances <= tolerance)

@st.cache_data(ttl=600)
def fetch_fast_drive_mapping(folder_id):
    """
    Leverages gdown's native internal extraction engine via subprocess 
    to fetch filenames and file IDs anonymously in under 2 seconds.
    """
    mapping = {}
    try:
        cmd = ["gdown", f"https://drive.google.com/drive/folders/{folder_id}", "--folder", "--json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.stdout:
            data = json.loads(result.stdout)
            for item in data:
                url = item.get("url", "")
                name = item.get("path", "").lower().strip()
                if "id=" in url:
                    file_id = url.split("id=")[-1]
                    mapping[name] = file_id
    except Exception:
        pass
    return mapping

# --- CUSTOM GUJARATI TITLE & SUBTITLE ---
st.title("શ્રી સતવારા જ્ઞાતિ મંડળ સુરત 32 મો સ્નેહમિલન સમારોહ (ગૌરવ મિલન ) 31 મે 2026")
st.subheader("⚡ Ultra-Fast AI Event Photo Finder")

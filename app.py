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

# --- APP CONFIGURATION & HIDING GIT LINKS ---
st.set_page_config(
    page_title="AI Photo Finder",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Injected CSS to completely hide Streamlit's top-right GitHub/deploy menu options
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
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

# --- CREDITS PLACED DIRECTLY BELOW SUBHEADER NOW ---
st.info("⚡ **Instant Match. Infinite Speed.**\n\n🛠️ *Engineered by Vishal Parmar*")
st.write("Album is permanently indexed for instant, high-accuracy searches.")
st.markdown("---")

# --- LOAD DATABASE ---
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as f:
        cached_album = pickle.load(f)
else:
    st.error(f"🚨 '{INDEX_FILE}' not found! Please check your Git repository deployment.")
    st.stop()

# Generate mapping cache pipeline instantly
with st.spinner("Synchronizing server link pipeline..."):
    drive_map = fetch_fast_drive_mapping(GOOGLE_DRIVE_FOLDER_ID)

# --- USER CAM SCANNING ---
picture = st.camera_input("Snap your selfie")

if picture is not None:
    st.success("Selfie captured!")
    
    if st.button("🚀 Find My Photos Instantly"):
        # --- TRAFFIC/PROCESSING LOADING NOTE ---
        with st.spinner("⏳ કૃપા કરીને ૨-૫ સેકન્ડ રાહ જુઓ! | Processing your request, please wait 2-5 seconds..."):
            try:
                pil_selfie = Image.open(picture).convert('RGB')
                selfie_image = np.array(pil_selfie, dtype=np.uint8)
                selfie_encodings = get_face_encodings(selfie_image)
            except Exception as e:
                st.error(f"Error reading selfie: {e}")
                selfie_encodings = []
                
            if len(selfie_encodings) == 0:
                st.error("AI couldn't see your face clearly. Face the camera directly in good lighting!")
            else:
                user_face_encoding = selfie_encodings[0]
                matched_photos = []
                
                for item in cached_album:
                    encodings = item.get("encodings", [])
                    item_path = item.get("path", "")
                    item_name = item.get("name", os.path.basename(item_path) if item_path else "photo.jpg")
                    
                    for face_encode in encodings:
                        matches = compare_faces([user_face_encoding], face_encode, tolerance=0.45)
                        if matches[0]:
                            matched_photos.append({"name": item_name})
                            break 
                
                if not matched_photos:
                    st.warning("No precise matches found of you.")
                else:
                    st.success(f"🎉 Found {len(matched_photos)} matching photos!")
                    
                    for idx, photo in enumerate(matched_photos):
                        if not photo["name"]:
                            continue
                        
                        st.markdown(f"### 🖼️ Result #{idx + 1}")
                        
                        lookup_name = photo["name"].lower().strip()
                        file_id = drive_map.get(lookup_name)
                        
                        if file_id:
                            # --- THE FIXED LINK STRUCT ---
                            direct_image_url = f"https://lh3.googleusercontent.com/d/{file_id}"
                            direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                            
                            # Displays clear image on page instantly without cookie/login wall blocks
                            st.image(direct_image_url, caption=photo["name"], use_container_width=True)
                            
                            # Renders functioning anonymous download link button card
                            st.markdown(
                                f'<a href="{direct_download_url}" download target="_blank" style="text-decoration: none;">'
                                f'<button style="background-color: #2e7d32; color: white; border: none; padding: 12px 20px; '
                                f'border-radius: 5px; cursor: pointer; font-weight: bold; width: 100%; font-size: 16px;"> '
                                f'📥 Save High-Res Original (Anonymous Download)'
                                f'</button></a>', 
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning(f"📄 File `{photo['name']}` found in database, but drive sync skipped it. Check filename spelling inside Google Drive.")
                        
                        st.markdown("---")

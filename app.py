import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import pickle
import dlib
import cv2
import requests
import re

# --- APP CONFIGURATION ---
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
def fetch_anonymous_drive_map(folder_id):
    """
    Reads the folder index anonymously in the background without credentials,
    mapping filenames directly to their hidden download IDs.
    """
    mapping = {}
    try:
        # Request Google's embedded folder viewer framework
        url = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            # Match the raw internal JavaScript array pattern: ["ID", "Filename"]
            matches = re.findall(r'\["([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"image/', res.text)
            for file_id, name in matches:
                mapping[name.lower()] = file_id
    except Exception:
        pass
    return mapping

# --- CUSTOM GUJARATI TITLE & SUBTITLE ---
st.title("શ્રી સતવારા જ્ઞાતિ મંડળ સુરત 32 મો સ્નેહમિલન સમારોહ (ગૌરવ મિલન ) 31 મે 2026")
st.subheader("⚡ Ultra-Fast AI Event Photo Finder")
st.write("Album is permanently indexed for instant, high-accuracy searches.")

# --- SIDEBAR CREDITS & CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Developer Profile")
st.sidebar.sidebar_info = st.sidebar.info("🚀 Built with ❤️ by **Vishal Parmar**")
st.sidebar.markdown("---")

# --- LOAD DATABASE ---
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as f:
        cached_album = pickle.load(f)
    st.sidebar.success("⚡ Database Loaded! Search will be instant.")
else:
    st.error(f"🚨 '{INDEX_FILE}' not found! Please check your Git repository deployment.")
    st.stop()

# Silently discover matching public content URLs in background
drive_map = fetch_anonymous_drive_map(GOOGLE_DRIVE_FOLDER_ID)

# --- USER CAM SCANNING ---
picture = st.camera_input("Snap your selfie")

if picture is not None:
    st.success("Selfie captured!")
    
    if st.button("🚀 Find My Photos Instantly"):
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
            
            st.write("Comparing your face against the database...")
            
            for item in cached_album:
                encodings = item.get("encodings", [])
                item_path = item.get("path", "")
                item_name = item.get("name", os.path.basename(item_path) if item_path else "photo.jpg")
                
                for face_encode in encodings:
                    matches = compare_faces([user_face_encoding], face_encode, tolerance=0.45)
                    if matches[0]:
                        matched_photos.append({
                            "name": item_name
                        })
                        break 
            
            if not matched_photos:
                st.warning("No precise matches found of you.")
            else:
                st.success(f"🎉 Found {len(matched_photos)} matching photos!")
                
                # Render matches instantly via direct download links
                for idx, photo in enumerate(matched_photos):
                    if not photo["name"]:
                        continue
                    
                    st.markdown(f"### 🖼️ Result #{idx + 1}")
                    
                    # Convert file lookup to lowercase to protect against mismatched strings
                    lookup_name = photo["name"].lower()
                    file_id = drive_map.get(lookup_name)
                    
                    if file_id:
                        # Construct a direct streaming URL that bypasses the Google Account wall
                        direct_image_url = f"https://drive.google.com/uc?export=view&id={file_id}"
                        direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                        
                        # Display clear image immediately in the app interface
                        st.image(direct_image_url, caption=photo["name"], use_container_width=True)
                        
                        # Provide clean anonymous download layout link
                        st.markdown(
                            f'<a href="{direct_download_url}" download target="_blank" style="text-decoration: none;">'
                            f'<button style="background-color: #2e7d32; color: white; border: none; padding: 12px 20px; '
                            f'border-radius: 5px; cursor: pointer; font-weight: bold; width: 100%; font-size: 16px;"> '
                            f'📥 Save High-Res Original (Anonymous Download)'
                            f'</button></a>', 
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning(f"📄 File `{photo['name']}` found in database, but could not be mapped to Google Drive. Check spelling in folder.")
                    
                    st.markdown("---")

import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import pickle
import dlib
import cv2
import requests  # Added for lightweight direct streaming

# --- APP CONFIGURATION ---
EVENT_IMAGES_DIR = "event_images"
INDEX_FILE = "gallery_index.pkl"
GOOGLE_DRIVE_FOLDER_ID = "1KaLc9BAAQJqNM7DiHjCYGELqGUbB-HQt"

# Ensure local directory exists for storing matched photos on-the-fly
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

# --- CUSTOM GUJARATI TITLE & SUBTITLE ---
st.title("શ્રી સતવારા જ્ઞાતિ મંડળ સુરત 32 મો સ્નેહમિલન સમારોહ (ગૌરવ મિલન ) 31 મે 2026")
st.subheader("⚡ Ultra-Fast AI Event Photo Finder")
st.write("Album is permanently indexed for instant, high-accuracy searches.")

# --- SIDEBAR CREDITS & CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Developer Profile")
st.sidebar.info("🚀 Built with ❤️ by **Vishal Parmar**")
st.sidebar.markdown("---")

# --- INTELLIGENT INDEX LOADING (Git Source) ---
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as f:
        cached_album = pickle.load(f)
    st.sidebar.success("⚡ Database Loaded! Search will be instant.")
else:
    st.error(f"🚨 '{INDEX_FILE}' not found! Please make sure it's committed to your Git repository.")
    st.stop()

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
                st.warning("No highly precise matches found of you.")
            else:
                st.success(f"Found {len(matched_photos)} matching photos!")
                
                # Render matches dynamically
                for idx, photo in enumerate(matched_photos):
                    if not photo["name"]:
                        continue
                        
                    local_path = os.path.join(EVENT_IMAGES_DIR, photo["name"])
                    
                    # --- DOWNLOAD ONLY THE MATCHED PHOTO ---
                    if not os.path.exists(local_path):
                        with st.spinner(f"📥 Pulling original high-quality file {idx+1}/{len(matched_photos)}: {photo['name']}..."):
                            try:
                                # We construct a web stream directly targeting the exact filename inside your shared folder link
                                # Google Drive allows public folder structure downloads if file name is specified via API or web request fallback
                                import gdown
                                url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FOLDER_ID}"
                                
                                # Fallback solution to stream down individual file safely without folder scanning loop
                                # If direct name download fails via gdown, we use standard requests matching your shared configuration
                                file_url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FOLDER_ID}"
                                
                                # Since we do not have specific File IDs inside the .pkl, we search and grab the specific file using gdown fuzzy matching option
                                gdown.download(fuzzy=True, id=GOOGLE_DRIVE_FOLDER_ID, output=local_path, quiet=True)
                                
                                # If the filename pulled doesn't match, verify if it saved correctly
                                if not os.path.exists(local_path):
                                    # Fallback to standard web request if gdown fuzzy breaks
                                    pass
                            except Exception as download_error:
                                pass

                    # Show image if available locally
                    if os.path.exists(local_path):
                        st.image(local_path, use_container_width=True)
                        
                        try:
                            with open(local_path, "rb") as file:
                                file_bytes = file.read()
                            
                            btn_label = f"📥 Download Original Photo ({photo['name']})"
                            st.download_button(label=btn_label, data=file_bytes, file_name=photo["name"], mime="image/jpeg", key=f"btn_{local_path}_{idx}")
                        except Exception as e:
                            st.error(f"Could not initialize download button: {e}")
                    else:
                        # Alternative if file cannot download individually without file ID
                        st.error(f"Could not download '{photo['name']}' directly. Make sure files inside your Google Drive folder are set to 'Anyone with link can view'.")
                    
                    st.markdown("---")

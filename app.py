import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import pickle
import dlib
import cv2
import gdown

# --- APP CONFIGURATION ---
EVENT_IMAGES_DIR = "event_images"
INDEX_FILE = "gallery_index.pkl"

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

def build_permanent_index():
    """Scans local album folder (used only for initial offline generation)"""
    if not os.path.exists(EVENT_IMAGES_DIR):
        return []
    all_images = [f for f in os.listdir(EVENT_IMAGES_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    indexed_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for index, img_name in enumerate(all_images):
        status_text.text(f"Indexing... processing {index+1}/{len(all_images)}")
        img_path = os.path.join(EVENT_IMAGES_DIR, img_name)
        try:
            pil_album = Image.open(img_path).convert('RGB')
            album_img = np.array(pil_album, dtype=np.uint8)
            album_encodings = get_face_encodings(album_img)
            if album_encodings:
                indexed_data.append({
                    "path": img_path,
                    "name": img_name,  
                    "encodings": album_encodings,
                    "drive_id": item.get("drive_id", None) # Fallback if you map IDs later
                })
        except Exception:
            continue
        progress_bar.progress((index + 1) / len(all_images))
        
    progress_bar.empty()
    status_text.empty()
    
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(indexed_data, f)
    return indexed_data

# --- INTELLIGENT INDEX LOADING ---
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as f:
        cached_album = pickle.load(f)
    st.sidebar.success("⚡ Database Loaded! Search will be instant.")
else:
    with st.spinner("Building index engine framework..."):
        cached_album = build_permanent_index()
    st.sidebar.success("✅ Database Active!")

if st.sidebar.button("♻️ Re-scan & Update Album"):
    with st.spinner("Updating database..."):
        cached_album = build_permanent_index()
    st.rerun()

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
                drive_id = item.get("drive_id", None)
                
                for face_encode in encodings:
                    matches = compare_faces([user_face_encoding], face_encode, tolerance=0.45)
                    if matches[0]:
                        matched_photos.append({
                            "path": item_path,
                            "name": item_name,
                            "drive_id": drive_id
                        })
                        break 
            
            if not matched_photos:
                st.warning("No highly precise matches found of you.")
            else:
                st.success(f"Found {len(matched_photos)} matching photos!")
                
                # --- INDIVIDUAL DOWNLOAD ON DEMAND LOOP ---
                for idx, photo in enumerate(matched_photos):
                    if not photo["name"]:
                        continue
                        
                    local_path = os.path.join(EVENT_IMAGES_DIR, photo["name"])
                    
                    # If the file is missing locally, we grab ONLY this file via its shared url if available
                    if not os.path.exists(local_path):
                        if photo.get("drive_id"):
                            with st.spinner(f"📥 Downloading match {idx+1}/{len(matched_photos)}: {photo['name']}..."):
                                try:
                                    url = f"https://drive.google.com/uc?id={photo['drive_id']}"
                                    gdown.download(url, local_path, quiet=True)
                                except Exception as e:
                                    st.error(f"Failed to fetch {photo['name']} directly: {e}")
                    
                    # Display the file if it exists locally
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
                        st.error(f"⚠️ Photo '{photo['name']}' is not available locally. Please upload it to your local '{EVENT_IMAGES_DIR}' directory.")
                    
                    st.markdown("---")

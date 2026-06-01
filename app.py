import setuptools  # Forces Streamlit to acknowledge the package presence
import pkg_resources  # Pre-loads the missing module into container memory
import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import face_recognition_models
import dlib

# Load the pre-trained face detectors from the models package
FACE_DETECTOR = dlib.get_frontal_face_detector()
POSE_PREDICTOR = dlib.shape_predictor(face_recognition_models.face_recognition_model_location())
FACE_ENCODER = dlib.face_recognition_model_v1(face_recognition_models.face_recognition_model_location())

EVENT_IMAGES_DIR = "event_images"
INDEX_FILE = "gallery_index.pkl"

st.title("⚡ Ultra-Fast AI Event Photo Finder")
st.write("Album is permanently indexed for instant, high-accuracy searches.")

def get_face_encodings(img_array):
    """Bypasses face_recognition package to calculate face vectors directly using dlib models"""
    try:
        # Detect face bounding boxes
        faces = FACE_DETECTOR(img_array, 1)
        encodings = []
        for face in faces:
            # Get landmarks and generate 128D face vector
            shape = POSE_PREDICTOR(img_array, face)
            encoding = np.array(FACE_ENCODER.compute_face_descriptor(img_array, shape, 1))
            encodings.append(encoding)
        return encodings
    except Exception:
        return []

def build_permanent_index():
    """Scans the album folder exactly ONCE and saves the face models to a file"""
    if not os.path.exists(EVENT_IMAGES_DIR):
        return []
        
    all_images = [f for f in os.listdir(EVENT_IMAGES_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    indexed_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for index, img_name in enumerate(all_images):
        status_text.text(f"Indexing album... processing {index+1}/{len(all_images)}")
        img_path = os.path.join(EVENT_IMAGES_DIR, img_name)
        
        try:
            pil_album = Image.open(img_path).convert('RGB')
            album_img = np.array(pil_album, dtype=np.uint8)
            album_encodings = get_face_encodings(album_img)
            
            if album_encodings:
                indexed_data.append({
                    "path": img_path,
                    "name": img_name,
                    "encodings": album_encodings
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
    with st.spinner("First-time setup: Building high-precision index file..."):
        cached_album = build_permanent_index()
    st.sidebar.success("✅ Database Created!")

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
                for face_encode in item["encodings"]:
                    # Euclidean distance check (tolerance=0.45 matches your original setup)
                    dist = np.linalg.norm(user_face_encoding - face_encode)
                    if dist <= 0.45:
                        matched_photos

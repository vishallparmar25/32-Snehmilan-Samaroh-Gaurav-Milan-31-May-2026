import streamlit as st
import os
import requests
from PIL import Image
import numpy as np
import pickle
import dlib

# --- AUTOMATIC FACE DETECTOR WEIGHTS DOWNLOADER ---
MODEL_DIR = "models"
PREDICTOR_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
RECOGNITION_PATH = os.path.join(MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat")

@st.cache_resource
def initialize_dlib_models():
    """Downloads required facial matrix files if they aren't already present"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    urls = {
        PREDICTOR_PATH: "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat",
        RECOGNITION_PATH: "https://github.com/stefanopini/simple-Face-Recognition/raw/master/models/dlib_face_recognition_resnet_model_v1.dat"
    }
    
    for path, url in urls.items():
        if not os.path.exists(path):
            with st.spinner(f"Downloading required AI engine asset: {os.path.basename(path)}..."):
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    st.error(f"Failed to fetch model file from source repository.")
                    st.stop()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    encoder = dlib.face_recognition_model_v1(RECOGNITION_PATH)
    return detector, predictor, encoder

# Initialize the structural engines securely
FACE_DETECTOR, POSE_PREDICTOR, FACE_ENCODER = initialize_dlib_models()

EVENT_IMAGES_DIR = "event_images"
INDEX_FILE = "gallery_index.pkl"

st.title("⚡ Ultra-Fast AI Event Photo Finder")
st.write("Album is permanently indexed for instant, high-accuracy searches.")

def get_face_encodings(img_array):
    """Calculates high-precision 128D face vectors safely without legacy packaging wrappers"""
    try:
        faces = FACE_DETECTOR(img_array, 1)
        encodings = []
        for face in faces:
            shape = POSE_PREDICTOR(img_array, face)
            encoding = np.array(FACE_ENCODER.compute_face_descriptor(img_array, shape, 1))
            encodings.append(encoding)
        return encodings
    except Exception:
        return []

def build_permanent_index():
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

# --- INTLLIGENT INDEX LOADING ---
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
                    dist = np.linalg.norm(user_face_encoding - face_encode)
                    if dist <= 0.45:
                        matched_photos.append({
                            "path": item["path"],
                            "name": item.get("name", os.path.basename(item["path"]))
                        })
                        break 
            
            if not matched_photos:
                st.warning("No highly precise matches found of you.")
            else:
                st.success(f"Found {len(matched_photos)} matching photos!")
                
                for photo in matched_photos:
                    st.image(photo["path"], use_container_width=True)
                    try:
                        with open(photo["path"], "rb") as file:
                            file_bytes = file.read()
                            
                        st.download_button(
                            label=f"📥 Download Original Photo ({photo['name']})",
                            data=file_bytes,
                            file_name=photo["name"],
                            mime="image/jpeg",
                            key=photo["path"]
                        )
                    except Exception as e:
                        st.error(f"Could not load download button: {e}")
                    
                    st.markdown("---")

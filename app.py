import streamlit as st
import face_recognition
import os
from PIL import Image
import numpy as np
import pickle

EVENT_IMAGES_DIR = "event_images"
INDEX_FILE = "gallery_index.pkl"

st.title("⚡ Ultra-Fast AI Event Photo Finder")
st.write("Album is permanently indexed for instant, high-accuracy searches.")

def build_permanent_index():
    """Scans the album folder exactly ONCE and saves the face models to a file"""
    if not os.path.exists(EVENT_IMAGES_DIR):
        return []
        
    all_images = [f for f in os.listdir(EVENT_IMAGES_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    indexed_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for index, img_name in enumerate(all_images):
        status_text.text(f"Indexing album for instant access... processing {index+1}/{len(all_images)}")
        img_path = os.path.join(EVENT_IMAGES_DIR, img_name)
        
        try:
            pil_album = Image.open(img_path).convert('RGB')
            album_img = np.array(pil_album, dtype=np.uint8)
            
            album_encodings = face_recognition.face_encodings(album_img)
            
            if album_encodings:
                indexed_data.append({
                    "path": img_path,
                    "name": img_name,  # Saved to use as the download filename
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
            selfie_encodings = face_recognition.face_encodings(selfie_image)
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
                    matches = face_recognition.compare_faces([user_face_encoding], face_encode, tolerance=0.45)
                    if matches[0]:
                        # Save both path and name for the download feature
                        matched_photos.append({
                            "path": item["path"],
                            "name": item.get("name", os.path.basename(item["path"]))
                        })
                        break 
            
            if not matched_photos:
                st.warning("No highly precise matches found of you.")
            else:
                st.success(f"Found {len(matched_photos)} matching photos!")
                
                # --- NEW: DISPLAY IMAGE + DOWNLOAD BUTTON ---
                for photo in matched_photos:
                    # Show preview
                    st.image(photo["path"], use_container_width=True)
                    
                    # Read original file bytes to pass to download button
                    try:
                        with open(photo["path"], "rb") as file:
                            file_bytes = file.read()
                            
                        st.download_button(
                            label=f"📥 Download Original Photo ({photo['name']})",
                            data=file_bytes,
                            file_name=photo["name"],
                            mime="image/jpeg",
                            key=photo["path"]  # Unique key for Streamlit rendering
                        )
                    except Exception as e:
                        st.error(f"Could not load download button for this image: {e}")
                    
                    # Add a visual separator between photos
                    st.markdown("---")
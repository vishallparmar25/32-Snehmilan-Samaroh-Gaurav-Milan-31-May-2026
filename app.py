import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Face Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert PIL image to OpenCV format
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        cv_image = img_array.copy()

    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Convert back to RGB to display in Streamlit
    result_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    st.image(result_image, caption=f"Detected {len(faces)} face(s)", use_container_width=True)

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set the Gujarati event title with styling
st.markdown(
    """
    <h2 style='text-align: center; color: #E65100; font-family: sans-serif; margin-bottom: 20px;'>
        શ્રી સતવારા જ્ઞાતિ મંડળ સુરત <br> 
        32 મો સ્નેહમિલન સમારોહ (ગૌરવ મિલન) <br> 
        31 મે 2026
    </h2>
    """, 
    unsafe_allow_html=True
)

# 1. Camera Input for Selfie
picture = st.camera_input("Take a selfie! / સેલ્ફી લો!")

if picture is not None:
    # Open the captured image
    image = Image.open(picture)
    img_array = np.array(image)
    
    # Convert RGB (Streamlit/PIL format) to BGR (OpenCV format)
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
    
    # Draw boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
    # Convert back to RGB to display it correctly
    result_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Display the result
    st.image(result_image, caption=f"Detected {len(faces)} face(s) in your selfie!", use_container_width=True)

# ---------------------------------------------------------
# Footer Section to display your name
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 14px;">
        Build by <strong>Vishal Parmar</strong>
    </div>
    """, 
    unsafe_allow_html=True
)

import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle

# Use dlib's native binding or stable face_recognition hooks
import dlib
# If your code explicitly calls face_recognition, we can use dlib's face detector directly:
import face_recognition_models

EVENT_IMAGES_DIR = "event_images"
INDEX_FILE = "gallery_index.pkl"
# ... rest of your code stays exactly the same

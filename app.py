@st.cache_resource
def initialize_dlib_models():
    """Downloads required facial matrix files safely from open direct-link mirrors"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    # Using open, unauthenticated direct storage mirrors that do not throw 401 errors
    urls = {
        PREDICTOR_PATH: "https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2",
        RECOGNITION_PATH: "https://raw.githubusercontent.com/davisking/dlib-models/master/dlib_face_recognition_resnet_model_v1.dat.bz2"
    }
    
    import bz2  # Standard Python library to unpack compressed models instantly
    
    for path, url in urls.items():
        # Check if the uncompressed file already exists
        if not os.path.exists(path):
            bz2_path = path + ".bz2"
            with st.spinner(f"Downloading required AI engine asset: {os.path.basename(path)}..."):
                try:
                    headers = {"User-Agent": "Mozilla/5.0"}
                    response = requests.get(url, headers=headers, stream=True, timeout=60)
                    
                    if response.status_code == 200:
                        # Write the compressed file down first
                        with open(bz2_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                                    
                        # Decompress it inline instantly
                        with open(path, "wb") as new_file, bz2.BZ2File(bz2_path, "rb") as decompressed:
                            for data in iter(lambda: decompressed.read(100 * 1024), b""):
                                new_file.write(data)
                                
                        # Clean up the compressed archive download to save space
                        if os.path.exists(bz2_path):
                            os.remove(bz2_path)
                    else:
                        st.error(f"Mirror failed: Status code {response.status_code}. Retrying connection...")
                        st.stop()
                except Exception as e:
                    st.error(f"Network error while extracting files: {str(e)}")
                    st.stop()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    encoder = dlib.face_recognition_model_v1(RECOGNITION_PATH)
    return detector, predictor, encoder

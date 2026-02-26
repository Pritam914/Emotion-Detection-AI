import streamlit as st
import cv2
import numpy as np
# Error Fix: Using tensorflow.keras for better compatibility
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# UI Configuration
st.set_page_config(page_title="Emotion AI - Pritam Kumar", layout="centered")
st.title("🎭 Real-Time Emotion Recognition")
st.subheader("Deep Learning based Facial Expression Analysis")

# Load model and cascade with caching
@st.cache_resource
def setup_resources():
    # Ensure these filenames match exactly what you uploaded to GitHub
    model = load_model("emotion_detection.h5")
    # Adding .xml extension if it was missing
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = setup_resources()
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

# Processing logic (Robust and Optimized)
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        # Verbose=0 keeps the logs clean
        prediction = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(prediction)]
        
        # Visualizing the results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# WebRTC Transformer for Cloud Environment
class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img = process_frame(img)
        return processed_img

# Sidebar Navigation
option = st.sidebar.radio("Navigation", ["Home", "Live Webcam", "Upload Image"])

if option == "Home":
    st.write(f"### Hello Pritam!")
    st.markdown("""
    This project uses a **Convolutional Neural Network (CNN)** to analyze facial expressions.
    - **Step 1:** Select 'Live Webcam' or 'Upload Image'.
    - **Step 2:** The model identifies the face using Haar Cascades.
    - **Step 3:** The CNN predicts the emotion from 5 different classes.
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*6S3t69d6jX6_O1J0U-Y8_Q.png", caption="AI Emotion Mapping")

elif option == "Live Webcam":
    st.write("### 🎥 Real-Time Detection")
    st.info("Ensure you are in a well-lit environment for better accuracy.")
    # WebRTC is essential for Cloud deployment to access user camera
    webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionTransformer)

elif option == "Upload Image":
    st.write("### 📸 Image Analysis")
    file = st.file_uploader("Choose a photo...", type=['jpg', 'png', 'jpeg'])
    if file:
        img = Image.open(file)
        img_array = np.array(img.convert('RGB'))
        # OpenCV works with BGR, so we convert before processing
        processed = process_frame(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        # Convert back to RGB for Streamlit display
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Processed Result")
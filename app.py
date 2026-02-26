import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
# Keras error fix: Import through tensorflow
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# UI Setup
st.set_page_config(page_title="Emotion AI - Pritam Kumar", layout="centered")
st.title("🎭 Real-Time Emotion Recognition")
st.subheader("Deep Learning based Facial Expression Analysis")

# Load model and cascade safely
@st.cache_resource
def setup_resources():
    # Make sure 'emotion_detection.h5' is in the same folder on GitHub
    model = load_model("emotion_detection.h5")
    
    # Cascade loading fix: Check exact filename in your GitHub repo
    # If your file doesn't have .xml at the end, remove it from the string below
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = setup_resources()
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

# Detection Logic
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(prediction)]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# WebRTC Streamer for Cloud Deployment
class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img = process_frame(img)
        return processed_img

# UI Navigation
option = st.sidebar.radio("Navigation", ["Home", "Live Webcam", "Upload Image"])

if option == "Home":
    st.write("### Welcome Pritam!")
    st.markdown("Select an option from the sidebar to test the **CNN Model**.")
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*6S3t69d6jX6_O1J0U-Y8_Q.png")

elif option == "Live Webcam":
    st.write("### 🎥 Live Video Stream")
    # This solves the camera hardware access issue on cloud servers
    webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionTransformer)

elif option == "Upload Image":
    st.write("### 📸 Static Image Analysis")
    file = st.file_uploader("Upload a photo", type=['jpg', 'png', 'jpeg'])
    if file:
        img = Image.open(file)
        img_array = np.array(img.convert('RGB'))
        processed = process_frame(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Analysis Result")
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# UI Configuration
st.set_page_config(page_title="Emotion AI - Pritam Kumar", layout="centered")
st.title("🎭 Real-Time Emotion Recognition")
st.subheader("Deep Learning based Facial Expression Analysis")

# Load model and cascade
@st.cache_resource
def setup_resources():
    # compile=False helps avoid training-related errors on cloud
    model = load_model("emotion_detection.h5", compile=False)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = setup_resources()
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

# Core logic for processing each frame
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

# WebRTC Transformer class for LIVE Camera
class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        processed_img = process_frame(img)
        return processed_img

# Sidebar Options
option = st.sidebar.radio("Navigation", ["Home", "Live Webcam", "Upload Image"])

if option == "Home":
    st.write("### Welcome Pritam!")
    st.write("Go to 'Live Webcam' to test real-time detection.")

elif option == "Live Webcam":
    st.write("### 🎥 Live Stream")
    st.info("Allow camera access in your browser to start.")
    # This is the MAGIC part that works on Cloud
    webrtc_streamer(
        key="emotion-detection", 
        video_transformer_factory=EmotionTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

elif option == "Upload Image":
    file = st.file_uploader("Upload a face photo", type=['jpg', 'png', 'jpeg'])
    if file:
        img = Image.open(file)
        img_array = np.array(img.convert('RGB'))
        # Convert to BGR for processing and back to RGB for display
        processed = process_frame(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Analysis Result")

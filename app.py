import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer
import av

# Global settings for stability
cv2.setNumThreads(0)

st.set_page_config(page_title="Emotion AI - Pritam Kumar")
st.title("🎭 Real-Time Emotion Recognition")

@st.cache_resource
def load_all():
    # Model load without training metadata to save RAM
    model = load_model("emotion_detection.h5", compile=False)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = load_all()
labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
        roi = np.expand_dims(np.expand_dims(roi, -1), 0)
        
        # Predict
        pred = model.predict(roi, verbose=0)
        emotion = labels[np.argmax(pred)]
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI Logic
st.sidebar.title("Settings")
mode = st.sidebar.selectbox("Choose Mode", ["Live Webcam", "Upload Image"])

if mode == "Live Webcam":
    # The only way to keep cam active on cloud
    webrtc_streamer(
        key="emotion", 
        video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
else:
    file = st.file_uploader("Upload photo", type=['jpg', 'png'])
    if file:
        st.image(file)

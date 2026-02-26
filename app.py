import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import gc

# Memory management for stable execution
gc.collect()
cv2.setNumThreads(0)

st.set_page_config(page_title="Emotion AI - Pritam Kumar", layout="centered")
st.title("🎭 Emotion Recognition System")

# RTC Configuration for stable cloud connection
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def load_resources():
    try:
        # Loading without compiling to save RAM
        model = load_model("emotion_detection.h5", compile=False)
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        return model, cascade
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, face_cascade = load_resources()
labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        label = labels[np.argmax(prediction)]
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main Interface
st.write("### 🎥 Live Real-Time Analysis")
st.info("Ensure proper lighting for the CNN to extract features correctly.")



webrtc_streamer(
    key="emotion-ai",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

import os
# Must be at the very top
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Global settings for stability
cv2.setNumThreads(0)

st.set_page_config(page_title="Emotion AI - Pritam Kumar")
st.title("🎭 Real-Time Emotion Recognition")

# Using Google STUN server for better connectivity
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def load_all():
    model = load_model("emotion_detection.h5", compile=False)
    # Using the local file you have in GitHub
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = load_all()
labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

        pred = model.predict(roi, verbose=0)
        emotion = labels[np.argmax(pred)]
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.sidebar.title("Settings")
mode = st.sidebar.selectbox("Choose Mode", ["Live Webcam", "Upload Image"])

if mode == "Live Webcam":
    webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    file = st.file_uploader("Upload photo", type=['jpg', 'png'])
    if file:
        st.image(file)

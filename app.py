import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os

# System optimization
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# Patch to handle Keras 3 metadata error for purani .h5 files
import keras
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI - Pritam Kumar", layout="centered")
st.title("🎭 Real-Time Emotion Recognition")
st.subheader("Deep Learning based Facial Expression Analysis")

# Load model and cascade
@st.cache_resource
def setup_resources():
    model = load_model("emotion_detection.h5", custom_objects={"Dense": PatchedDense}, compile=False)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = setup_resources()
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(prediction)]
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI Logic
option = st.sidebar.radio("Navigation", ["Home", "Live Webcam", "Upload Image"])

# --- Connectivity Fix: Adding Multiple STUN Servers Pool ---

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]}
)

if option == "Live Webcam":
    st.info("Directly accessing HP TrueVision HD Camera. If it takes too long, try switching to a Mobile Hotspot.")
    
    webrtc_streamer(
        key="emotion-ai-final",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION, 
        video_frame_callback=callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 20}
            },
            "audio": False
        },
        async_processing=True,
    )

elif option == "Upload Image":
    file = st.file_uploader("Upload a face photo", type=['jpg', 'png', 'jpeg'])
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        processed_img = callback(av.VideoFrame.from_ndarray(img, format="bgr24")).to_ndarray(format="bgr24")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Analysis Result")

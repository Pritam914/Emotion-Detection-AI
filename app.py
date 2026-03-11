import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os

# System level fixes
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# --- THE BRUTE FORCE PATCH ---
# This overrides the internal Keras Dense layer loading logic globally
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        # Nuke the problematic keys from metadata before they hit the real Dense layer
        kwargs.pop('quantization_config', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI - Pritam Kumar", layout="centered")
st.title("🎭 Real-Time Emotion Recognition")

@st.cache_resource
def load_all():
    try:
        # Register PatchedDense globally so Sequential model uses it
        custom_objects = {"Dense": PatchedDense}
        
        model = keras.models.load_model(
            "emotion_model.keras", 
            custom_objects=custom_objects, 
            compile=False, 
            safe_mode=False
        )
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        return model, cascade
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, face_cascade = load_all()
labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    if model is None: return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

        # Predicting
        pred = model.predict(roi, verbose=0)
        emotion = labels[np.argmax(pred)]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI Logic - Optimized WebRTC
# 
webrtc_streamer(
    key="emotion-ai",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]
    },
    # REMOVED camera selection dropdown - strictly auto camera access
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)

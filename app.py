import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
from pathlib import Path

# Fix for system threads and buffers
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# Keras metadata patch
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI - Pritam Kumar", layout="centered")
st.title("🎭 Real-Time Emotion Recognition")

@st.cache_resource
def load_all():
    BASE_DIR = Path(__file__).resolve().parent
    model = None
    # Auto-scan root directory for any model file
    files = list(BASE_DIR.glob("*.keras")) + list(BASE_DIR.glob("*.h5"))
    
    if files:
        target_path = files[0]
        try:
            model = keras.models.load_model(
                str(target_path), 
                custom_objects={"Dense": PatchedDense}, 
                compile=False, 
                safe_mode=False
            )
            st.sidebar.success(f"✅ Auto-Detected: {target_path.name}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = load_all()

if model is None:
    st.error("❌ Model file (.keras or .h5) not found in root directory!")
    st.stop()

labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)
        pred = model.predict(roi, verbose=0)
        emotion = labels[np.argmax(pred)]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# UNIVERSAL AUTO-CAMERA (Direct access)
webrtc_streamer(
    key="emotion-ai-universal",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=callback,
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

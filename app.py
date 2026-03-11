import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os

# Performance and Compatibility
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# Patch to handle Keras metadata issues
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI - Pritam Kumar", layout="centered")
st.title("🎭 Real-Time Emotion Recognition")

@st.cache_resource
def load_all():
    # Try multiple filenames in case of mismatch
    filenames = ["emotion_model.keras", "emotion_detection.h5", "model.h5"]
    model = None
    
    for fname in filenames:
        if os.path.exists(fname):
            try:
                model = keras.models.load_model(
                    fname, 
                    custom_objects={"Dense": PatchedDense}, 
                    compile=False, 
                    safe_mode=False
                )
                break
            except:
                continue
    
    # Load Face Detector
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if not os.path.exists("haarcascade_frontalface_default.xml"):
         # Fallback to local if xml exists in repo
         cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
         
    return model, cascade

model, face_cascade = load_all()

if model is None:
    st.error("❌ Model File Not Found! Please check if your .keras or .h5 file is in the main folder of GitHub.")

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

        pred = model.predict(roi, verbose=0)
        emotion = labels[np.argmax(pred)]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Universal WebRtc (No Device Selection dropdown)
webrtc_streamer(
    key="emotion-universal",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=callback,
    async_processing=True,
    # This hides the extra device UI and forces auto-start
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)

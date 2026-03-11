import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import os

# Set environment variables
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# --- THE ULTIMATE PATCH ---
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI - Pritam Kumar")
st.title("🎭 Real-Time Emotion Recognition")

@st.cache_resource
def load_all():
    # FIX: Corrected the try-except block structure
    try:
        from keras.src.saving import serialization_lib
        serialization_lib.add_rewrite_data("Dense", PatchedDense)
        
        model = keras.models.load_model(
            "emotion_model.keras", 
            custom_objects={"Dense": PatchedDense}, 
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
    if model is None or face_cascade is None: 
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
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

# UI Settings
webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

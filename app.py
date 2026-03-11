import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import os

# Performance Tuning
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# Keras Metadata Error Patch
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
    # SCANNING ALL FILES IN THE DIRECTORY
    all_files = os.listdir('.')
    st.sidebar.write("Found Files:", all_files) # Ye debug ke liye help karega
    
    # Model dhoondne ka flexible tareeka
    model_files = [f for f in all_files if f.endswith('.keras') or f.endswith('.h5')]
    
    model = None
    if model_files:
        target = model_files[0]
        try:
            model = keras.models.load_model(
                target, 
                custom_objects={"Dense": PatchedDense}, 
                compile=False, 
                safe_mode=False
            )
            st.sidebar.success(f"✅ Loaded: {target}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = load_all()

if model is None:
    st.error("❌ Model file still not found in GitHub Root Folder.")
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

webrtc_streamer(
    key="emotion-final",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

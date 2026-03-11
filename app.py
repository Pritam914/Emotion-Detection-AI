import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import os
from pathlib import Path

# Fix for Protocol Buffers and Threads
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# Keras 3 Patch for Metadata
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI - Pritam Kumar")
st.title("🎭 Real-Time Emotion Recognition")

@st.cache_resource
def load_all():
    # Streamlit Cloud ke liye absolute path nikalna
    BASE_DIR = Path(__file__).resolve().parent
    model_path = BASE_DIR / "emotion_model.keras"
    cascade_path = BASE_DIR / "haarcascade_frontalface_default.xml"
    
    model = None
    if model_path.exists():
        try:
            model = keras.models.load_model(
                str(model_path), 
                custom_objects={"Dense": PatchedDense}, 
                compile=False, 
                safe_mode=False
            )
            st.sidebar.success(f"✅ Model Loaded: {model_path.name}")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    else:
        # Emergency Scan if path fails
        files = list(BASE_DIR.glob("*.keras")) + list(BASE_DIR.glob("*.h5"))
        if files:
            model = keras.models.load_model(str(files[0]), custom_objects={"Dense": PatchedDense}, compile=False, safe_mode=False)
            st.sidebar.warning(f"⚠️ Loaded via scan: {files[0].name}")

    cascade = cv2.CascadeClassifier(str(cascade_path))
    return model, cascade

model, face_cascade = load_all()

if model is None:
    st.error("❌ Model file still not found. Check your GitHub root directory!")
    st.stop()

labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
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

webrtc_streamer(
    key="emotion-ai-final",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import os

# Protobuf and CPU optimizations
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# --- PATCH FOR QUANTIZATION ERROR ---
# Ye class unknown 'quantization_config' ko filter kar degi
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI - Pritam Kumar", layout="centered")
st.title("🎭 Real-Time Emotion Recognition")

# Load model and face detector
@st.cache_resource
def load_all():
    # Register our patched layer as a custom object
    custom_objects = {"Dense": PatchedDense}
    
    model = keras.models.load_model(
        "emotion_model.keras", 
        custom_objects=custom_objects, 
        compile=False, 
        safe_mode=False 
    )
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = load_all()

labels = {
    0: "Angry",
    1: "Happy",
    2: "Neutral",
    3: "Sad",
    4: "Surprised"
}

# Emotion detection function
def detect_emotion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

        # Predicting
        pred = model.predict(roi, verbose=0)
        emotion = labels[np.argmax(pred)]

        # Drawing UI
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            img, emotion, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
    return img

# Webcam callback
def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = detect_emotion(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI Logic
st.sidebar.title("Settings")
mode = st.sidebar.selectbox("Choose Mode", ["Live Webcam", "Upload Image"])

if mode == "Live Webcam":
    webrtc_streamer(
        key="emotion",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )
else:
    file = st.file_uploader("Upload photo", type=["jpg", "png"])
    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = detect_emotion(img)
        st.image(img, channels="BGR", use_container_width=True)

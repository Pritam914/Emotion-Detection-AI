import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
from collections import deque

# System optimization
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# Keras Patch for legacy .h5 files
import keras
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

# UI Styling
st.set_page_config(page_title="Emotion AI | Pritam", layout="centered")

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 5px; display: flex; overflow-x: auto; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130; border-radius: 5px; color: white; 
        padding: 8px 12px; white-space: nowrap; min-width: 100px;
    }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b !important; }
    div[data-testid="stSidebarNav"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎭 Emotion Recognition AI")

@st.cache_resource
def setup_resources():
    model = load_model("emotion_detection.h5", custom_objects={"Dense": PatchedDense}, compile=False)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = setup_resources()
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

# Buffer for temporal smoothing (Stabilizes fluctuations)
if 'emotion_buffer' not in st.session_state:
    st.session_state.emotion_buffer = deque(maxlen=5)

def process_emotion(frame, smoothing=False):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, fw, fh) in faces:
        fc = gray[y:y+fh, x:x+fw]
        roi = cv2.resize(fc, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        
        if smoothing:
            st.session_state.emotion_buffer.append(prediction)
            avg_prediction = np.mean(st.session_state.emotion_buffer, axis=0)
            label_idx = np.argmax(avg_prediction)
        else:
            label_idx = np.argmax(prediction)
            
        label = emotion_labels[label_idx]
        
        # UI visibility logic (prevents label cutting)
        text_y = y - 20 if y - 20 > 40 else y + fh + 40
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 4)
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        
    return frame

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed = process_emotion(img, smoothing=True)
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

tab_home, tab_live, tab_upload = st.tabs(["🏠 Home Info", "🎥 Live Detect", "📤 Upload Img"])

with tab_home:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    with col2:
        st.subheader("Developer: Pritam")
        st.write("Specialization: CSE (AIML)")
        st.write("System: Real-time Emotion Analysis")

with tab_live:
    # Direct Camera Access Configuration
    webrtc_streamer(
        key="emotion-direct-access",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {
                "facingMode": "user",
                "width": {"ideal": 1280},
                "height": {"ideal": 720}
            },
            "audio": False
        },
        async_processing=True,
    )

with tab_upload:
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        with st.spinner('Analyzing...'):
            processed_img = process_emotion(img, smoothing=False)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            _, img_encoded = cv2.imencode('.jpg', processed_img)
            st.download_button("📥 Download Result Image", data=img_encoded.tobytes(), 
                               file_name="emotion_detected.jpg", mime="image/jpeg")

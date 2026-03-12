import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os

# Performance optimizations
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# Keras Patch for Deserialization
import keras
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI | Pritam", layout="centered")

# Professional Dark Theme UI
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130; border-radius: 8px; color: white; padding: 10px 15px;
    }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b !important; }
    div[data-testid="stSidebarNav"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎭 Real-Time Emotion Recognition")

@st.cache_resource
def setup_resources():
    model = load_model("emotion_detection.h5", custom_objects={"Dense": PatchedDense}, compile=False)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = setup_resources()
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

# Color Map (BGR)
color_map = {
    0: (0, 0, 255),    # Red
    1: (0, 255, 0),    # Green
    2: (255, 255, 255),# White
    3: (255, 0, 0),    # Blue
    4: (0, 255, 255)   # Yellow
}

def process_frame(frame):
    # --- Fix 1: Image Standardization to prevent OpenCV crashes ---
    h, w = frame.shape[:2]
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(60, 60))
    
    # Responsive Font Settings
    thickness = max(2, int(w / 400))
    font_scale = max(0.7, w / 700)
    
    for (x, y, fw, fh) in faces:
        roi = gray[y:y+fh, x:x+fw]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        idx = np.argmax(prediction)
        label = emotion_labels[idx]
        color = color_map[idx]
        
        text_y = y - 15 if y - 15 > 30 else y + fh + 40
        
        # --- Fix 2: High-Contrast Text (Outline/Shadow) ---
        # Draw black outline first for visibility
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 2)
        # Draw actual colored text over it
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, thickness)
        
    return frame

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed = process_frame(img)
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

tab_home, tab_live, tab_upload = st.tabs(["🏠 Info", "🎥 Live Camera", "📤 Upload Image"])

with tab_home:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    with col2:
        st.subheader("Emotion AI System")
        st.markdown("""
        **Features:**
        - Live facial expression analysis.
        - Support for high-resolution photo uploads.
        - Multi-face detection capability.
        - Color-coded results for better UX.
        """)

with tab_live:
    webrtc_streamer(
        key="emotion-ai-v3",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {"facingMode": "user", "width": {"ideal": 640}, "height": {"ideal": 480}},
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
            processed_img = process_frame(img)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            _, img_encoded = cv2.imencode('.jpg', processed_img)
            st.download_button("📥 Download Analysis", data=img_encoded.tobytes(), 
                               file_name="emotion_result.jpg", mime="image/jpeg")

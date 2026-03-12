import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os

# System level optimizations for speed
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# Keras 3 Patch - Global Registration
import keras
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI | Pritam Kumar", layout="centered")

# Professional UI Styling - Minimalist and Fast
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130; border-radius: 10px; color: white; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎭 Real-Time Emotion Recognition")

@st.cache_resource
def setup_resources():
    # Loading original model files
    model = load_model("emotion_detection.h5", custom_objects={"Dense": PatchedDense}, compile=False)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = setup_resources()
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

# Color mapping for different emotions (BGR)
color_map = {
    0: (0, 0, 255),    # Red (Angry)
    1: (0, 255, 0),    # Green (Happy)
    2: (255, 255, 255),# White (Neutral)
    3: (255, 0, 0),    # Blue (Sad)
    4: (0, 255, 255)   # Yellow (Surprised)
}

def process_frame(frame, is_live=True):
    # Responsive font and box scaling
    h, w = frame.shape[:2]
    thickness = max(3, int(w / 300))
    font_scale = max(1.0, w / 500)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Optimized detection parameters
    faces = face_cascade.detectMultiScale(gray, 1.2, 6, minSize=(60, 60))
    
    for (x, y, fw, fh) in faces:
        roi_gray = gray[y:y+fh, x:x+fw]
        roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
        
        prediction = model.predict(roi_gray, verbose=0)
        idx = np.argmax(prediction)
        label = emotion_labels[idx]
        color = color_map[idx]
        
        # UI visibility logic (prevents text cutting)
        text_y = y - 20 if y - 20 > 40 else y + fh + 50
        
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, thickness)
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        
    return frame

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed = process_frame(img, is_live=True)
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

# --- Tabs for Cleaner Navigation ---
tab_home, tab_live, tab_upload = st.tabs(["🏠 Home", "🎥 Live Detect", "📤 Upload"])

with tab_home:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=150)
    with col2:
        st.subheader("Developer: Pritam Kumar")
        st.write("Degree: B.Tech (CSE AIML)")
        st.markdown("""
        **System Specs:**
        - Model: Deep CNN (FER2013)
        - Framework: TensorFlow 2.16+
        - Engine: OpenCV + WebRTC
        """)

with tab_live:
    st.info("Directly accessing front camera. Click Start below.")
    # Robust STUN pool for faster handshake
    # 
    webrtc_streamer(
        key="emotion-ai-fast",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
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
        with st.spinner('Analyzing Image...'):
            processed_img = process_frame(img, is_live=False)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            _, img_encoded = cv2.imencode('.jpg', processed_img)
            st.download_button("📥 Download Analysis", data=img_encoded.tobytes(), 
                               file_name="emotion_result.jpg", mime="image/jpeg")

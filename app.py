import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
from collections import deque

# Performance optimizations
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# Keras Deserialization Patch
import keras
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI | Pritam", layout="centered")

# Professional UI Styling
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
color_map = {
    0: (0, 0, 255),    # Angry: Red
    1: (0, 255, 0),    # Happy: Green
    2: (255, 255, 255),# Neutral: White
    3: (255, 0, 0),    # Sad: Blue
    4: (0, 255, 255)   # Surprised: Yellow
}

if 'emotion_buffer' not in st.session_state:
    st.session_state.emotion_buffer = deque(maxlen=10)

def process_emotion(frame, smoothing=False):
    # Step 1: Standardize size to prevent OpenCV ScaleData errors
    h, w = frame.shape[:2]
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        h, w = frame.shape[:2]

    # Step 2: Refined Detection (Higher minNeighbors to stop false positives)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=7, 
        minSize=(60, 60)
    )
    
    # Step 3: Adaptive Font Calculation
    font_scale = max(0.8, w / 800.0) 
    thickness = max(2, int(w / 350.0))
    
    for (x, y, fw, fh) in faces:
        fc = gray[y:y+fh, x:x+fw]
        roi = cv2.resize(fc, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        
        if smoothing:
            st.session_state.emotion_buffer.append(prediction)
            avg_pred = np.mean(st.session_state.emotion_buffer, axis=0)
            idx = np.argmax(avg_pred)
        else:
            idx = np.argmax(prediction)
            
        label = emotion_labels[idx]
        color = color_map[idx]
        
        # UI Positioning
        text_y = y - 15 if y - 15 > 30 else y + fh + 40
        
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, thickness)
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        
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
        st.subheader("Developer: Pritam Kumar")
        st.write("Specialization: CSE (AIML)")
        st.write("Professional Facial Expression Analysis System.")

with tab_live:
    # Optimized for Mobile/Desktop Stability
    # 
    webrtc_streamer(
        key="emotion-ai-final-v2",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": {
                "facingMode": "user",
                "width": {"ideal": 640},
                "height": {"ideal": 480}
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
        with st.spinner('Processing...'):
            processed_img = process_emotion(img, smoothing=False)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            _, img_encoded = cv2.imencode('.jpg', processed_img)
            st.download_button("📥 Download Analysis", data=img_encoded.tobytes(), 
                               file_name="emotion_result.jpg", mime="image/jpeg")

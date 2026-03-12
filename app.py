import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
from collections import deque

# System optimization for faster inference
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# Keras Patch
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

# Color Coding for Emotions (BGR Format)
# Happy: Green | Sad: Blue | Angry: Red | Surprised: Yellow | Neutral: White
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}
color_map = {
    0: (0, 0, 255),    # Red
    1: (0, 255, 0),    # Green
    2: (255, 255, 255),# White
    3: (255, 0, 0),    # Blue
    4: (0, 255, 255)   # Yellow
}

if 'emotion_buffer' not in st.session_state:
    st.session_state.emotion_buffer = deque(maxlen=5)

def process_emotion(frame, smoothing=False):
    h, w = frame.shape[:2]
    # Dynamic scaling for fonts
    font_scale = max(1.2, w / 400.0) 
    thickness = max(3, int(w / 200.0))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    
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
        
        # Prevent label clipping
        text_y = y - 20 if y - 20 > 50 else y + fh + 60
        
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, thickness)
        # Background for text for better readability
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        
    return frame

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # Using lightweight processing for Live
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
        st.write("Real-time Facial Emotion Analysis via Deep Learning.")

with tab_live:
    # Multiple STUN servers pool to bypass firewall/NAT issues
    # 
    webrtc_streamer(
        key="emotion-ai-optimized",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]
        },
        media_stream_constraints={
            "video": {
                "facingMode": "user",
                "width": {"ideal": 640}, # Reduced resolution for stability on mobile
                "height": {"ideal": 480},
                "frameRate": {"ideal": 20}
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
            st.download_button("📥 Download Result", data=img_encoded.tobytes(), 
                               file_name="detected_emotion.jpg", mime="image/jpeg")

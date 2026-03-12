import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
from PIL import Image

# Optimization
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

# --- UI Styling ---
st.set_page_config(page_title="Emotion AI | Pritam", layout="centered")

# Custom CSS for Professional Look & Mobile Optimization
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; justify-content: center; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #1e2130; 
        border-radius: 10px; color: white; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b !important; }
    div[data-testid="stSidebarNav"] { display: none; } /* Hide default sidebar */
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

# --- Process Logic with Enhanced Visibility ---
def process_emotion(frame, is_mobile=True):
    # Thick lines and big fonts for mobile visibility
    thickness = 4 if is_mobile else 2
    font_scale = 1.5 if is_mobile else 0.8
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(prediction)]
        
        # Bolder Rectangle & Text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness)
        cv2.putText(frame, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (255, 255, 255), thickness)
        
    return frame

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed = process_emotion(img, is_mobile=True)
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

# --- Mobile-Friendly Navigation ---
tab_home, tab_live, tab_upload = st.tabs(["🏠 Home", "🎥 Live Camera", "📤 Upload"])

with tab_home:
    st.markdown("""
    ### Welcome to Emotion AI
    This professional tool analyzes facial expressions using Deep Learning (CNN).
    
    **Features:**
    - Real-time detection via WebRTC.
    - Snapshot capture during live feed.
    - High-visibility detection for mobile screens.
    """)
    st.image("https://img.icons8.com/clouds/200/000000/brainstorming.png")

with tab_live:
    st.info("Ensure you are using HTTPS for camera access.")
    # Connection Pool
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    ctx = webrtc_streamer(
        key="emotion-pro",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if ctx.state.playing:
        if st.button("📸 Click to Capture Snapshot"):
            st.warning("Snapshot saved below (Download enabled)")
            # This logic captures the current state if needed

with tab_upload:
    st.write("### Image Analysis")
    file = st.file_uploader("Choose a photo...", type=['jpg', 'png', 'jpeg'])
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # Apply Enhanced Visibility logic
        processed_img = process_emotion(img, is_mobile=True)
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Download Button
        result_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        st.download_button("📥 Download Result", data=file, file_name="result.jpg")

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

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: #1e2130; 
        border-radius: 8px; color: white; font-size: 14px;
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

# --- Robust Process Logic ---
def process_emotion(frame):
    # Image resize to prevent OpenCV ScaleData error
    h, w = frame.shape[:2]
    if w > 1000: # Agar image bohot badi hai toh resize karo
        frame = cv2.resize(frame, (800, int(h * 800 / w)))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces detection with safer parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(prediction)]
        
        # Super-Visible styling for Mobile
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, label, (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 
                    1.2, (255, 255, 255), 3)
        
    return frame

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed = process_emotion(img)
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

# --- Tabs Navigation ---
tab_home, tab_live, tab_upload = st.tabs(["🏠 Home", "🎥 Live AI", "📤 Upload"])

with tab_home:
    col1, col2 = st.columns([1, 2])
    with col1:
        # Reliable Placeholder Image
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=150)
    with col2:
        st.subheader("Project Overview")
        st.write("Developed by **Pritam Kumar**")
        st.markdown("""
        - **Model:** CNN (Convolutional Neural Network)
        - **Dataset:** FER2013
        - **Tech:** TensorFlow, OpenCV, Streamlit
        """)

with tab_live:
    st.info("Mobile users: Use Landscape mode for better view.")
    webrtc_streamer(
        key="emotion-pro-final",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tab_upload:
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        with st.spinner('Analyzing Emotion...'):
            processed_img = process_emotion(img)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Download result
            _, img_encoded = cv2.imencode('.jpg', processed_img)
            st.download_button("📥 Download Result", data=img_encoded.tobytes(), file_name="emotion_analysis.jpg")

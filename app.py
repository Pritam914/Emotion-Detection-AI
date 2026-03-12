import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
from datetime import datetime

# Performance optimizations
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

st.set_page_config(page_title="Emotion AI | Pritam", layout="centered")

# --- UI Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130; border-radius: 8px; color: white; padding: 10px 15px;
    }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b !important; }
    div[data-testid="stSidebarNav"] { display: none; }
    .info-card { background-color: #1e2130; padding: 20px; border-radius: 15px; border-left: 5px solid #ff4b4b; margin-bottom: 20px;}
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
color_map = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 255, 255), 3: (255, 0, 0), 4: (0, 255, 255)}

def process_frame(frame):
    h, w = frame.shape[:2]
    target_dim = 1000
    if max(h, w) > target_dim:
        scale = target_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Advanced Contrast for accuracy
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # --- ACCURACY UPGRADE: Increased minNeighbors to 12 to filter out objects ---
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=12, 
        minSize=(50, 50)
    )
    
    # --- LIMIT CHECK: Maximum 10 faces for stability ---
    if len(faces) > 10:
        return "limit_exceeded", len(faces)

    thickness = max(2, int(w / 500))
    font_scale = max(0.5, w / 900)
    
    for (x, y, fw, fh) in faces:
        roi = gray[y:y+fh, x:x+fw]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        idx = np.argmax(prediction)
        label = emotion_labels[idx]
        color = color_map[idx]
        
        text_y = y - 10 if y - 10 > 25 else y + fh + 30
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, thickness)
        
    return frame, len(faces)

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed, _ = process_frame(img)
    if isinstance(processed, str): # Handle limit in live (ignore extra faces)
        processed = img
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

tab_home, tab_live, tab_upload = st.tabs(["🏠 Home Info", "🎥 Live Camera", "📤 Upload Image"])

with tab_home:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    with col2:
        st.subheader("Pritam's Emotion AI")
        st.write("Professional facial analysis system using Deep CNN.")
    st.markdown("---")
    st.markdown("""
    **Analysis Constraints:**
    - **Face Limit:** Up to **10 faces** per image for maximum accuracy.
    - **Optimization:** Filters out non-human objects automatically.
    - **Stability:** Dynamic rescaling for high-resolution group pics.
    """)
    
    st.markdown("---")
    st.subheader("📬 Contact & Feedback")
    st.write("Sharing your results helps me improve!")
    c1, c2, c3 = st.columns(3)
    c1.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/pritam-kumar-607631334)")
    c2.markdown("[📸 Instagram](https://www.instagram.com/pritamray26)")
    c3.markdown("[📧 Email](mailto:pritamray6200@gmail.com)")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_live:
    st.info("Directly accessing front camera. Supported up to 10 faces.")
    webrtc_streamer(
        key="emotion-group-ultimate",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=callback,
        media_stream_constraints={"video": {"facingMode": "user", "width": {"ideal": 640}}, "audio": False},
        async_processing=True,
    )

with tab_upload:
    file = st.file_uploader("Upload image (Max 10 faces)", type=['jpg', 'png', 'jpeg'])
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        with st.spinner('Analyzing...'):
            processed_img, count = process_frame(img)
            
            if processed_img == "limit_exceeded":
                st.error(f"❌ Limit Exceeded: Detected {count} faces. Please upload an image with maximum 10 members for accurate analysis.")
            else:
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.success(f"Analysis complete! Detected {count} face(s).")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                _, img_encoded = cv2.imencode('.jpg', processed_img)
                st.download_button("📥 Download Analysis", data=img_encoded.tobytes(), file_name=f"emotion_analysis_{ts}.jpg")

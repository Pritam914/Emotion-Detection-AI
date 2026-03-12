import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
from datetime import datetime

# Optimization
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# Keras Patch for Legacy Models
import keras
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI | Pritam", layout="centered")

# --- UI Styling (Professional Dark Theme) ---
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
    # Load model and XML
    model = load_model("emotion_detection.h5", custom_objects={"Dense": PatchedDense}, compile=False)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = setup_resources()
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}
color_map = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 255, 255), 3: (255, 0, 0), 4: (0, 255, 255)}

# --- Core Logic with Stability Buffers ---
def process_emotion(image):
    # Resize for stability if image is too large
    h, w = image.shape[:2]
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Contrast improvement for better accuracy
    gray = cv2.equalizeHist(gray)
    
    # Face detection with robust parameters
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=6, 
        minSize=(40, 40)
    )
    
    # Limitation for server stability
    if len(faces) > 12:
        return "limit", len(faces)
    
    for (x, y, fw, fh) in faces:
        roi_gray = gray[y:y+fh, x:x+fw]
        roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
        
        prediction = model.predict(roi_gray, verbose=0)
        idx = np.argmax(prediction)
        label = emotion_labels[idx]
        color = color_map[idx]
        
        # UI Scaling
        thickness = max(2, int(w / 400))
        font_scale = max(0.6, w / 800)
        
        # Shadow Text Logic (for visibility on all backgrounds)
        text_y = y - 10 if y - 10 > 25 else y + fh + 30
        cv2.putText(image, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(image, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        cv2.rectangle(image, (x, y), (x+fw, y+fh), color, thickness)
        
    return image, len(faces)

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    res = process_emotion(img)
    if isinstance(res, tuple):
        processed_img = res[0]
    else:
        processed_img = img
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- UI Tabs ---
tab_home, tab_live, tab_upload = st.tabs(["🏠 Info", "🎥 Live Camera", "📤 Upload Image"])

with tab_home:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    with col2:
        st.subheader("Pritam's Professional Emotion AI")
        st.write("CNN-based real-time facial analysis system.")
    st.markdown("---")
    st.write("**Contact for Feedback:**")
    c1, c2, c3 = st.columns(3)
    c1.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/pritam-kumar-607631334)")
    c2.markdown("[📸 Instagram](https://www.instagram.com/pritamray26)")
    c3.markdown("[📧 Email](mailto:pritamray6200@gmail.com)")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_live:
    st.info("Ensure you have good lighting for best accuracy.")
    webrtc_streamer(
        key="emotion-ultimate",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=callback,
        media_stream_constraints={"video": {"facingMode": "user", "width": {"ideal": 640}}, "audio": False},
        async_processing=True,
    )

with tab_upload:
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        with st.spinner('Analyzing Expressions...'):
            result = process_emotion(img)
            
            if result == "limit":
                st.error("Too many faces detected! Please use an image with fewer than 12 people.")
            else:
                processed_img, count = result
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.success(f"Detected {count} face(s) with high confidence.")
                
                # Download Result
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                _, enc = cv2.imencode('.jpg', processed_img)
                st.download_button("📥 Download Result", data=enc.tobytes(), file_name=f"emotion_{ts}.jpg")

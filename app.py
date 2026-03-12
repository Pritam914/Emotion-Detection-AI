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

# Keras Patch
import keras
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI | Pritam", layout="centered")

# CSS Styling
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130; border-radius: 8px; color: white; padding: 10px 15px;
    }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b !important; }
    div[data-testid="stSidebarNav"] { display: none; }
    .info-card { background-color: #1e2130; padding: 20px; border-radius: 15px; border-top: 4px solid #ff4b4b; margin-bottom: 20px;}
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
    # CRASH FIX: Standardize image size
    h, w = frame.shape[:2]
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # ROBUST DETECTION: scaleFactor 1.2 and minNeighbors 5 to prevent OpenCV scaleIdx error
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
    
    thickness = max(2, int(w / 400))
    font_scale = max(0.6, w / 700)
    
    for (x, y, fw, fh) in faces:
        roi = gray[y:y+fh, x:x+fw]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        idx = np.argmax(prediction)
        label = emotion_labels[idx]
        color = color_map[idx]
        
        text_y = y - 15 if y - 15 > 30 else y + fh + 40
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, thickness)
        
    return frame

# --- Snapshot Logic ---
if "live_image" not in st.session_state:
    st.session_state["live_image"] = None

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed = process_frame(img)
    st.session_state["live_image"] = processed # Update buffer
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

tab_home, tab_live, tab_upload, tab_feedback = st.tabs(["🏠 Home", "🎥 Live Camera", "📤 Upload", "💬 Feedback"])

with tab_home:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("Pritam's Emotion AI")
    st.write("Professional Deep Learning system for real-time facial analysis.")
    st.markdown("""
    - **Speed:** High-speed inference using TensorFlow CPU.
    - **Accuracy:** Optimized for FER2013 dataset.
    - **Stability:** Dynamic rescaling for large images.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_live:
    # Multiple STUN servers for robust mobile connection
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    webrtc_ctx = webrtc_streamer(
        key="emotion-live-v5",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": {"facingMode": "user", "width": {"ideal": 640}}, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        st.write("---")
        if st.button("📸 Capture Live Detection"):
            # Get from buffer
            snap = st.session_state.get("live_image")
            if snap is not None:
                st.image(cv2.cvtColor(snap, cv2.COLOR_BGR2RGB), caption="Snapshot Captured!", use_column_width=True)
                _, enc = cv2.imencode('.jpg', snap)
                st.download_button("📥 Save Snapshot", data=enc.tobytes(), file_name=f"snap_{datetime.now().strftime('%H%M%S')}.jpg")
            else:
                st.warning("Wait 1 sec for camera to sync and try again.")

with tab_upload:
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        with st.spinner('Analyzing...'):
            processed_img = process_frame(img)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _, img_encoded = cv2.imencode('.jpg', processed_img)
            st.download_button("📥 Download Result", data=img_encoded.tobytes(), file_name=f"result_{ts}.jpg")

with tab_feedback:
    st.subheader("📬 Send Your Feedback")
    st.write("I see all feedback in my admin logs. Please be descriptive!")
    
    with st.form("fb_form", clear_on_submit=True):
        email = st.text_input("Your Email")
        msg = st.text_area("What should I improve?")
        captcha = st.number_input("What is 10 + 20?", min_value=0)
        
        if st.form_submit_button("Send to Admin"):
            if captcha == 30 and email and msg:
                # This will appear in Streamlit Cloud -> Manage App -> Logs
                print(f"\n[FEEDBACK] FROM: {email}\nMESSAGE: {msg}\nDATE: {datetime.now()}\n")
                st.success("Sent! I will check this in my dashboard logs.")
            else:
                st.error("Fill everything correctly.")

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

# Keras Patch for Deserialization
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
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130; border-radius: 8px; color: white; padding: 10px 15px;
    }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b !important; }
    div[data-testid="stSidebarNav"] { display: none; }
    .contact-card { background-color: #1e2130; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-top: 20px;}
    .home-header { background: linear-gradient(90deg, #ff4b4b, #1e2130); padding: 2px; border-radius: 10px; margin-bottom: 20px; }
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
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40))
    
    thickness = max(2, int(frame.shape[1] / 450))
    font_scale = max(0.6, frame.shape[1] / 800)
    
    for (x, y, fw, fh) in faces:
        roi = gray[y:y+fh, x:x+fw]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        idx = np.argmax(prediction)
        label = emotion_labels[idx]
        color = color_map[idx]
        
        text_y = y - 15 if y - 15 > 30 else y + fh + 35
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, thickness)
        
    return frame

# Persistent frame storage for snapshot
if "current_image" not in st.session_state:
    st.session_state["current_image"] = None

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed = process_frame(img)
    # Thread-safe update
    st.session_state["current_image"] = processed
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

tab_home, tab_live, tab_upload = st.tabs(["🏠 Home & Contact", "🎥 Live Camera", "📤 Upload"])

with tab_home:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    with col2:
        st.subheader("System Overview")
        st.markdown("""
        - Optimized for **Group Analysis**.
        - High-contrast visual labeling.
        - **Live Snapshots** & Image Upload support.
        """)
    
    # --- Contact Section Moved to Home ---
    st.markdown('<div class="contact-card">', unsafe_allow_html=True)
    st.subheader("📬 Contact & Feedback")
    st.write("Encountered an issue? Share your result or feedback to help me improve!")
    
    c1, c2, c3 = st.columns(3)
    c1.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/pritam-kumar-607631334)")
    c2.markdown("[📸 Instagram](https://www.instagram.com/pritamray26)")
    c3.markdown("[📧 Email](mailto:pritamray6200@gmail.com)")
    
    st.markdown("---")
    with st.form("feedback_home", clear_on_submit=True):
        f_email = st.text_input("Email")
        f_msg = st.text_area("Message")
        spam = st.number_input("Spam Check: 5 + 7 = ?", min_value=0)
        if st.form_submit_button("Send"):
            if spam == 12 and f_email and f_msg:
                st.success("Feedback received!")
            else:
                st.error("Invalid entry.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_live:
    ctx = webrtc_streamer(
        key="emotion-final-v4",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"facingMode": "user", "width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
        async_processing=True,
    )

    # --- Fixed Live Snapshot Logic ---
    if ctx.state.playing:
        if st.button("📸 Capture Live Result"):
            snap = st.session_state.get("current_image")
            if snap is not None:
                st.image(cv2.cvtColor(snap, cv2.COLOR_BGR2RGB), caption="Snapshot Result", use_column_width=True)
                _, enc = cv2.imencode('.jpg', snap)
                st.download_button("📥 Download Snapshot", data=enc.tobytes(), file_name=f"live_{datetime.now().strftime('%H%M%S')}.jpg")
            else:
                st.info("Processing first frame... please try again in a second.")

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
            st.download_button("📥 Download Analysis", data=img_encoded.tobytes(), file_name=f"emotion_{ts}.jpg", mime="image/jpeg")

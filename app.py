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
    .info-card { background-color: #1e2130; padding: 20px; border-radius: 15px; border-top: 4px solid #ff4b4b; margin-bottom: 20px;}
    .feedback-section { background-color: #161b22; padding: 25px; border-radius: 15px; border: 1px solid #30363d; }
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

# Global buffer for Snapshot
if "snapshot_buffer" not in st.session_state:
    st.session_state["snapshot_buffer"] = None

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed = process_frame(img)
    st.session_state["snapshot_buffer"] = processed
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

# --- Tabs Structure ---
tab_home, tab_live, tab_upload, tab_feedback = st.tabs(["🏠 Home", "🎥 Live Camera", "📤 Upload", "💬 Feedback"])

with tab_home:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    with col2:
        st.subheader("Pritam's Emotion AI")
        st.write("This system uses a Deep Convolutional Neural Network (CNN) to interpret human micro-expressions in real-time.")
    
    st.markdown("---")
    st.subheader("System Capabilities")
    st.markdown("""
    - **Face Tracking:** Robust Haar-Cascade multi-face detection.
    - **Express Inference:** Low-latency emotion classification (Angry, Happy, Neutral, Sad, Surprised).
    - **Stability:** Dynamic rescaling and histogram equalization for varying light conditions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_live:
    ctx = webrtc_streamer(
        key="emotion-ai-final-production",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"facingMode": "user", "width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing:
        st.markdown("---")
        if st.button("📸 Capture & Review Results"):
            snap = st.session_state.get("snapshot_buffer")
            if snap is not None:
                st.image(cv2.cvtColor(snap, cv2.COLOR_BGR2RGB), caption="Analysis Result Captured", use_column_width=True)
                _, enc = cv2.imencode('.jpg', snap)
                st.download_button("📥 Save to Device", data=enc.tobytes(), file_name=f"live_result_{datetime.now().strftime('%H%M%S')}.jpg")
            else:
                st.warning("Syncing frames... Click again in 1 second.")

with tab_upload:
    file = st.file_uploader("Choose an image file", type=['jpg', 'png', 'jpeg'])
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        with st.spinner('Analyzing expressions...'):
            processed_img = process_frame(img)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _, img_encoded = cv2.imencode('.jpg', processed_img)
            st.download_button("📥 Download Analysis", data=img_encoded.tobytes(), file_name=f"emotion_{ts}.jpg")

with tab_feedback:
    st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
    st.subheader("📩 Community Feedback")
    st.write("Your insights are vital for refining this AI. If you notice any misclassifications, please let me know!")
    
    # Contact Links
    c1, c2, c3 = st.columns(3)
    c1.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/pritam-kumar-607631334)")
    c2.markdown("[📸 Instagram](https://www.instagram.com/pritamray26)")
    c3.markdown("[📧 Email](mailto:pritamray6200@gmail.com)")
    
    st.markdown("---")
    with st.form("admin_feedback", clear_on_submit=True):
        u_email = st.text_input("Verified Email Address")
        u_msg = st.text_area("Observations or Suggestions")
        u_spam = st.number_input("Human Check: 5 + 7 = ?", min_value=0)
        
        if st.form_submit_button("Submit Feedback"):
            if u_spam == 12 and u_email and u_msg:
                # ADMIN LOGGING: You will see this in your Streamlit Cloud logs
                print(f"\n--- NEW FEEDBACK RECEIVED ---\nUser: {u_email}\nMessage: {u_msg}\nTimestamp: {datetime.now()}\n----------------------------")
                st.success("Success! I have received your message in my logs. Thank you for contributing!")
            else:
                st.error("Please complete all fields correctly.")
    st.markdown('</div>', unsafe_allow_html=True)

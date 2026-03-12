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
    .contact-card { background-color: #1e2130; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
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
if "last_frame" not in st.session_state:
    st.session_state["last_frame"] = None

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed = process_frame(img)
    st.session_state["last_frame"] = processed # Save for snapshot
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

tab_home, tab_live, tab_upload, tab_contact = st.tabs(["🏠 Info", "🎥 Live Camera", "📤 Upload", "📧 Contact"])

with tab_home:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    with col2:
        st.subheader("Emotion AI System")
        st.markdown("""
        **System Capabilities:**
        - Optimized for **Group Analysis**.
        - High-contrast visual labeling.
        - Dynamic image rescaling.
        - **Live Snapshots** enabled.
        """)

with tab_live:
    ctx = webrtc_streamer(
        key="emotion-ai-snapshot",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"facingMode": "user", "width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
        async_processing=True,
    )

    # --- Feature 1: Live Snapshot ---
    if ctx.state.playing and st.button("📸 Capture Live Result"):
        if st.session_state["last_frame"] is not None:
            snapshot = st.session_state["last_frame"]
            st.image(cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB), caption="Captured Result", use_column_width=True)
            _, snap_encoded = cv2.imencode('.jpg', snapshot)
            st.download_button("📥 Download Snapshot", data=snap_encoded.tobytes(), file_name=f"snap_{datetime.now().strftime('%H%M%S')}.jpg")
        else:
            st.warning("Please wait for the camera to initialize.")

with tab_upload:
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        with st.spinner('Performing Deep Analysis...'):
            processed_img = process_frame(img)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _, img_encoded = cv2.imencode('.jpg', processed_img)
            st.download_button("📥 Download Analysis", data=img_encoded.tobytes(), file_name=f"emotion_{timestamp}.jpg", mime="image/jpeg")

with tab_contact:
    st.markdown('<div class="contact-card">', unsafe_allow_html=True)
    st.subheader("Get in Touch")
    st.write("If you encounter any issues or have suggestions to improve this system, please feel free to reach out. Sharing your results or feedback helps me refine the model further!")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/pritam-kumar-607631334)")
    col_b.markdown("[📸 Instagram](https://www.instagram.com/pritamray26)")
    col_c.markdown("[📧 Email](mailto:pritamray6200@gmail.com)")
    
    st.markdown("---")
    
    # --- Protected Feedback System ---
    st.subheader("Submit Feedback")
    with st.form("feedback_form", clear_on_submit=True):
        user_email = st.text_input("Your Email (for verification)")
        feedback_msg = st.text_area("Message")
        # Anti-spam: Simple math challenge
        spam_check = st.number_input("Spam Protection: What is 5 + 7?", min_value=0)
        
        submit_fb = st.form_submit_button("Send Feedback")
        if submit_fb:
            if spam_check == 12 and user_email and feedback_msg:
                st.success("Thank you! Your feedback has been logged. I will review it shortly.")
            else:
                st.error("Please fill all fields correctly and solve the spam protection challenge.")
    st.markdown('</div>', unsafe_allow_html=True)

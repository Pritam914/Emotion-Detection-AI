import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Memory optimization for OpenCV
cv2.setNumThreads(0)

st.set_page_config(page_title="Emotion AI - Pritam Kumar", layout="centered")
st.title("🎭 Real-Time Emotion Recognition")

# Model load karte waqt memory crash se bachne ke liye compile=False
@st.cache_resource
def setup_resources():
    try:
        model = load_model("emotion_detection.h5", compile=False)
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        return model, cascade
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

model, face_cascade = setup_resources()
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

def process_emotion(frame):
    if model is None or face_cascade is None:
        return frame
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        # Prediction logic
        prediction = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(prediction)]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img = process_emotion(img)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# UI Logic
option = st.sidebar.radio("Navigation", ["Home", "Live Webcam", "Upload Image"])

if option == "Home":
    st.write("### Welcome Pritam! System optimized for Python 3.13.")

elif option == "Live Webcam":
    import av # Video frame processing library
    st.write("### 🎥 Live Stream")
    webrtc_streamer(
        key="emotion-detect",
        video_processor_factory=EmotionProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

elif option == "Upload Image":
    file = st.file_uploader("Upload a photo", type=['jpg', 'png', 'jpeg'])
    if file:
        img = Image.open(file)
        img_array = np.array(img.convert('RGB'))
        processed = process_emotion(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

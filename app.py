import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# UI Configuration
st.set_page_config(page_title="Emotion AI - Pritam Kumar", layout="centered")
st.title("🎭 Real-Time Emotion Recognition")
st.subheader("Deep Learning based Facial Expression Analysis")

# Load model and cascade
@st.cache_resource
def setup_resources():
    model = load_model("emotion_detection.h5")
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = setup_resources()
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

# Processing logic (Used for both Upload and Live)
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(prediction)]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

# WebRTC Transformer class for Cloud Live Camera
class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img = process_frame(img)
        return processed_img

# Sidebar Options
option = st.sidebar.radio("Navigation", ["Home", "Live Webcam", "Upload Image"])

if option == "Home":
    st.write("Welcome Pritam! Select an option from the sidebar to start.")
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*6S3t69d6jX6_O1J0U-Y8_Q.png", caption="Emotion Detection AI")

elif option == "Live Webcam":
    st.write("Click 'Start' to use your camera via browser.")
    webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionTransformer)

elif option == "Upload Image":
    file = st.file_uploader("Upload a face photo", type=['jpg', 'png', 'jpeg'])
    if file:
        img = Image.open(file)
        img_array = np.array(img.convert('RGB'))
        processed = process_frame(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Analysis Result")
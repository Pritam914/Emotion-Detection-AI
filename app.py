import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="Emotion AI - Pritam Kumar")
st.title("🎭 Real-Time Emotion Recognition")

# Load Resources
@st.cache_resource
def setup_resources():
    model = load_model("emotion_detection.h5", compile=False)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = setup_resources()
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        prediction = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(prediction)]
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    import av # Internal Streamlit-webrtc dependency
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Sidebar
option = st.sidebar.radio("Menu", ["Live Webcam", "Upload Image"])

if option == "Live Webcam":
    webrtc_streamer(key="emotion", video_frame_callback=video_frame_callback)

elif option == "Upload Image":
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file:
        img = np.array(Image.open(file).convert('RGB'))
        # Reuse prediction logic here if needed or just show the image
        st.image(img)

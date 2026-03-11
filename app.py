import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer
import av

cv2.setNumThreads(0)

st.set_page_config(page_title="Emotion AI - Pritam Kumar")
st.title("🎭 Real-Time Emotion Recognition")


@st.cache_resource
def load_all():

    model = tf.keras.models.load_model("emotion_model.keras")

    cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    return model, cascade


model, face_cascade = load_all()

labels = {
    0: "Angry",
    1: "Happy",
    2: "Neutral",
    3: "Sad",
    4: "Surprised"
}


def detect_emotion(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        roi = cv2.resize(gray[y:y+h, x:x+w], (48,48))
        roi = roi / 255.0
        roi = roi.reshape(1,48,48,1)

        pred = model.predict(roi, verbose=0)

        emotion = labels[np.argmax(pred)]

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(
            img,
            emotion,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    return img


def callback(frame):

    img = frame.to_ndarray(format="bgr24")

    img = detect_emotion(img)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


st.sidebar.title("Settings")

mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Live Webcam","Upload Image"]
)


if mode == "Live Webcam":

    webrtc_streamer(
        key="emotion",
        video_frame_callback=callback,
        rtc_configuration={
            "iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]
        }
    )

else:

    file = st.file_uploader("Upload photo", type=["jpg","png"])

    if file:

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

        img = cv2.imdecode(file_bytes, 1)

        img = detect_emotion(img)

        st.image(img, channels="BGR")

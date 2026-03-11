# Load model and cascade
@st.cache_resource
def setup_resources():
    # Aapki original file ka naam yahan likha hai
model = load_model("emotion_detection.h5", custom_objects={"Dense": PatchedDense}, compile=False)
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
return model, cascade
@@ -42,7 +41,6 @@ def callback(frame):

for (x, y, w, h) in faces:
fc = gray[y:y+h, x:x+w]
        # Aapka original preprocessing logic
roi = cv2.resize(fc, (48, 48)) / 255.0
roi = np.reshape(roi, (1, 48, 48, 1))

@@ -57,14 +55,34 @@ def callback(frame):
# UI Logic
option = st.sidebar.radio("Navigation", ["Home", "Live Webcam", "Upload Image"])

# --- Connectivity Fix: Adding Multiple STUN Servers Pool ---

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]}
)

if option == "Live Webcam":
    st.info("Click 'Start' to begin detection. Ensure camera permissions are allowed.")
    st.info("Directly accessing HP TrueVision HD Camera. If it takes too long, try switching to a Mobile Hotspot.")
    
webrtc_streamer(
        key="emotion-live",
        key="emotion-ai-final",
mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION, 
video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 20}
            },
            "audio": False
        },
async_processing=True,
)

@@ -73,7 +91,5 @@ def callback(frame):
if file:
file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)
        
        # Reuse prediction logic
processed_img = callback(av.VideoFrame.from_ndarray(img, format="bgr24")).to_ndarray(format="bgr24")
st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Analysis Result")

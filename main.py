# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
# import av
# import cv2
# from PIL import Image
# from ultralytics import YOLO
# import numpy as np

# # Load YOLOv8 model
# model = YOLO("yolov8n.pt")

# # Page config
# st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
# st.title("🚀 YOLOv8 Multi-Mode Object Detection App")

# # Sidebar menu
# option = st.sidebar.selectbox("Choose Detection Mode", ["Image", "Video", "Webcam", "Live Streaming (Desktop Only)"])

# # --- IMAGE ---
# if option == "Image":
#     st.header("📷 Image Detection")
#     uploaded_img = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
#     if uploaded_img:
#         img = Image.open(uploaded_img).convert("RGB")
#         res = model.predict(np.array(img), verbose=False)[0]
#         img_with_boxes = res.plot()
#         st.image(img_with_boxes, caption="Detected Image", use_column_width=True)

# # --- VIDEO ---
# elif option == "Video":
#     st.header("🎞️ Video Detection")
#     uploaded_vid = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
#     if uploaded_vid:
#         tfile = open("temp_video.mp4", 'wb')
#         tfile.write(uploaded_vid.read())
#         cap = cv2.VideoCapture("temp_video.mp4")
#         stframe = st.empty()

#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 break
#             res = model.predict(frame, verbose=False)[0]
#             frame = res.plot()
#             stframe.image(frame, channels="BGR")
#         cap.release()

# # --- WEBCAM ---
# elif option == "Webcam":
#     st.header("💻 Real-Time Webcam Detection")

#     class YOLOProcessor(VideoProcessorBase):
#         def recv(self, frame):
#             img = frame.to_ndarray(format="bgr24")
#             res = model.predict(img, verbose=False)[0]
#             annotated = res.plot()
#             return av.VideoFrame.from_ndarray(annotated, format="bgr24")

#     webrtc_streamer(key="webcam", mode=WebRtcMode.SENDRECV,
#                     video_processor_factory=YOLOProcessor,
#                     media_stream_constraints={"video": True, "audio": False},
#                     async_processing=True)

# # --- LIVE STREAMING (PHONE CAMERA / IP CAM) ---
# elif option == "Live Streaming (Desktop Only)":
#     st.header("📡 Live Streaming via IP Camera")
#     st.markdown("💡 **Note:** This feature is available only on desktop browsers.")
#     ip_url = st.text_input("Enter IP Camera URL (e.g. http://192.168.1.100:8080/video)")

#     if ip_url:
#         stframe = st.empty()
#         cap = cv2.VideoCapture(ip_url)

#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 st.warning("🔌 Failed to connect or stream ended.")
#                 break
#             res = model.predict(frame, verbose=False)[0]
#             frame = res.plot()
#             stframe.image(frame, channels="BGR")
#         cap.release()




import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Page config
st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title("🚀 YOLOv8 Multi-Mode Object Detection App")

# Session state to hold selected mode
if "mode" not in st.session_state:
    st.session_state.mode = "Image"

mode_list = ["Image", "Video", "Webcam", "Live Streaming (Desktop Only)"]

option = st.sidebar.selectbox(
    "Choose Detection Mode",
    mode_list,
    index=mode_list.index(st.session_state.mode),
    key="mode"
)

# Debug print (optional)
st.write("🛠️ Selected Mode:", option)

# --- IMAGE ---
if option == "Image":
    st.header("📷 Image Detection")
    uploaded_img = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        res = model.predict(np.array(img), verbose=False)[0]
        img_with_boxes = res.plot()
        st.image(img_with_boxes, caption="Detected Image", use_column_width=True)

# --- VIDEO ---
elif option == "Video":
    st.header("🎞️ Video Detection")
    uploaded_vid = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_vid:
        tfile = open("temp_video.mp4", 'wb')
        tfile.write(uploaded_vid.read())
        cap = cv2.VideoCapture("temp_video.mp4")
        stframe = st.empty()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            res = model.predict(frame, verbose=False)[0]
            frame = res.plot()
            stframe.image(frame, channels="BGR")
        cap.release()

# --- WEBCAM ---
elif option == "Webcam":
    st.header("💻 Real-Time Webcam Detection")

    start = st.checkbox("✅ Start Webcam")

    if start:
        class YOLOProcessor(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                res = model.predict(img, verbose=False)[0]
                annotated = res.plot()
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        webrtc_streamer(
            key="webcam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YOLOProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )


import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

class DummyProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Just convert to grayscale as a test
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return av.VideoFrame.from_ndarray(gray_3c, format="bgr24")

st.title("WebRTC Test")
webrtc_streamer(key="test", video_processor_factory=DummyProcessor)

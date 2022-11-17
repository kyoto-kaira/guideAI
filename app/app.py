from streamlit_webrtc import webrtc_streamer
import streamlit as st
from PIL import Image
import numpy as np
import requests
import base64
import cv2
import av


img_source = None


def encode_img(img):
    _, data = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(data).decode("utf-8")
    return img_base64


def decode_img(img_base64):
    img_data = base64.b64decode(img_base64)
    img_byte = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_byte, cv2.IMREAD_ANYCOLOR)
    return img


def preprocess(img):
    results = requests.post(
        "http://mtcnn-container:8006/detect",
        json={"img_base64": encode_img(img)}
    )

    if not results.status_code == 200:
        return img

    results = results.json()
    x0 = results["x0"]
    x1 = results["x1"]
    y0 = results["y0"]
    y1 = results["y1"]
    x = (x0 + x1) / img.shape[1] - 1.0

    results = requests.post(
        "http://animator-container:8007/moving",
        json={"x": x}
    )
    results = results.json()
    img_generated = decode_img(results["img_base64"])
    img_generated = cv2.cvtColor(img_generated, cv2.COLOR_RGB2BGR)

    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 5)
    img = cv2.resize(img, (256, 256))

    return np.hstack([img, img_generated])


def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = preprocess(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


def encode():
    requests.get("http://animator-container:8007/reset")
    
    results = requests.post(
        "http://animator-container:8007/encode",
        json={"img_base64": encode_img(img_source)}
    )


if __name__ == "__main__":
    st.title("demo")

    img_source = st.file_uploader("input image", type=["jpg", "png"])
    if img_source is not None:
        img_source = np.array(Image.open(img_source))
        encode()


    webrtc_streamer(
        key="demo",
        video_frame_callback=callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )
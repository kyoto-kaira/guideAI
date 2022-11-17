FROM python:3.8.15-slim-bullseye

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install libopencv-dev curl

RUN pip install -U pip
RUN pip install streamlit streamlit-webrtc opencv-python numpy

WORKDIR /workspace
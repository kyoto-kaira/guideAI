FROM nvcr.io/nvidia/pytorch:21.03-py3

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install libopencv-dev curl

RUN pip3 install -U pip

RUN pip3 install fastapi "uvicorn[standard]"
RUN pip3 install opencv-python==4.5.5.64

WORKDIR /workspace
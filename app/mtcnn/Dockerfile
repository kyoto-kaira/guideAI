FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install python3 python3-pip git libopencv-dev 

RUN pip3 install -U pip

RUN pip3 install fastapi "uvicorn[standard]"
RUN pip3 install opencv-python
RUN pip3 install facenet_pytorch

WORKDIR /workspace
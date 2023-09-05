FROM nvcr.io/nvidia/pytorch:21.10-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Ekaterinburg

WORKDIR /root

RUN apt-get update && apt-get -y install python3-pip ffmpeg libsm6 libxext6 libfreetype6-dev git cargo rustc

COPY requirements.txt requirements.txt

RUN python3 -m pip install --user -r requirements.txt

RUN python3 -m pip install --user --ignore-installed Pillow

RUN mkdir weights && cd weights && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

RUN git clone https://github.com/IDEA-Research/GroundingDINO && cd /root/GroundingDINO && git checkout feature/more_compact_inference_api

WORKDIR /app

CMD ["jupyter", "lab"]

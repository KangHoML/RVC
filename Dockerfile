# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

EXPOSE 7865

WORKDIR /app

COPY . .

# 환경 변수 설정 (시간대 문제 방지)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y -qq \
    ffmpeg \
    aria2 \
    software-properties-common \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 3.9 설치 (deadsnakes PPA 사용)
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && \
    apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3.9-distutils \
    curl \
    && apt-get clean

# pip 설치
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.9

# Python 대안 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --set python3 /usr/bin/python3.9

RUN python3 -m pip install --upgrade pip==24.0
RUN python3 -m pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# 모델 다운로드 (이전과 동일)
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth -d assets/pretrained_v2/ -o D40k.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth -d assets/pretrained_v2/ -o G40k.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth -d assets/pretrained_v2/ -o f0D40k.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth -d assets/pretrained_v2/ -o f0G40k.pth

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth -d assets/uvr5_weights/ -o HP2-人声vocals+非人声instrumentals.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth -d assets/uvr5_weights/ -o HP5-主旋律人声vocals+其他instrumentals.pth

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -d assets/hubert -o hubert_base.pt

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt -d assets/rmvpe -o rmvpe.pt

VOLUME [ "/app/weights", "/app/opt" ]

CMD ["python3", "infer-web.py"]
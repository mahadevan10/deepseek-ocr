# CUDA 11.8 runtime base (GPU-enabled)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (python, pip, poppler for pdf2image, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    poppler-utils \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CUDA 11.8 compatible PyTorch first
RUN pip3 install --upgrade pip && \
    pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Python deps
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# App code (and optional local model sources)
COPY DeepSeek-OCR ./models/DeepSeek-OCR
COPY . .

EXPOSE 8080
CMD ["sh", "-c", "gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT main:app"]

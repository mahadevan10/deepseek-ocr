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

# Python deps
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# App code and DeepSeek-OCR sources
COPY DeepSeek-OCR ./models/DeepSeek-OCR
COPY . .

# Expose Cloud Run port
EXPOSE 8080

# Start with gunicorn + uvicorn worker; Cloud Run provides $PORT
CMD ["sh", "-c", "gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT main:app"]

# Use a base image with CUDA 11.8 support (matches torch==2.6.0+cu118)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    git-lfs \
    libgl1-mesa-glx \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt

# Install CUDA-enabled torch and other dependencies
RUN pip3 install --upgrade pip && \
    pip3 install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir -r requirements.txt

# (Optional) Download DeepSeek-OCR model
RUN git lfs install && git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR ./models/DeepSeek-OCR

COPY main.py .

EXPOSE 8080

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "main:app"]

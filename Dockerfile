# Use a base image with CUDA support. This image is over 10GB.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set up the environment and install Python and Git LFS
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    git \
    git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the cloned DeepSeek-OCR folder
COPY DeepSeek-OCR ./models/DeepSeek-OCR

# Copy the rest of your application's code
COPY . .

# Preload the model during build to avoid runtime download delays
RUN python3 -c "import torch; from transformers import AutoModel, AutoTokenizer; model = AutoModel.from_pretrained('deepseek-ai/deepseek-ocr', trust_remote_code=True, torch_dtype=torch.bfloat16); tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-ocr', trust_remote_code=True); print('Model preloaded')"

# Expose the port (Cloud Run sets PORT dynamically)
EXPOSE 8080

# Command to run the application using uvicorn with dynamic PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]

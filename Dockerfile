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

# Copy the cloned DeepSeek-OCR folder (instead of cloning)
COPY DeepSeek-OCR ./models/DeepSeek-OCR

# Copy the rest of your application's code
COPY . .

# Expose the port the app will run on
EXPOSE 8080

# Command to run the application using Gunicorn
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "main:app"]

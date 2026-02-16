# Use NVIDIA CUDA base image with PyTorch
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES=0 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    pandoc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the package in editable mode for easier development/tweaks
RUN pip install -e .

# Expose the Web UI port
EXPOSE 8008

# Default command to run the server
# Binding to 0.0.0.0 is necessary for Docker
CMD ["clarityocr", "serve", "--host", "0.0.0.0", "--port", "8008"]

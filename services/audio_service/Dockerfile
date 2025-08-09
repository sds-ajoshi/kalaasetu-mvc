FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/.cache

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    ffmpeg git curl sox libsndfile1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements early to leverage Docker cache (common + service)
COPY ./services/common_requirements.txt ./common_requirements.txt
COPY ./services/audio_service/requirements.txt ./requirements.txt

# Install Python dependencies
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install --no-cache-dir -r common_requirements.txt -r requirements.txt

# Copy audio service code
COPY ./services/audio_service /app/services/audio_service

# Set default command
CMD ["uvicorn", "services.audio_service.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
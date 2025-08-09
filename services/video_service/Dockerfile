# services/video_service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install FFmpeg for video processing
RUN apt-get update && apt-get install -y ffmpeg

# Copy and install common + service-specific requirements
COPY ./services/common_requirements.txt /app/common_requirements.txt
COPY ./services/video_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r common_requirements.txt -r requirements.txt

COPY ./services/video_service /app/services/video_service

CMD ["uvicorn", "services.video_service.main:app", "--host", "0.0.0.0", "--port", "8003"]
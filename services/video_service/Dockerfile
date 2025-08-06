# services/video_service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install FFmpeg for video processing
RUN apt-get update && apt-get install -y ffmpeg

COPY ./services/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./services/video_service /app/services/video_service

CMD ["uvicorn", "services.video_service.main:app", "--host", "0.0.0.0", "--port", "8003"]
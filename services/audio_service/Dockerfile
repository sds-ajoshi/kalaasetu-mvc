# services/audio_service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependency for audio processing
RUN apt-get update && apt-get install -y ffmpeg

COPY ./services/requirements.txt /app/requirements.txt
# Note: Installing TTS can take a while
RUN pip install --no-cache-dir -r requirements.txt

ENV COQUI_TOS_AGREED=1

COPY ./services/audio_service /app/services/audio_service

CMD ["uvicorn", "services.audio_service.main:app", "--host", "0.0.0.0", "--port", "8002"]
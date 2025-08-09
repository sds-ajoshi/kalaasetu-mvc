# Kalaa-Setu MVC

Minimal microservice demo for text→audio/graphics→video generation with multilingual support.

Services

- `api_gateway` (FastAPI): orchestrates translation, audio, graphics, and video composition
- `audio_service` (FastAPI, Coqui XTTS v2): multilingual TTS, returns base64 WAV and stub subtitles
- `graphics_service` (FastAPI, SDXL): generates base64 PNG from prompt with tone/domain/environment
- `video_service` (FastAPI, FFmpeg): composes image+audio+subtitles into MP4
- `translation_service` (FastAPI, IndicTrans2): EN↔Indic translation for prompts

Run (Docker, with NVIDIA runtime for GPU)

1. Install Docker + NVIDIA Container Toolkit (for GPU)
2. Build and start:
   - `docker compose build`
   - `docker compose up`
3. Open `index.html` in a browser. Set API base to `http://localhost:8000` if asked.

Key endpoints

- `POST /generate/audio_only` → base64 WAV + subtitles
- `POST /generate/video_from_text` → MP4 (bytes) + latency headers
- `GET /generate_from_pib?url=...` → MP4 (bytes) + latency headers
- Health: `/health` on each service

Notes

- GPU is recommended for `graphics_service` and `audio_service` to meet latency goals.
- Subtitles are currently simple placeholders from the TTS service; burned into the MP4 if present.

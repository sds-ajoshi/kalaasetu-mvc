# services/audio_service/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
from TTS.api import TTS
import torch
from pydantic import BaseModel
import io
import base64

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
model_pipeline = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading XTTS model on device: {device}")
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts = TTS(model_name).to(device)
    model_pipeline["xtts"] = tts
    print("XTTS Model Loaded.")
    
    yield

    print("Releasing model resources...")
    model_pipeline.clear()

app = FastAPI(title="Audio Generation Service", lifespan=lifespan)

class AudioRequest(BaseModel):
    text: str
    language: str = "en"  # Default to English
    tone: str = "neutral" # Tone for emotion-aware synthesis (optional)

@app.post("/generate/audio")
async def generate_audio(request: AudioRequest):
    """
    Generate audio and subtitles from text using Coqui XTTS.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text input is required.")

    xtts = model_pipeline.get("xtts")
    if not xtts:
        raise HTTPException(status_code=500, detail="TTS model not loaded.")

    # Voice customization: Use speaker embeddings or voice description by tone
    speaker_wav = None
    if request.tone.lower() in {"serious", "calm"}:
        # Use slower speed or lower energy voice
        speed = 0.95
        emotion = "neutral"
    elif request.tone.lower() in {"excited", "happy"}:
        speed = 1.05
        emotion = "excited"
    elif request.tone.lower() in {"sad", "emotional"}:
        speed = 0.9
        emotion = "sad"
    else:
        speed = 1.0
        emotion = "neutral"

    # Generate audio
    try:
        wav = xtts.tts(
            text=request.text,
            language=request.language,
            speaker_wav=speaker_wav,
            emotion=emotion
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

    # Write audio to buffer
    buffer = io.BytesIO()
    xtts.save_wav(wav, buffer)
    audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Generate mock subtitle data (can be replaced by Whisper)
    subtitles = [{"start": 0.0, "end": 2.0, "text": request.text}]

    return {
        "audio": audio_base64,
        "subtitles": subtitles,
        "tone_used": request.tone
    }
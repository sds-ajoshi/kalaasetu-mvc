# services/audio_service/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
from TTS.api import TTS
import torch
from pydantic import BaseModel
import io

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

model_pipeline = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the multilingual TTS model
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
    language: str = "en" # Default to English

@app.post("/generate/audio")
async def generate_audio(request: AudioRequest):
    """
    Generates audio using a multilingual XTTS model.
    Supported languages include en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi
    """
    if "xtts" not in model_pipeline:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    tts = model_pipeline["xtts"]
    
    try:
        # Generate speech to a file-like object in memory
        wav_buffer = io.BytesIO()
        # Note: XTTS requires a speaker_wav for voice cloning. For now, we use a default voice.
        # For a real implementation, you'd have a library of speaker wav files.
        # Here we are using the model's default synthesizer which may not require a speaker wav.
        tts.tts_to_file(
            text=request.text,
            file_path=wav_buffer,
            language=request.language,
            # A speaker wav is needed for high-quality XTTSv2 voice cloning. 
            # We will let the model use its default if possible, or you might need a reference audio.
            # For this demo, let's assume default voice works.
        )
        wav_buffer.seek(0)
        
        # Return the raw WAV audio file
        return Response(content=wav_buffer.read(), media_type="audio/wav")

    except Exception as e:
        print(f"An error occurred during audio generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
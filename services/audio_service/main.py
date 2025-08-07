import os
import base64
import torch
import tempfile
import sys
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Set up FastAPI
app = FastAPI(title="Kalaa-Setu Audio Service")

# 🛡️ Handle license acceptance for TTS models
def setup_license_acceptance():
    """Set up automatic license acceptance for TTS models"""
    # Set environment variable to skip license check
    os.environ["TTS_ACCEPT_LICENSE"] = "true"
    
    # Create the cache directory if it doesn't exist
    cache_dir = os.path.expanduser("~/.cache/tts")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a tos.txt file with license acceptance
    tos_file = os.path.join(cache_dir, "tos.txt")
    if not os.path.exists(tos_file):
        with open(tos_file, "w") as f:
            f.write("I agree to the terms of the non-commercial CPML: https://coqui.ai/cpml\n")

# 🛡️ Mock input function to automatically accept license
original_input = input
def mock_input(prompt=""):
    if "agree to the terms" in prompt.lower() or "y/n" in prompt.lower():
        return "y"
    return original_input(prompt)

# Replace the input function
import builtins
builtins.input = mock_input

# 🧠 Model setup
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Set up license acceptance before model loading
setup_license_acceptance()

# Initialize model with error handling
try:
    model = Xtts.init_from_config(XttsConfig())
    model.load_checkpoint(config=model.config, checkpoint_dir=ModelManager().download_model(MODEL_NAME), eval=True)
    model.cuda()  # Move model to GPU
    print("✅ TTS Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading TTS model: {e}")
    model = None

#  Input schema
class AudioRequest(BaseModel):
    text: str
    tone: str = "neutral"
    language: str = "eng_Latn"

@app.post("/generate/audio")
def generate_audio(req: AudioRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="TTS model not loaded")
    
    try:
        # 🧠 Inference
        outputs = model.synthesize(
            text=req.text,
            speaker_wav=None,
            language=req.language,
            split_sentences=True
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            outputs.save_wav(tmp.name)
            tmp.seek(0)
            audio_data = base64.b64encode(tmp.read()).decode("utf-8")

        # 🧾 Return base64 audio and subtitles (stub for now)
        return JSONResponse(content={
            "audio": audio_data,
            "subtitles": [{"text": req.text, "start": 0.0, "end": 5.0}]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
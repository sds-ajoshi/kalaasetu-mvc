import os
import base64
import torch
import tempfile
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Set up FastAPI
app = FastAPI(title="Kalaa-Setu Audio Service")

# 🧠 Model setup
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
model = Xtts.init_from_config(XttsConfig())
model.load_checkpoint(config=model.config, checkpoint_dir=ModelManager().download_model(MODEL_NAME), eval=True)
model.cuda()  # Move model to GPU

# 🧾 Input schema
class AudioRequest(BaseModel):
    text: str
    tone: str = "neutral"
    language: str = "eng_Latn"

@app.post("/generate/audio")
def generate_audio(req: AudioRequest):
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
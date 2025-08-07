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

# Accept Coqui CPML TOS non-interactively
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# FastAPI app
app = FastAPI(title="Kalaa-Setu Audio Service")

# Model setup
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

try:
    manager = ModelManager()
    model_dir = manager.download_model(MODEL_NAME)
    # download_model may return (path, meta); normalize to path
    if isinstance(model_dir, tuple):
        model_dir = model_dir[0]

    cfg = XttsConfig()
    cfg.load_json(os.path.join(model_dir, "config.json"))

    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(config=cfg, checkpoint_dir=model_dir, eval=True)

    if torch.cuda.is_available():
        model.cuda()

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
import os
import base64
import torch
import tempfile
import sys
import numpy as np
import soundfile as sf
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig
try:
    # Some versions expose XttsArgs used during checkpoint load
    from TTS.tts.models.xtts import XttsArgs  # type: ignore
except Exception:  # pragma: no cover
    XttsArgs = None  # type: ignore
from TTS.config.shared_configs import BaseDatasetConfig

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from torch.serialization import add_safe_globals
# Allowlist required classes for torch.load safe unpickling on PyTorch >= 2.6
_allowlist = [XttsConfig, XttsAudioConfig, BaseDatasetConfig]
if XttsArgs is not None:
    _allowlist.append(XttsArgs)
add_safe_globals(_allowlist)

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
            split_sentences=True,
        )

        # Save to temp file robustly depending on output type
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            saved = False
            # Case 1: object exposes save_wav
            if hasattr(outputs, "save_wav"):
                try:
                    outputs.save_wav(tmp.name)
                    saved = True
                except Exception:
                    saved = False

            if not saved:
                # Case 2: dict/tuple/ndarray
                wav = None
                if isinstance(outputs, dict):
                    wav = outputs.get("wav") or outputs.get("audio") or outputs.get("samples")
                elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                    wav = outputs[0]
                elif isinstance(outputs, np.ndarray):
                    wav = outputs

                if wav is None:
                    raise RuntimeError("Unknown synthesize output format")

                # Ensure 1-D numpy array
                wav = np.asarray(wav).squeeze()
                sr = getattr(getattr(model, "config", None), "audio", None)
                sample_rate = getattr(sr, "sample_rate", 24000) if sr else 24000
                sf.write(tmp.name, wav, sample_rate)

            tmp.seek(0)
            audio_data = base64.b64encode(tmp.read()).decode("utf-8")

        # 🧾 Return base64 audio and subtitles (stub for now)
        return JSONResponse(content={
            "audio": audio_data,
            "subtitles": [{"text": req.text, "start": 0.0, "end": 5.0}]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import os
import base64
import torch
import tempfile
import sys
import numpy as np
import soundfile as sf
import time
from gtts import gTTS
from pydub import AudioSegment
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig
try:
    # High-level API fallback
    from TTS.api import TTS as CoquiTTS
except Exception:  # pragma: no cover
    CoquiTTS = None  # type: ignore
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
last_metrics = {"last_inference_ms": None, "device": None}

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
tts_api_model = None  # high-level API fallback instance

#  Input schema
class AudioRequest(BaseModel):
    text: str
    tone: str = "neutral"
    language: str = "eng_Latn"

def _generate_with_tts_api(text: str, xtts_lang: str):
    if CoquiTTS is None:
        raise RuntimeError("High-level TTS API unavailable")
    global tts_api_model
    if tts_api_model is None:
        tts_api_model = CoquiTTS(MODEL_NAME)
        if torch.cuda.is_available():
            try:
                tts_api_model = tts_api_model.to('cuda')
            except Exception:
                pass
    wav_np = tts_api_model.tts(text=text, language=xtts_lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        # Try to get sample rate from config if available
        sample_rate = 24000
        try:
            sample_rate = getattr(getattr(model, "config", None), "audio", None).sample_rate or 24000
        except Exception:
            pass
        sf.write(tmp.name, np.asarray(wav_np).squeeze(), sample_rate)
        tmp.seek(0)
        try:
            info = sf.info(tmp.name)
            duration_sec = float(info.duration)
        except Exception:
            duration_sec = 5.0
        wav_bytes = tmp.read()
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return audio_b64, duration_sec


def _lang_to_gtts(lang_code: str) -> str | None:
    # Map IndicTrans-like codes to gTTS short codes
    mapping = {
        "eng_Latn": "en",
        "hin_Deva": "hi",
        "tam_Taml": "ta",
        "tel_Telu": "te",
        "ben_Beng": "bn",
        "mar_Deva": "mr",
        "guj_Gujr": "gu",
        "kan_Knda": "kn",
        "mal_Mlym": "ml",
        "pan_Guru": "pa",
        "urd_Arab": "ur",
        "ory_Orya": "or",
        "asm_Beng": "as",
    }
    return mapping.get(lang_code)


def _generate_with_gtts(text: str, tgt_lang_short: str):
    # Sentence-synthesize with gTTS and join via pydub; produce simple subtitles
    sentences = []
    try:
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
    except Exception:
        sentences = [text]
    if not sentences:
        sentences = [text]

    combined = AudioSegment.empty()
    segments_durations = []
    for s in sentences:
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        fp.close()
        tts = gTTS(text=s, lang=tgt_lang_short)
        tts.save(fp.name)
        seg = AudioSegment.from_file(fp.name, format="mp3")
        combined += seg
        segments_durations.append(len(seg) / 1000.0)
        try:
            os.remove(fp.name)
        except Exception:
            pass

    out_fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    out_fp.close()
    combined.export(out_fp.name, format="mp3")
    with open(out_fp.name, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    try:
        os.remove(out_fp.name)
    except Exception:
        pass

    # Build subtitles by cumulative durations
    subtitles = []
    cursor = 0.0
    for s, dur in zip(sentences, segments_durations):
        start = cursor
        end = cursor + dur
        subtitles.append({"text": s, "start": float(start), "end": float(end)})
        cursor = end
    duration_sec = cursor
    return audio_b64, duration_sec, subtitles


@app.post("/generate/audio")
def generate_audio(req: AudioRequest):
    try:
        t0 = time.perf_counter()
        # Prefer gTTS for supported languages (fast, robust), else use XTTS
        gtts_lang = _lang_to_gtts(req.language)
        if gtts_lang:
            audio_data, duration_sec, subtitles = _generate_with_gtts(req.text, gtts_lang)
            def seconds_to_srt_timestamp(seconds: float) -> str:
                if seconds < 0:
                    seconds = 0.0
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int(round((seconds - int(seconds)) * 1000))
                return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
            srt_lines = []
            for i, it in enumerate(subtitles, start=1):
                srt_lines.append(str(i))
                srt_lines.append(f"{seconds_to_srt_timestamp(it['start'])} --> {seconds_to_srt_timestamp(it['end'])}")
                srt_lines.append(it['text'])
                srt_lines.append("")
            srt_content = "\n".join(srt_lines).strip() + "\n"
            t1 = time.perf_counter()
            last_metrics["last_inference_ms"] = int((t1 - t0) * 1000)
            last_metrics["device"] = "cpu"  # gTTS path is CPU-based
            return JSONResponse(content={
                "audio": audio_data,
                "subtitles": subtitles,
                "subtitles_srt": srt_content,
                "duration_sec": duration_sec,
                "audio_format": "mp3",
            })

        # Map IndicTrans-like tags to XTTS language codes where needed
        lang_map = {
            "eng_Latn": "en",
            "hin_Deva": "hi",
            "tam_Taml": "ta",
            "tel_Telu": "te",
            "ben_Beng": "bn",
            "mar_Deva": "mr",
            "guj_Gujr": "gu",
            "kan_Knda": "kn",
            "mal_Mlym": "ml",
            "pan_Guru": "pa",
            "urd_Arab": "ur",
            "ory_Orya": "or",
            "asm_Beng": "as",
        }
        xtts_lang = lang_map.get(req.language, req.language)

        # 🧠 Inference
        if model is None:
            audio_data, duration_sec = _generate_with_tts_api(req.text, xtts_lang)
        else:
            try:
                outputs = model.synthesize(
                    text=req.text,
                    config=cfg,  # XTTS requires config param in recent versions
                    speaker_wav=None,
                    language=xtts_lang,
                )
            except Exception:
                audio_data, duration_sec = _generate_with_tts_api(req.text, xtts_lang)
            else:
                # Save to temp file robustly depending on output type
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    saved = False
                    if hasattr(outputs, "save_wav"):
                        try:
                            outputs.save_wav(tmp.name)
                            saved = True
                        except Exception:
                            saved = False
                    if not saved:
                        wav = None
                        if isinstance(outputs, dict):
                            wav = outputs.get("wav") or outputs.get("audio") or outputs.get("samples")
                        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                            wav = outputs[0]
                        elif isinstance(outputs, np.ndarray):
                            wav = outputs
                        if wav is None:
                            raise RuntimeError("Unknown synthesize output format")
                        wav = np.asarray(wav).squeeze()
                        sr = getattr(getattr(model, "config", None), "audio", None)
                        sample_rate = getattr(sr, "sample_rate", 24000) if sr else 24000
                        sf.write(tmp.name, wav, sample_rate)
                    try:
                        info = sf.info(tmp.name)
                        duration_sec = float(info.duration)
                    except Exception:
                        duration_sec = 5.0
                    tmp.seek(0)
                    wav_bytes = tmp.read()
                    audio_data = base64.b64encode(wav_bytes).decode("utf-8")
            # Build subtitles and return early
            # Build simple subtitles: split text into sentences and spread over duration
            def split_sentences(text: str):
                import re
                parts = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
                if not parts:
                    parts = [text]
                return parts
            def seconds_to_srt_timestamp(seconds: float) -> str:
                if seconds < 0:
                    seconds = 0.0
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int(round((seconds - int(seconds)) * 1000))
                return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
            def to_srt(items):
                lines = []
                for idx, it in enumerate(items, start=1):
                    lines.append(str(idx))
                    lines.append(f"{seconds_to_srt_timestamp(it['start'])} --> {seconds_to_srt_timestamp(it['end'])}")
                    lines.append(it['text'])
                    lines.append("")
                return "\n".join(lines).strip() + "\n"
            sentence_list = split_sentences(req.text)
            num_segments = max(1, min(len(sentence_list), 10))
            total_chars = sum(len(s) for s in sentence_list[:num_segments]) or 1
            min_seg = 0.5
            subtitles = []
            cursor = 0.0
            for i, sent in enumerate(sentence_list[:num_segments]):
                share = len(sent) / total_chars
                dur = max(min_seg, duration_sec * share)
                start_t = cursor
                end_t = min(duration_sec, start_t + dur)
                if i == num_segments - 1:
                    end_t = duration_sec
                subtitles.append({"text": sent, "start": float(start_t), "end": float(end_t)})
                cursor = end_t
            srt_content = to_srt(subtitles)
            t1 = time.perf_counter()
            last_metrics["last_inference_ms"] = int((t1 - t0) * 1000)
            last_metrics["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            return JSONResponse(content={
                "audio": audio_data,
                "subtitles": subtitles,
                "subtitles_srt": srt_content,
                "duration_sec": duration_sec,
                "audio_format": "wav",
            })

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

            # Calculate duration and read bytes
            try:
                info = sf.info(tmp.name)
                duration_sec = float(info.duration)
            except Exception:
                duration_sec = 5.0
            tmp.seek(0)
            wav_bytes = tmp.read()
            audio_data = base64.b64encode(wav_bytes).decode("utf-8")

        # Build simple subtitles: split text into sentences and spread over duration
        def split_sentences(text: str):
            import re
            parts = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
            if not parts:
                parts = [text]
            return parts

        def seconds_to_srt_timestamp(seconds: float) -> str:
            if seconds < 0:
                seconds = 0.0
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int(round((seconds - int(seconds)) * 1000))
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        def to_srt(items):
            lines = []
            for idx, it in enumerate(items, start=1):
                lines.append(str(idx))
                lines.append(f"{seconds_to_srt_timestamp(it['start'])} --> {seconds_to_srt_timestamp(it['end'])}")
                lines.append(it['text'])
                lines.append("")
            return "\n".join(lines).strip() + "\n"

        sentence_list = split_sentences(req.text)
        num_segments = max(1, min(len(sentence_list), 10))
        # proportional duration per segment (fallback equal split)
        total_chars = sum(len(s) for s in sentence_list[:num_segments]) or 1
        min_seg = 0.5
        subtitles = []
        cursor = 0.0
        for i, sent in enumerate(sentence_list[:num_segments]):
            share = len(sent) / total_chars
            dur = max(min_seg, duration_sec * share)
            start_t = cursor
            end_t = min(duration_sec, start_t + dur)
            # ensure the last segment ends exactly at duration
            if i == num_segments - 1:
                end_t = duration_sec
            subtitles.append({"text": sent, "start": float(start_t), "end": float(end_t)})
            cursor = end_t

        srt_content = to_srt(subtitles)
        t1 = time.perf_counter()
        last_metrics["last_inference_ms"] = int((t1 - t0) * 1000)
        last_metrics["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        # 🧾 Return base64 audio and subtitles (proportional timing)
        return JSONResponse(content={
            "audio": audio_data,
            "subtitles": subtitles,
            "subtitles_srt": srt_content,
            "duration_sec": duration_sec,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {type(e).__name__}: {e}")


@app.get("/health")
def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "service": "audio_service",
        "model_loaded": model is not None,
        "device": device,
    }


@app.get("/metrics")
def metrics():
    return {
        "service": "audio_service",
        **last_metrics,
    }
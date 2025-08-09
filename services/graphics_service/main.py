from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
from diffusers import DiffusionPipeline
import psutil
import subprocess
import open_clip
import torch.nn.functional as F
from pydantic import BaseModel
import base64
from io import BytesIO

# This dictionary will hold our loaded model
model_pipeline = {}
clip_model_state = {}

# Use a lifespan manager to load the model on startup and release it on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model
    print("Loading SDXL model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
    )

    # Memory/perf tweaks for limited environments
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    pipe.to(device)
    model_pipeline["sdxl"] = pipe
    print("SDXL Model Loaded.")

    # Load CLIP for scoring (ViT-H-14 laion2b_s32b_b79k)
    try:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14', pretrained='laion2b_s32b_b79k'
        )
        clip_model.eval()
        if torch.cuda.is_available():
            clip_model = clip_model.cuda()
        clip_model_state["model"] = clip_model
        clip_model_state["preprocess"] = clip_preprocess
        print("CLIP model loaded for scoring.")
    except Exception as e:
        print(f"CLIP load failed: {e}")

    yield

    print("Releasing model resources...")
    model_pipeline.clear()


app = FastAPI(title="Graphics Generation Service", lifespan=lifespan)
last_metrics = {"last_inference_ms": None, "device": None}


class GraphicsRequest(BaseModel):
    text: str
    tone: str = "neutral"
    domain: str = "general"
    environment: str = "neutral"


@app.post("/generate/graphics")
async def generate_graphic(req: GraphicsRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Input text is required.")

    # Build enhanced prompt
    prompt = (
        f"{req.text}, illustrated in a {req.tone} tone, "
        f"related to {req.domain}, set in a {req.environment} background"
    )

    try:
        import time
        t0 = time.perf_counter()
        # Reduce steps to keep latency reasonable on CPU
        image = model_pipeline["sdxl"](prompt=prompt, num_inference_steps=10, guidance_scale=5.0).images[0]
        t1 = time.perf_counter()
        last_metrics["last_inference_ms"] = int((t1 - t0) * 1000)
        last_metrics["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

    # Convert image to base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Optional CLIPScore
    clip_score = None
    try:
        if clip_model_state and clip_model_state.get("model") is not None:
            model = clip_model_state["model"]
            preprocess = clip_model_state["preprocess"]
            img_tensor = preprocess(image).unsqueeze(0)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            with torch.no_grad():
                image_features = model.encode_image(img_tensor)
                text = open_clip.tokenize([prompt])
                if torch.cuda.is_available():
                    text = text.cuda()
                text_features = model.encode_text(text)
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                sim = (image_features @ text_features.T).item()
                clip_score = float(sim)
    except Exception:
        clip_score = None

    return {
        "prompt": prompt,
        "image": image_base64,
        "clip_score": clip_score,
    }


@app.get("/health")
async def health():
    loaded = "sdxl" in model_pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "service": "graphics_service",
        "model_loaded": loaded,
        "device": device,
    }


@app.get("/metrics")
async def metrics():
    # Basic system metrics
    cpu = psutil.cpu_percent(interval=0.0)
    mem = psutil.virtual_memory().percent
    gpu = None
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits']).decode().strip()
        if out:
            util, mem_used = out.split(',')[0:2]
            gpu = {"util_percent": float(util), "mem_used_mb": float(mem_used)}
    except Exception:
        gpu = None
    return {
        "service": "graphics_service",
        **last_metrics,
        "cpu_percent": cpu,
        "ram_percent": mem,
        "gpu": gpu,
    }
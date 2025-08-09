from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
from diffusers import DiffusionPipeline
from pydantic import BaseModel
import base64
from io import BytesIO

# This dictionary will hold our loaded model
model_pipeline = {}

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

    return {
        "prompt": prompt,
        "image": image_base64
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
    return {
        "service": "graphics_service",
        **last_metrics,
    }
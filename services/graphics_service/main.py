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
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to("cuda")
    model_pipeline["sdxl"] = pipe
    print("SDXL Model Loaded.")

    yield

    print("Releasing model resources...")
    model_pipeline.clear()


app = FastAPI(title="Graphics Generation Service", lifespan=lifespan)


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
        image = model_pipeline["sdxl"](prompt=prompt).images[0]
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
# services/graphics_service/main.py
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
    
    yield # The application runs here
    
    # Clean up and release memory
    print("Releasing model resources...")
    model_pipeline.clear()


app = FastAPI(title="Graphics Generation Service", lifespan=lifespan)

# Define the input data structure for type checking and API docs
class GraphicsRequest(BaseModel):
    text: str
    parameters: dict = {}

@app.post("/generate/graphics")
async def generate_graphics(request: GraphicsRequest):
    """
    Generates graphics using the SDXL model based on text and parameters.
    """
    if "sdxl" not in model_pipeline:
        raise HTTPException(status_code=503, detail="Model is not ready. Please try again later.")

    # --- Basic Prompt Engineering ---
    # Combine user text with parameters for a richer prompt
    style_prompt = "infographic, digital art, clean vector style"
    color_scheme = request.parameters.get("color_scheme", "vibrant")
    tone = request.parameters.get("tone", "neutral")

    final_prompt = f"{request.text}, in a {tone} tone, {style_prompt}, {color_scheme} colors"
    print(f"Generating image for prompt: {final_prompt}")

    # --- Model Inference ---
    pipe = model_pipeline["sdxl"]
    try:
        image = pipe(prompt=final_prompt).images[0]

        # --- Convert image to Base64 to return in JSON ---
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "prompt": final_prompt,
            "image_base64": img_str
        }
    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate image.")
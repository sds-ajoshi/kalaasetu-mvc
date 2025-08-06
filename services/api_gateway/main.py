# services/api_gateway/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import httpx
import os
import asyncio # Import asyncio

app = FastAPI(title="Kalaa-Setu API Gateway")

# Get service URLs from environment variables
GRAPHICS_SERVICE_URL = os.getenv("GRAPHICS_SERVICE_URL", "http://localhost:8001/generate/graphics")
AUDIO_SERVICE_URL = os.getenv("AUDIO_SERVICE_URL", "http://localhost:8002/generate/audio")

@app.get("/")
def read_root():
    return {"message": f"Kalaa-Setu API Gateway is running. Targeting graphics service at: {GRAPHICS_SERVICE_URL}"}

@app.post("/generate/audio_only")
async def generate_audio_only(request_data: dict):
    """
    Endpoint to generate only audio. Proxies the request to the audio service.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(AUDIO_SERVICE_URL, json=request_data)
            response.raise_for_status()
        
        # Return the raw audio content with the correct content type
        return Response(content=response.content, media_type=response.headers['content-type'])

    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Error communicating with the audio service: {exc}")

@app.post("/generate/video_from_text")
async def generate_content(request_data: dict):
    """
    Main endpoint to generate a full video.
    This is a placeholder for the full orchestration logic.
    """
    text = request_data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    try:
        # Use the configured URL to call the graphics service
        async with httpx.AsyncClient() as client:
            response = await client.post(GRAPHICS_SERVICE_URL, json={"text": "A test graphic for: " + text})
            response.raise_for_status() # Raises an exception for 4xx/5xx responses
        return response.json()

    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Error communicating with a worker service: {exc}")
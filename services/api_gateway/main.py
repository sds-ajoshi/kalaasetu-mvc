# services/api_gateway/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import httpx
import os
import asyncio
import base64

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
    
@app.post("/generate/video_from_text")
async def generate_content(request_data: dict):
    """
    Main endpoint to generate a full video.
    Orchestrates calls to graphics and audio services in parallel.
    """
    text = request_data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    # Prepare individual requests for each service
    graphics_payload = {"text": text, "parameters": request_data.get("parameters", {})}
    audio_payload = {"text": text, "language": request_data.get("language", "en")}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # --- Run calls in parallel for efficiency ---
            graphics_task = client.post(GRAPHICS_SERVICE_URL, json=graphics_payload)
            audio_task = client.post(AUDIO_SERVICE_URL, json=audio_payload)

            # Wait for both tasks to complete
            results = await asyncio.gather(
                graphics_task,
                audio_task,
                return_exceptions=True # Continue even if one fails
            )

            # Process results
            graphics_response, audio_response = results

            if isinstance(graphics_response, Exception):
                raise HTTPException(status_code=500, detail=f"Graphics service failed: {graphics_response}")
            if isinstance(audio_response, Exception):
                raise HTTPException(status_code=500, detail=f"Audio service failed: {audio_response}")

            # Combine the results into a single JSON response
            return {
                "status": "success",
                "generated_content": {
                    "graphics": graphics_response.json(),
                    "audio_b64": base64.b64encode(audio_response.content).decode("utf-8")
                }
            }

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"An error occurred during orchestration: {e}")
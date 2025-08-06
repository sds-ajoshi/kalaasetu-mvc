# services/api_gateway/main.py
import os
import asyncio
import base64
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

app = FastAPI(title="Kalaa-Setu API Gateway")

# Get service URLs from environment variables
GRAPHICS_SERVICE_URL = os.getenv("GRAPHICS_SERVICE_URL", "http://localhost:8001/generate/graphics")
AUDIO_SERVICE_URL = os.getenv("AUDIO_SERVICE_URL", "http://localhost:8002/generate/audio")
VIDEO_SERVICE_URL = os.getenv("VIDEO_SERVICE_URL", "http://localhost:8003/create_video")

@app.get("/")
def read_root():
    """A simple endpoint to confirm the gateway is running."""
    return {"message": "Kalaa-Setu API Gateway is running."}


@app.post("/generate/video_from_text")
async def generate_content(request_data: dict):
    """
    Main endpoint to generate content.
    Orchestrates parallel calls to the graphics and audio services.
    """
    text = request_data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Input text is required.")

    # Prepare individual requests for each downstream service
    graphics_payload = {"text": text, "parameters": request_data.get("parameters", {})}
    audio_payload = {"text": text, "language": request_data.get("language", "en")}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # --- Run calls in parallel for better performance ---
            graphics_task = client.post(GRAPHICS_SERVICE_URL, json=graphics_payload)
            audio_task = client.post(AUDIO_SERVICE_URL, json=audio_payload)

            # Wait for both tasks to complete using asyncio.gather
            results = await asyncio.gather(graphics_task, audio_task, return_exceptions=True)

            graphics_response, audio_response = results

            # --- Error handling for each service ---
            if isinstance(graphics_response, Exception):
                raise HTTPException(status_code=503, detail=f"Graphics service failed: {graphics_response}")
            if isinstance(audio_response, Exception):
                raise HTTPException(status_code=503, detail=f"Audio service failed: {audio_response}")

            if graphics_response.status_code != 200:
                raise HTTPException(status_code=graphics_response.status_code, detail=f"Graphics service error: {graphics_response.text}")
            if audio_response.status_code != 200:
                 raise HTTPException(status_code=audio_response.status_code, detail=f"Audio service error: {audio_response.text}")


            # --- Combine results into a single JSON response ---
            # We base64 encode the audio to package it neatly in the JSON
            return {
                "status": "success",
                "generated_content": {
                    "graphics": graphics_response.json(),
                    "audio": {
                        "format": "wav",
                        "content_b64": base64.b64encode(audio_response.content).decode("utf-8")
                    }
                }
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during orchestration: {e}")


@app.post("/generate/audio_only")
async def generate_audio_only(request_data: dict):
    """
    Endpoint to generate only audio. Useful for debugging.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(AUDIO_SERVICE_URL, json=request_data)
            response.raise_for_status()
        
        return Response(content=response.content, media_type=response.headers['content-type'])

    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Error communicating with the audio service: {exc}")
    
@app.post("/create/final_video")
async def create_final_video(request_data: dict):
    """
    Top-level endpoint to trigger the full text-to-video pipeline.
    """
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(VIDEO_SERVICE_URL, json=request_data)
            response.raise_for_status()
        
        # Stream the video file back to the client
        return Response(content=response.content, media_type=response.headers['content-type'])

    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Error communicating with the video service: {exc}")
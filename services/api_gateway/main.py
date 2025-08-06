# services/api_gateway/main.py
from fastapi import FastAPI, HTTPException
import httpx
import os # Import the os module

app = FastAPI(title="Kalaa-Setu API Gateway")

# Get the worker service URL from an environment variable.
# Provide a default value for local running without Docker.
GRAPHICS_SERVICE_URL = os.getenv("GRAPHICS_SERVICE_URL", "http://localhost:8001/generate/graphics")

@app.get("/")
def read_root():
    return {"message": f"Kalaa-Setu API Gateway is running. Targeting graphics service at: {GRAPHICS_SERVICE_URL}"}

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
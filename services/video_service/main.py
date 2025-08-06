# services/video_service/main.py
import os
import httpx
import base64
import uuid
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Video Composition Service")

# The gateway is our single source for generated content
GATEWAY_URL = "http://api_gateway:8000/generate/video_from_text"

class VideoRequest(BaseModel):
    text: str
    language: str = "en"
    parameters: dict = {}

@app.post("/create_video")
async def create_video(request: VideoRequest):
    # 1. Get the generated assets from our API gateway
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(GATEWAY_URL, json=request.dict())
            response.raise_for_status()
            content = response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to get assets from gateway: {e}")

    # 2. Decode the base64 content and save to temporary files
    # Using UUID ensures unique filenames to prevent conflicts
    request_id = str(uuid.uuid4())
    temp_dir = "/tmp"
    image_path = os.path.join(temp_dir, f"{request_id}.png")
    audio_path = os.path.join(temp_dir, f"{request_id}.wav")
    output_path = os.path.join(temp_dir, f"{request_id}.mp4")

    try:
        # Save image
        img_data = base64.b64decode(content["generated_content"]["graphics"]["image_base64"])
        with open(image_path, "wb") as f:
            f.write(img_data)

        # Save audio
        audio_data = base64.b64decode(content["generated_content"]["audio"]["content_b64"])
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        # 3. Run FFmpeg to combine the image and audio
        # -loop 1: Loop the image
        # -i: Input files (image and audio)
        # -c:v libx264: Video codec
        # -tune stillimage: Optimize for static images
        # -c:a aac: Audio codec
        # -pix_fmt yuv420p: Pixel format for compatibility
        # -shortest: Finish encoding when the shortest input (the audio) ends
        ffmpeg_command = [
            "ffmpeg",
            "-loop", "1",
            "-i", image_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path,
        ]
        
        subprocess.run(ffmpeg_command, check=True)

        # 4. Return the generated video file
        return FileResponse(path=output_path, media_type="video/mp4", filename="kalaa_setu_video.mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create video: {e}")
    finally:
        # 5. Clean up temporary files
        for path in [image_path, audio_path, output_path]:
            if os.path.exists(path):
                os.remove(path)
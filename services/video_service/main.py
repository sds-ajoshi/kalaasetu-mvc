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

GATEWAY_URL = "http://api_gateway:8000/generate/video_from_text"

class VideoRequest(BaseModel):
    text: str
    language: str = "en"
    parameters: dict = {}

@app.post("/create_video")
async def create_video(request: VideoRequest):
    # 1. Get assets from the gateway
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(GATEWAY_URL, json=request.dict())
            response.raise_for_status()
            content = response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to get assets from gateway: {e}")

    # 2. Decode content and save to temporary files
    request_id = str(uuid.uuid4())
    temp_dir = "/tmp"
    image_path = os.path.join(temp_dir, f"{request_id}.png")
    audio_path = os.path.join(temp_dir, f"{request_id}.wav")
    subtitle_path = os.path.join(temp_dir, f"{request_id}.srt") # New file for subtitles
    output_path = os.path.join(temp_dir, f"{request_id}.mp4")

    try:
        # Extract and save image, audio, and subtitle data
        graphics_content = content["generated_content"]["graphics"]
        audio_content = content["generated_content"]["audio"]

        with open(image_path, "wb") as f:
            f.write(base64.b64decode(graphics_content["image_base64"]))
        
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(audio_content["audio_b64"]))

        with open(subtitle_path, "w") as f:
            f.write(audio_content["srt_content"])

        # 3. Run FFmpeg with the subtitles filter
        # The -vf "subtitles=..." filter burns the SRT file onto the video
        ffmpeg_command = [
            "ffmpeg",
            "-loop", "1",
            "-i", image_path,
            "-i", audio_path,
            "-vf", f"subtitles={subtitle_path}", # Add subtitle filter
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path,
        ]
        
        subprocess.run(ffmpeg_command, check=True)

        # 4. Return the final video
        return FileResponse(path=output_path, media_type="video/mp4", filename="kalaa_setu_video.mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create video: {e}")
    finally:
        # 5. Clean up all temporary files
        for path in [image_path, audio_path, subtitle_path, output_path]:
            if os.path.exists(path):
                os.remove(path)
import os
import asyncio
import base64
import httpx
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

app = FastAPI(title="Kalaa-Setu API Gateway")

# Microservice URLs
GRAPHICS_SERVICE_URL = os.getenv("GRAPHICS_SERVICE_URL", "http://localhost:8001/generate/graphics")
AUDIO_SERVICE_URL = os.getenv("AUDIO_SERVICE_URL", "http://localhost:8002/generate/audio")
VIDEO_SERVICE_URL = os.getenv("VIDEO_SERVICE_URL", "http://localhost:8003/create_video")
TRANSLATION_SERVICE_URL = os.getenv("TRANSLATION_SERVICE_URL", "http://localhost:8004/translate")


@app.get("/")
def read_root():
    return {"message": "Kalaa-Setu API Gateway is running."}


@app.post("/generate/audio_only")
async def generate_audio_only(request_data: dict):
    text = request_data.get("text")
    tone = request_data.get("tone", "neutral")
    lang = request_data.get("language", "eng_Latn")

    if not text:
        raise HTTPException(status_code=400, detail="Text input is required.")

    async with httpx.AsyncClient() as client:
        response = await client.post(AUDIO_SERVICE_URL, json={
            "text": text,
            "tone": tone,
            "language": lang
        })

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to generate audio.")

    return response.json()


@app.post("/generate/video_from_text")
async def generate_content(request_data: dict):
    text = request_data.get("text")
    tone = request_data.get("tone", "neutral")
    domain = request_data.get("domain", "general")
    environment = request_data.get("environment", "neutral")
    src_lang = request_data.get("language", "eng_Latn")  # Language code: default to English

    if not text:
        raise HTTPException(status_code=400, detail="Text input is required.")

    original_text = text
    english_text = text

    # Translate if input is not English
    if src_lang != "eng_Latn":
        async with httpx.AsyncClient() as client:
            resp = await client.post(TRANSLATION_SERVICE_URL, json={
                "text": text,
                "src_lang": src_lang,
                "tgt_lang": "eng_Latn"
            })
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail="Translation failed.")
            english_text = resp.json().get("translated_text")

    async with httpx.AsyncClient() as client:
        audio_task = client.post(AUDIO_SERVICE_URL, json={
            "text": original_text,
            "tone": tone,
            "language": src_lang
        })
        graphics_task = client.post(GRAPHICS_SERVICE_URL, json={
            "text": english_text,
            "tone": tone,
            "domain": domain,
            "environment": environment
        })

        audio_response, graphics_response = await asyncio.gather(audio_task, graphics_task)

        if audio_response.status_code != 200 or graphics_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to generate audio or graphics.")

        audio_data = audio_response.json()
        image_data = graphics_response.json()

    video_payload = {
        "audio": audio_data.get("audio"),
        "image": image_data.get("image"),
        "subtitles": audio_data.get("subtitles"),
        "format": "mp4"
    }

    async with httpx.AsyncClient() as client:
        video_response = await client.post(VIDEO_SERVICE_URL, json=video_payload)

    if video_response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to create video.")

    return Response(content=video_response.content, media_type="video/mp4")


@app.post("/create/final_video")
async def create_final_video(request_data: dict):
    required_keys = ["audio", "image"]
    for key in required_keys:
        if key not in request_data:
            raise HTTPException(status_code=400, detail=f"{key} is required.")

    async with httpx.AsyncClient() as client:
        response = await client.post(VIDEO_SERVICE_URL, json=request_data)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Video creation failed.")

    return Response(content=response.content, media_type="video/mp4")


@app.get("/generate_from_pib")
async def generate_from_pib(url: str = Query(..., description="PIB press release URL")):
    try:
        response = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch PIB page: {str(e)}")

    soup = BeautifulSoup(response.content, "html.parser")

    # Try all known PIB content selectors
    content_block = (
        soup.find("span", {"id": "ContentPlaceHolder1_lblContents"}) or
        soup.find("div", {"id": "content"}) or
        soup.find("div", class_="content-area") or
        soup.find("div", class_="col-sm-12")
    )

    if not content_block:
        raise HTTPException(status_code=400, detail="Could not locate PIB content block.")

    text = content_block.get_text(separator="\n", strip=True)
    title = soup.title.string.strip() if soup.title else "PIB Release"

    full_text = f"{title}\n\n{text}"

    return await generate_content({
        "text": full_text,
        "tone": "formal",
        "domain": "governance",
        "environment": "official",
        "language": "eng_Latn"
    })
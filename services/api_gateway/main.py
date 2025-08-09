import os
import asyncio
import base64
import httpx
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Kalaa-Setu API Gateway")

# CORS (allow your web page to call the API from a different origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or ["http://your-site.com"] in prod
    allow_methods=["*"],          # allows OPTIONS automatically
    allow_headers=["*"],
    allow_credentials=False,      # keep False if you use "*"
    expose_headers=["Content-Disposition"],
    max_age=86400,
)

# Microservice URLs
GRAPHICS_SERVICE_URL = os.getenv("GRAPHICS_SERVICE_URL", "http://graphics-service:8001/generate/graphics")
AUDIO_SERVICE_URL = os.getenv("AUDIO_SERVICE_URL", "http://audio-service:8002/generate/audio")
VIDEO_SERVICE_URL = os.getenv("VIDEO_SERVICE_URL", "http://video-service:8003/create_video")
TRANSLATION_SERVICE_URL = os.getenv("TRANSLATION_SERVICE_URL", "http://translation-service:8004/translate")

# Increase client timeouts to tolerate model cold starts
HTTPX_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


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

    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        response = await client.post(AUDIO_SERVICE_URL, json={
            "text": text,
            "tone": tone,
            "language": lang
        })

    if response.status_code != 200:
        detail = None
        try:
            detail = response.text
        except Exception:
            detail = None
        raise HTTPException(status_code=500, detail=(detail or "Failed to generate audio."))

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
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            resp = await client.post(TRANSLATION_SERVICE_URL, json={
                "text": text,
                "src_lang": src_lang,
                "tgt_lang": "eng_Latn"
            })
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail="Translation failed.")
            english_text = resp.json().get("translated_text")

    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
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
            audio_err = audio_response.text if audio_response.status_code != 200 else None
            graphics_err = graphics_response.text if graphics_response.status_code != 200 else None
            raise HTTPException(status_code=500, detail=f"Audio error: {audio_err}; Graphics error: {graphics_err}")

        audio_data = audio_response.json()
        image_data = graphics_response.json()

    video_payload = {
        "audio": audio_data.get("audio"),
        "image": image_data.get("image"),
        "subtitles": audio_data.get("subtitles"),
        "format": "mp4"
    }

    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
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

    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        response = await client.post(VIDEO_SERVICE_URL, json=request_data)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Video creation failed.")

    return Response(content=response.content, media_type="video/mp4")

@app.get("/generate_from_pib")
async def generate_from_pib(url: str = Query(..., description="PIB press release URL")):
    # Step 1: Force English version
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        query_params['lang'] = ['1']  # Force English version
        new_query = urlencode(query_params, doseq=True)
        english_url = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.params,
            new_query,
            parsed_url.fragment
        ))

        # Step 2: Fetch page content
        response = requests.get(english_url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch PIB page: {str(e)}")

    # Step 3: Parse PIB content using the correct class
    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", class_="innner-page-main-about-us-content-right-part")

    if not content_div:
        raise HTTPException(status_code=400, detail="Could not locate PIB content block.")

    # Step 4: Extract text and title
    text = content_div.get_text(separator="\n", strip=True)
    title = soup.title.string.strip() if soup.title else "PIB Release"

    full_text = f"{title}\n\n{text}"

    # Step 5: Generate video
    return await generate_content({
        "text": full_text,
        "tone": "formal",
        "domain": "governance",
        "environment": "official",
        "language": "eng_Latn"
    })

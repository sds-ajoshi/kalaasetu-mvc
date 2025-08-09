import os
import base64
import uuid
import subprocess
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="Video Composition Service")
last_metrics = {"last_compose_ms": None}


class SubtitleItem(BaseModel):
    text: str
    start: float
    end: float


class ComposeRequest(BaseModel):
    # For backward compatibility (single-scene)
    image: Optional[str] = None  # base64 PNG
    # For multi-scene
    images: Optional[List[str]] = None  # list of base64 PNGs
    scene_durations: Optional[List[float]] = None  # seconds per scene, length must match images
    audio: str  # base64 WAV
    subtitles: Optional[Union[List[SubtitleItem], str]] = None  # list or SRT string
    format: str = "mp4"
    bgm: Optional[str] = None  # base64 WAV/MP3 for background music
    bgm_volume_db: float = -20.0


def seconds_to_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def subtitle_list_to_srt(items: List[SubtitleItem]) -> str:
    lines: List[str] = []
    for idx, it in enumerate(items, start=1):
        start_ts = seconds_to_srt_timestamp(it.start)
        end_ts = seconds_to_srt_timestamp(it.end)
        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(it.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


@app.post("/create_video")
async def create_video(request: ComposeRequest):
    request_id = str(uuid.uuid4())
    temp_dir = "/tmp"
    audio_path = os.path.join(temp_dir, f"{request_id}.wav")
    subtitle_path = os.path.join(temp_dir, f"{request_id}.srt")
    output_path = os.path.join(temp_dir, f"{request_id}.mp4")
    single_image_path = os.path.join(temp_dir, f"{request_id}.png")
    image_paths: List[str] = []
    bgm_path = os.path.join(temp_dir, f"{request_id}_bgm.bin")

    try:
        import time
        t0 = time.perf_counter()
        # Decode and write assets
        # Write images
        if request.images and len(request.images) > 0:
            for idx, img_b64 in enumerate(request.images):
                p = os.path.join(temp_dir, f"{request_id}_{idx}.png")
                with open(p, "wb") as f_img:
                    f_img.write(base64.b64decode(img_b64))
                image_paths.append(p)
        elif request.image:
            with open(single_image_path, "wb") as f_img:
                f_img.write(base64.b64decode(request.image))
            image_paths.append(single_image_path)
        else:
            raise HTTPException(status_code=400, detail="No image(s) provided")

        with open(audio_path, "wb") as f_aud:
            f_aud.write(base64.b64decode(request.audio))

        have_bgm = False
        if request.bgm:
            try:
                with open(bgm_path, "wb") as f_bgm:
                    f_bgm.write(base64.b64decode(request.bgm))
                have_bgm = True
            except Exception:
                have_bgm = False

        have_subtitles = False
        if request.subtitles:
            have_subtitles = True
            if isinstance(request.subtitles, str):
                srt_content = request.subtitles
            else:
                srt_content = subtitle_list_to_srt(request.subtitles)
            with open(subtitle_path, "w", encoding="utf-8") as f_srt:
                f_srt.write(srt_content)

        # Build FFmpeg command
        if len(image_paths) == 1:
            # Single-scene (backward compatible)
            if have_bgm:
                # Use filter_complex to mix audio and optionally burn subtitles
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", image_paths[0],
                    "-i", audio_path,
                    "-i", bgm_path,
                ]
                filter_parts: List[str] = []
                v_label_in = "0:v"
                if have_subtitles:
                    filter_parts.append(f"[{v_label_in}]subtitles={subtitle_path}:force_style='Fontsize=18,PrimaryColour=&HFFFFFF&'[vout]")
                    vmap = "[vout]"
                else:
                    vmap = "0:v"
                # audio mix: lower bgm volume then amix
                bgm_vol = max(-50.0, min(0.0, float(request.bgm_volume_db)))
                filter_parts.append(f"[2:a]volume={10 ** (bgm_vol/20):.6f}[bgm]")
                filter_parts.append("[1:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]")
                cmd += [
                    "-filter_complex", ";".join(filter_parts),
                    "-map", vmap,
                    "-map", "[aout]",
                    "-c:v", "libx264",
                    "-tune", "stillimage",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-shortest", output_path,
                ]
                subprocess.run(cmd, check=True)
            else:
                ffmpeg_command = [
                    "ffmpeg",
                    "-y",
                    "-loop",
                    "1",
                    "-i",
                    image_paths[0],
                    "-i",
                    audio_path,
                ]
                if have_subtitles:
                    ffmpeg_command += [
                        "-vf",
                        f"subtitles={subtitle_path}:force_style='Fontsize=18,PrimaryColour=&HFFFFFF&'",
                    ]
                ffmpeg_command += [
                    "-c:v",
                    "libx264",
                    "-tune",
                    "stillimage",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-pix_fmt",
                    "yuv420p",
                    "-shortest",
                    output_path,
                ]
            subprocess.run(ffmpeg_command, check=True)
        else:
            # Multi-scene: require scene durations
            durations = request.scene_durations or []
            if len(durations) != len(image_paths):
                raise HTTPException(status_code=400, detail="scene_durations length must match images length")

            # Inputs: each image as finite video with -loop 1 -t dur -i img
            cmd: List[str] = ["ffmpeg", "-y"]
            for p, d in zip(image_paths, durations):
                cmd += ["-loop", "1", "-t", str(float(d)), "-i", p]
            # Add audio as last input(s)
            cmd += ["-i", audio_path]
            if have_bgm:
                cmd += ["-i", bgm_path]

            # Build filter_complex: concat all image videos
            # [0:v][1:v]...[n-1:v]concat=n=N:v=1:a=0[vc]; [vc]subtitles=... [vout]
            n = len(image_paths)
            concat_inputs = "".join(f"[{i}:v]" for i in range(n))
            filter_parts = [f"{concat_inputs}concat=n={n}:v=1:a=0[vc]"]
            if have_subtitles:
                filter_parts.append(f"[vc]subtitles={subtitle_path}:force_style='Fontsize=18,PrimaryColour=&HFFFFFF&'[vout]")
                vmap = "[vout]"
            else:
                vmap = "[vc]"

            # Audio mapping/mix
            if have_bgm:
                # narration index is n, bgm is n+1
                bgm_vol = max(-50.0, min(0.0, float(request.bgm_volume_db)))
                filter_parts.append(f"[{n+1}:a]volume={10 ** (bgm_vol/20):.6f}[bgm]")
                filter_parts.append(f"[{n}:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]")
                amap = "[aout]"
            else:
                amap = f"{n}:a:0"

            cmd += [
                "-filter_complex",
                ";".join(filter_parts),
                "-map", vmap,
                "-map", amap,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                output_path,
            ]

            subprocess.run(cmd, check=True)

        # Read output into memory so we can safely clean up temp files
        with open(output_path, "rb") as f_out:
            data = f_out.read()
        t1 = time.perf_counter()
        last_metrics["last_compose_ms"] = int((t1 - t0) * 1000)
        return Response(content=data, media_type="video/mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create video: {e}")
    finally:
        for path in [single_image_path, *image_paths, audio_path, bgm_path, subtitle_path, output_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


@app.get("/health")
async def health():
    return {
        "service": "video_service",
        "ffmpeg": True,
    }


@app.get("/metrics")
async def metrics():
    return {
        "service": "video_service",
        **last_metrics,
    }
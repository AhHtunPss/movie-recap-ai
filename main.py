import os
import uuid
import subprocess
import tempfile
import threading
import base64
import io
from pathlib import Path

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from google import genai


app = FastAPI(title="Movie Recap AI")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE = "https://api.groq.com/openai/v1"

MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024

jobs = {}
jobs_lock = threading.Lock()


def update_job(job_id, **data):
    with jobs_lock:
        jobs.setdefault(job_id, {"status": "PENDING", "progress": 0}).update(data)


def groq_headers():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY ကို Render Environment မှာ ထည့်ပါ။")
    return {"Authorization": f"Bearer {GROQ_API_KEY}"}


def extract_audio(video_path, audio_path):
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "flac", str(audio_path)
        ],
        capture_output=True,
        text=True,
        timeout=900
    )

    if result.returncode != 0:
        raise RuntimeError("ဗီဒီယိုမှ အသံထုတ်ယူမရပါ။ အသံပါသောဗီဒီယိုကို စမ်းပါ။")

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError("ဗီဒီယိုထဲမှ အသံမရရှိပါ။")


def transcribe_audio(audio_path):
    if audio_path.stat().st_size > MAX_AUDIO_BYTES:
        raise RuntimeError(
            "အသံဖိုင်သည် 25 MB ကျော်နေပါသည်။ ပိုတိုသောဗီဒီယိုဖြင့် စမ်းပါ။"
        )

    with audio_path.open("rb") as audio_file:
        response = httpx.post(
            f"{GROQ_BASE}/audio/transcriptions",
            headers=groq_headers(),
            data={
                "model": os.getenv(
                    "GROQ_WHISPER_MODEL",
                    "whisper-large-v3-turbo"
                ),
                "response_format": "json"
            },
            files={
                "file": (
                    audio_path.name,
                    audio_file,
                    "audio/flac"
                )
            },
            timeout=300
        )

    if response.is_error:
        raise RuntimeError(
            f"Groq transcription error: {response.text[:500]}"
        )

    transcript = response.json().get("text", "").strip()

    if not transcript:
        raise RuntimeError("အသံမှ စာသားမရရှိပါ။")

    return transcript


def create_summary(transcript, mode="Storytelling"):
    modes = {
        "Storytelling": "ဇာတ်လမ်းကို စိတ်ဝင်စားဖွယ် ဇာတ်ကြောင်းပြောပုံစံဖြင့် ရေးပါ။",
        "Gen-Z": "လူငယ်များနှင့် သင့်တော်သော သဘာဝကျသည့် မြန်မာစကားပြောပုံစံဖြင့် ရေးပါ။",
        "Documentary": "မှတ်တမ်းရုပ်ရှင်တင်ဆက်မှုပုံစံဖြင့် ရေးပါ။",
        "Fast Recap": "အရေးကြီးဆုံးဖြစ်ရပ်များကို မြန်မြန်ဆန်ဆန် တိုတိုရှင်းရှင်း ရေးပါ။",
        "News": "သတင်းတင်ဆက်သူပုံစံဖြင့် ရေးပါ။"
    }

    instruction = modes.get(mode, modes["Storytelling"])

    response = httpx.post(
        f"{GROQ_BASE}/chat/completions",
        headers=groq_headers(),
        json={
            "model": os.getenv(
                "GROQ_TEXT_MODEL",
                "openai/gpt-oss-120b"
            ),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "သင်သည် မြန်မာဘာသာ Movie Recap စာရေးဆရာဖြစ်သည်။ "
                        + instruction
                        + " Transcript ထဲတွင် မပါသောအချက်များကို မတီထွင်ပါနှင့်။ "
                        "မြန်မာဘာသာဖြင့် ဇာတ်လမ်းအကျဉ်းနှင့် Voiceover Script "
                        "နှစ်ပိုင်းခွဲ၍ ရေးပါ။"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "အောက်ပါ transcript ကို Movie Recap ပြုလုပ်ပါ။\n\n"
                        + transcript[:40000]
                    )
                }
            ],
            "temperature": 0.4,
            "max_tokens": 2500
        },
        timeout=180
    )

    if response.is_error:
        raise RuntimeError(
            f"Groq summary error: {response.text[:500]}"
        )

    return response.json()["choices"][0]["message"]["content"].strip()


def process_video_job(job_id, video_path, mode):
    video = Path(video_path)

    try:
        update_job(
            job_id,
            status="PROGRESS",
            progress=10,
            stage="အသံထုတ်ယူနေသည်"
        )

        with tempfile.TemporaryDirectory(prefix="movie-recap-") as temp_dir:
            audio_path = Path(temp_dir) / "audio.flac"

            extract_audio(video, audio_path)

            update_job(
                job_id,
                status="PROGRESS",
                progress=40,
                stage="အသံမှ စာသားပြောင်းနေသည်"
            )

            transcript = transcribe_audio(audio_path)

            update_job(
                job_id,
                status="PROGRESS",
                progress=75,
                stage="ဇာတ်လမ်းအကျဉ်းရေးနေသည်"
            )

            summary = create_summary(transcript, mode)

            update_job(
                job_id,
                status="SUCCESS",
                progress=100,
                stage="ပြီးပါပြီ",
                result={
                    "transcript": transcript,
                    "summary": summary,
                    "recap": summary
                }
            )

    except Exception as exc:
        update_job(
            job_id,
            status="FAILURE",
            progress=0,
            error=str(exc),
            result={"error": str(exc)}
        )

    finally:
        try:
            video.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
def home():
    index_file = Path(__file__).with_name("index.html")

    if not index_file.exists():
        return HTMLResponse(
            "<h1>Movie Recap AI</h1><p>index.html မတွေ့ပါ။</p>",
            status_code=200
        )

    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok", "service": "movie-recap-ai"}


@app.post("/api/recap")
async def start_recap(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    mode: str = "Storytelling"
):
    filename = (video.filename or "").lower()

    if not filename.endswith((".mp4", ".mov", ".mkv", ".webm")):
        raise HTTPException(
            status_code=400,
            detail="MP4, MOV, MKV သို့မဟုတ် WEBM ဖိုင်ကို ရွေးပါ။"
        )

    job_id = str(uuid.uuid4())

    temp = tempfile.NamedTemporaryFile(
        prefix="movie-recap-",
        suffix=Path(filename).suffix,
        delete=False
    )

    temp_path = Path(temp.name)
    total = 0

    try:
        with temp:
            while True:
                chunk = await video.read(1024 * 1024)

                if not chunk:
                    break

                total += len(chunk)

                if total > MAX_VIDEO_BYTES:
                    temp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail="ဗီဒီယိုဖိုင်သည် 100 MB ထက်ကြီးနေပါသည်။"
                    )

                temp.write(chunk)

    finally:
        await video.close()

    if total == 0:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="ဗီဒီယိုဖိုင်အလွတ်ဖြစ်နေပါသည်။"
        )

    update_job(
        job_id,
        status="PENDING",
        progress=0,
        stage="စောင့်ဆိုင်းနေသည်"
    )

    background_tasks.add_task(
        process_video_job,
        job_id,
        str(temp_path),
        mode
    )

    return {
        "job_id": job_id,
        "status": "PENDING"
    }


@app.get("/api/status/{job_id}")
def recap_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="အလုပ်အခြေအနေ မတွေ့ပါ။ ထပ်မံတင်ပေးပါ။"
            )

        return dict(job)

import os
import subprocess
import tempfile
from pathlib import Path

import httpx
from celery import Celery

REDIS_URL = os.environ["REDIS_URL"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

celery_app = Celery(
    "movie_recap",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
)

GROQ_BASE = "https://api.groq.com/openai/v1"


def groq_headers():
    return {"Authorization": f"Bearer {GROQ_API_KEY}"}


def extract_audio(video_path, audio_path):
    """Extract mono 16kHz audio from the uploaded video."""
    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "flac",
        str(audio_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=900,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg audio extraction failed: "
            + result.stderr[-1500:]
        )

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError("No audio track was extracted.")


def transcribe_audio(audio_path):
    """Send extracted audio to Groq Whisper."""
    if audio_path.stat().st_size > 25 * 1024 * 1024:
        raise RuntimeError(
            "Audio exceeds the 25 MB transcription limit. "
            "Audio chunking is needed for longer videos."
        )

    with audio_path.open("rb") as audio_file:
        response = httpx.post(
            f"{GROQ_BASE}/audio/transcriptions",
            headers=groq_headers(),
            data={
                "model": "whisper-large-v3-turbo",
                "response_format": "json",
            },
            files={
                "file": (
                    audio_path.name,
                    audio_file,
                    "audio/flac",
                )
            },
            timeout=300,
        )

    response.raise_for_status()
    transcript = response.json().get("text", "").strip()

    if not transcript:
        raise RuntimeError("Speech-to-text returned empty text.")

    return transcript


def create_summary(transcript):
    """Generate a Burmese movie recap from the transcript."""
    response = httpx.post(
        f"{GROQ_BASE}/chat/completions",
        headers=groq_headers(),
        json={
            "model": os.getenv(
                "GROQ_TEXT_MODEL",
                "llama-3.3-70b-versatile",
            ),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "သင်သည် မြန်မာဘာသာဖြင့် Movie Recap "
                        "ရေးသားပေးသော စာရေးဆရာဖြစ်သည်။ "
                        "ပေးထားသော transcript ကို အခြေခံပြီး "
                        "ဇာတ်လမ်းအကျဉ်းကို သဘာဝကျသော မြန်မာစကားဖြင့် "
                        "ရေးပါ။ အဓိကဇာတ်ကောင်များ၊ အရေးကြီးသော "
                        "ဖြစ်ရပ်များနှင့် ဇာတ်လမ်းအဆုံးသတ်ကို "
                        "transcript တွင် ပါသလောက်သာ ဖော်ပြပါ။ "
                        "မပါသော အချက်အလက်များကို မတီထွင်ပါနှင့်။ "
                        "ထို့နောက် အသံဖတ်ရန် သင့်တော်သော "
                        "voiceover script ကို သီးခြားရေးပါ။"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "အောက်ပါ transcript ကို မြန်မာလို "
                        "Movie Recap ပြုလုပ်ပေးပါ။\n\n"
                        "TRANSCRIPT:\n" + transcript[:50000]
                    ),
                },
            ],
            "temperature": 0.4,
            "max_tokens": 2500,
        },
        timeout=180,
    )

    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"]

    if not result:
        raise RuntimeError("AI returned an empty summary.")

    return result


@celery_app.task(
    name="tasks.process_video",
    bind=True,
    max_retries=0,
)
def process_video(self, video_path, job_id):
    """Process one video asynchronously."""
    video = Path(video_path)

    if not video.exists():
        raise FileNotFoundError("Uploaded video file not found.")

    try:
        self.update_state(
            state="PROGRESS",
            meta={"stage": "extracting_audio"},
        )

        with tempfile.TemporaryDirectory(
            prefix="movie-recap-"
        ) as temp_dir:
            audio = Path(temp_dir) / "audio.flac"

            extract_audio(video, audio)

            self.update_state(
                state="PROGRESS",
                meta={"stage": "transcribing"},
            )

            transcript = transcribe_audio(audio)

            self.update_state(
                state="PROGRESS",
                meta={"stage": "summarizing"},
            )

            summary = create_summary(transcript)

            return {
                "job_id": job_id,
                "status": "completed",
                "transcript": transcript,
                "summary": summary,
            }

    finally:
        # Uploaded files are temporary; do not retain them.
        video.unlink(missing_ok=True)

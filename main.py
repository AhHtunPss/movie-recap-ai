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
from google.genai import types


app = FastAPI(title="Movie Recap AI")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE = "https://api.groq.com/openai/v1"

MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_VIDEO_FRAMES = 12

jobs = {}
jobs_lock = threading.Lock()


def update_job(job_id, **data):
    with jobs_lock:
        jobs.setdefault(
            job_id,
            {"status": "PENDING", "progress": 0}
        ).update(data)


def groq_headers():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY ကို Render Environment မှာ ထည့်ပါ။"
        )
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
        raise RuntimeError(
            "ဗီဒီယိုမှ အသံထုတ်ယူမရပါ။ အသံပါသောဗီဒီယိုကို စမ်းပါ။"
        )

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError("ဗီဒီယိုထဲမှ အသံမရရှိပါ။")


def transcribe_audio(audio_path):
    if audio_path.stat().st_size > MAX_AUDIO_BYTES:
        raise RuntimeError(
            "အသံဖိုင်သည် 25 MB ကျော်နေပါသည်။ "
            "ပိုတိုသောဗီဒီယိုဖြင့် စမ်းပါ။"
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


def get_video_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        raise RuntimeError(
            "ဗီဒီယိုကြာချိန်ကို မဖတ်နိုင်ပါ။ FFprobe ကို စစ်ဆေးပါ။"
        )

    try:
        duration = float(result.stdout.strip())
    except (ValueError, TypeError):
        raise RuntimeError("ဗီဒီယိုကြာချိန် မမှန်ပါ။")

    if duration <= 0:
        raise RuntimeError("ဗီဒီယိုကြာချိန် မမှန်ပါ။")

    return duration


def extract_video_frames(video_path, frames_dir):
    """
    ဗီဒီယိုအစမှအဆုံးအထိ အချိန်အလိုက် ဖရိမ် 12 ခုအထိ ရွေးထုတ်ပါ။
    """

    duration = get_video_duration(video_path)

    frame_count = min(
        MAX_VIDEO_FRAMES,
        max(1, int(duration // 5) + 1)
    )

    if frame_count == 1:
        timestamps = [duration / 2]
    else:
        timestamps = [
            duration * i / (frame_count - 1)
            for i in range(frame_count)
        ]

    frames = []

    for index, timestamp in enumerate(timestamps):
        frame_path = frames_dir / f"frame_{index:02d}.jpg"

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{timestamp:.3f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-vf", "scale=768:-2",
                "-q:v", "5",
                str(frame_path)
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        if (
            result.returncode == 0
            and frame_path.exists()
            and frame_path.stat().st_size > 0
        ):
            frames.append((timestamp, frame_path))

    if not frames:
        raise RuntimeError(
            "ဗီဒီယိုပုံရိပ် မထုတ်ယူနိုင်ပါ။ "
            "ဗီဒီယိုဖိုင်နှင့် FFmpeg ကို စစ်ဆေးပါ။"
        )

    return frames


def analyze_video_frames(video_path, frames_dir):
    """
    Gemini Vision သို့ ရွေးထားသော ဗီဒီယိုဖရိမ်များ ပို့ပြီး
    မြင်ရသောဖြစ်ရပ်များကို မြန်မာဘာသာဖြင့် ဖော်ပြစေသည်။
    """

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "ဗီဒီယိုပုံရိပ်ခွဲခြမ်းစိတ်ဖြာရန် "
            "Render Environment မှာ GEMINI_API_KEY ထည့်ပါ။"
        )

    frames = extract_video_frames(video_path, frames_dir)
    client = genai.Client(api_key=api_key)

    contents = [
        types.Part.from_text(
            text=(
                "သင်သည် ရုပ်ရှင်ဗီဒီယိုပုံရိပ်ကို ခွဲခြမ်းစိတ်ဖြာသူဖြစ်သည်။ "
                "ပေးထားသောပုံများကို အချိန်အစဉ်အတိုင်း ကြည့်ပါ။ "
                "မြန်မာဘာသာဖြင့် အစီအစဉ်တကျ ဖော်ပြပါ။\n\n"
                "ပုံတစ်ပုံစီအတွက်:\n"
                "၁။ မြင်ရသောနေရာနှင့် ပတ်ဝန်းကျင်\n"
                "၂။ ဇာတ်ကောင်များ၏ မြင်ရသောမျက်နှာအမူအရာ\n"
                "၃။ ကိုယ်ဟန်အနေအထား၊ လက်ခြေလှုပ်ရှားမှုနှင့် လုပ်ဆောင်ချက်\n"
                "၄။ ဇာတ်ကောင်များ အပြန်အလှန်ပြုမူပုံ\n"
                "၅။ ပုံမှန်မဟုတ်သော အရေးကြီးဖြစ်ရပ်\n\n"
                "ပုံထဲမှာ မမြင်ရသောအချက်ကို မတီထွင်ပါနှင့်။ "
                "မျက်နှာအမူအရာတစ်ခုတည်းမှ စိတ်ခံစားချက်ကို "
                "သေချာသည်ဟု မဆိုပါနှင့်။ မသေချာလျှင် မသေချာကြောင်း ဖော်ပြပါ။ "
                "ဇာတ်ကောင်အမည် မသိလျှင် အမည်မတီထွင်ပါနှင့်။ "
                "ပုံများကို တစ်ပုံချင်း အချိန်နှင့်တွဲဖော်ပြပါ။"
            )
        )
    ]

    for timestamp, frame_path in frames:
        contents.append(
            types.Part.from_text(
                text=f"\n\nဗီဒီယိုအချိန်: {timestamp:.1f} စက္ကန့်"
            )
        )

        contents.append(
            types.Part.from_bytes(
                data=frame_path.read_bytes(),
                mime_type="image/jpeg"
            )
        )

    response = client.models.generate_content(
        model=os.getenv(
            "GEMINI_VISION_MODEL",
            "gemini-2.5-flash"
        ),
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=5000
        )
    )

    visual_text = (response.text or "").strip()

    if not visual_text:
        raise RuntimeError(
            "Gemini Vision မှ ပုံရိပ်ခွဲခြမ်းစိတ်ဖြာချက် မရရှိပါ။"
        )

    return visual_text


def create_summary(transcript, mode="Storytelling", visual_analysis=""):
    modes = {
        "Storytelling": (
            "ရုပ်ရှင်ဇာတ်လမ်းကို စိတ်ဝင်စားဖွယ် "
            "ဇာတ်ကြောင်းပြောသူပုံစံဖြင့် ရေးပါ။ "
            "ဇာတ်ကောင်များ၏ မြင်ရသောအမူအရာနှင့် လှုပ်ရှားမှုများကို "
            "ဇာတ်ကြောင်းထဲ သဘာဝကျကျ ထည့်ပါ။"
        ),
        "Gen-Z": (
            "လူငယ်များနှင့် သင့်တော်သော သဘာဝကျသည့် "
            "မြန်မာစကားပြောပုံစံဖြင့် ရေးပါ။ "
            "ဇာတ်ကောင်များ၏ မြင်ရသောလှုပ်ရှားမှုများကို "
            "အသက်ဝင်စွာ ဖော်ပြပါ။ အလွန်အကျွံ slang မသုံးပါနှင့်။"
        ),
        "Documentary": (
            "မှတ်တမ်းရုပ်ရှင်တင်ဆက်မှုပုံစံဖြင့် ရေးပါ။ "
            "မြင်ရသောအဖြစ်အပျက်များကို တိကျစွာဖော်ပြပါ။ "
            "သက်သေမရှိသော စိတ်ခံစားချက်နှင့် အကြောင်းရင်းများကို "
            "အမှန်တရားကဲ့သို့ မတင်ပြပါနှင့်။"
        ),
        "Fast Recap": (
            "အရေးကြီးဆုံးဖြစ်ရပ်များကို မြန်မြန်ဆန်ဆန် "
            "တိုတိုရှင်းရှင်း ရေးပါ။ "
            "ဇာတ်လမ်းနားလည်ရန် လိုအပ်သော အမူအရာနှင့် "
            "လှုပ်ရှားမှုများကို မချန်ပါနှင့်။"
        ),
        "News": (
            "သတင်းတင်ဆက်သူပုံစံဖြင့် ရေးပါ။ "
            "အဖြစ်အပျက်များကို အစီအစဉ်တကျ တိကျစွာဖော်ပြပါ။"
        )
    }

    instruction = modes.get(mode, modes["Storytelling"])

    visual_section = (
        visual_analysis[:24000]
        if visual_analysis
        else "ဗီဒီယိုပုံရိပ်အချက်အလက် မရရှိပါ။"
    )

    system_prompt = """
သင်သည် မြန်မာဘာသာဖြင့် သဘာဝကျသော Movie Recap ရေးသားသူဖြစ်သည်။

အောက်ပါစည်းမျဉ်းများကို တိကျစွာလိုက်နာပါ။

၁။ Transcript နှင့် ဗီဒီယိုပုံရိပ်ခွဲခြမ်းစိတ်ဖြာချက်ကို ပေါင်းစပ်ပါ။
၂။ စကားမပြောသောအခန်းများတွင် မြင်ရသော မျက်နှာအမူအရာ၊
   ကိုယ်ဟန်အနေအထား၊ လှုပ်ရှားမှု၊ နေရာနှင့် ဖြစ်ရပ်များကို ဖော်ပြပါ။
၃။ မြင်ရသောအချက်နှင့် အသံမှသိရသောအချက်ကိုသာ အခြေခံပါ။
၄။ ပုံရိပ်နှင့် Transcript မတူညီလျှင် မသေချာသောအချက်ကို
   အတည်ပြုဖြစ်ရပ်အဖြစ် မရေးပါနှင့်။
၅။ မရှိသောဇာတ်ကောင်၊ နာမည်၊ စကားပြောခန်း၊ ဖြစ်ရပ်ကို မတီထွင်ပါနှင့်။
၆။ မသက်ဆိုင်သော အင်္ဂလိပ်စကားလုံးများ မထည့်ပါနှင့်။
၇။ စာကြောင်းတစ်ကြောင်း သို့မဟုတ် ဖြစ်ရပ်တစ်ခုကို ထပ်ခါထပ်ခါ မရေးပါနှင့်။
၈။ Prompt၊ AI ညွှန်ကြားချက်၊ မှတ်ချက်နှင့် ကိုယ်တိုင်ရှင်းပြချက်
   များကို နောက်ဆုံးအဖြေထဲ မထည့်ပါနှင့်။
၉။ မြန်မာစာကို သဘာဝကျပြီး ဖတ်ရလွယ်အောင် ရေးပါ။
၁၀။ Voiceover အတွက် အဆင်ပြေသော စာကြောင်းအရှည်ကို သုံးပါ။
၁၁။ မျက်နှာအမူအရာကို ဖော်ပြရာတွင် မမြင်ရသော စိတ်အတွင်းခံစားချက်
    ကို သေချာသည်ဟု မတင်ပြပါနှင့်။
၁၂။ အကျဉ်းချုပ်နှင့် Voiceover Script နှစ်ပိုင်းကိုသာ ထုတ်ပါ။

Output format:

ဇာတ်လမ်းအကျဉ်း:
(ဇာတ်လမ်းတစ်ခုလုံးကို အကျဉ်းချုပ်ပါ)

Voiceover Script:
(ဖတ်ပြရန်အဆင်သင့်ဖြစ်သော ဇာတ်ကြောင်းကို ရေးပါ။
ဇာတ်ကောင်၏ စကားကို လိုအပ်သည့်အခါသာ ထည့်ပါ။
မလိုအပ်သော ခေါင်းစဉ်များ မထပ်ထည့်ပါ။)
"""

    user_prompt = f"""
ရွေးချယ်ထားသောပုံစံ: {mode}

ပုံစံအတွက် သီးသန့်ညွှန်ကြားချက်:
{instruction}

အသံမှရရှိသော Transcript:
{transcript[:24000]}

ဗီဒီယိုပုံရိပ်ခွဲခြမ်းစိတ်ဖြာချက်:
{visual_section}

အထက်ပါအချက်အလက်များကို အခြေခံ၍ မြန်မာဘာသာဖြင့်
ဇာတ်လမ်းအကျဉ်းနှင့် Voiceover Script ကို ရေးပါ။
Transcript နှင့် ပုံရိပ်များမှ မသိနိုင်သော ဖြစ်ရပ်များကို မတီထွင်ပါနှင့်။
"""

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
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        },
        timeout=180
    )

    if response.is_error:
        raise RuntimeError(
            f"Groq summary error: {response.text[:500]}"
        )

    return (
        response.json()["choices"][0]["message"]["content"].strip()
    )


def process_video_job(job_id, video_path, mode):
    video = Path(video_path)

    try:
        update_job(
            job_id,
            status="PROGRESS",
            progress=10,
            stage="ဗီဒီယိုအသံ ထုတ်ယူနေသည်"
        )

        with tempfile.TemporaryDirectory(
            prefix="movie-recap-"
        ) as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "audio.flac"
            frames_dir = temp_path / "frames"
            frames_dir.mkdir()

            extract_audio(video, audio_path)

            update_job(
                job_id,
                status="PROGRESS",
                progress=30,
                stage="အသံမှ စာသားပြောင်းနေသည်"
            )

            transcript = transcribe_audio(audio_path)

            update_job(
                job_id,
                status="PROGRESS",
                progress=50,
                stage="ဗီဒီယိုပုံရိပ်များ စစ်ဆေးနေသည်"
            )

            visual_analysis = analyze_video_frames(
                video,
                frames_dir
            )

            update_job(
                job_id,
                status="PROGRESS",
                progress=80,
                stage="အသံနှင့် ပုံရိပ်ပေါင်းစပ်၍ ဇာတ်ညွှန်းရေးနေသည်"
            )

            summary = create_summary(
                transcript,
                mode,
                visual_analysis
            )

            update_job(
                job_id,
                status="SUCCESS",
                progress=100,
                stage="ပြီးပါပြီ",
                result={
                    "transcript": transcript,
                    "visual_analysis": visual_analysis,
                    "summary": summary,
                    "recap": summary
                }
            )

    except Exception as exc:
        print("Movie recap error:", repr(exc))
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

    return HTMLResponse(
        index_file.read_text(encoding="utf-8")
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "movie-recap-ai"
    }


@app.post("/api/tts")
async def create_tts(payload: dict):
    text = str(payload.get("text", "")).strip()
    voice = str(payload.get("voice", "Kore")).strip()
    style = str(
        payload.get(
            "style",
            "Natural, clear Burmese movie recap narration"
        )
    ).strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="အသံပြောင်းရန် စာသားထည့်ပါ။"
        )

    if len(text) > 8000:
        raise HTTPException(
            status_code=400,
            detail="စာသားသည် 8000 characters ထက် မကျော်ရပါ။"
        )

    allowed_voices = {
        "Kore", "Aoede", "Charon", "Sulafat"
    }

    if voice not in allowed_voices:
        voice = "Kore"

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Render Environment မှာ GEMINI_API_KEY ထည့်ပါ။"
        )

    try:
        client = genai.Client(api_key=api_key)

        interaction = client.interactions.create(
            model="gemini-3.8-flash-tts",
            input=[{
                "type": "user_input",
                "content": [{
                    "type": "text",
                    "text": text,
                    "annotations": [{
                        "type": "speech_metadata",
                        "style": style
                    }]
                }]
            }],
            response_format={"type": "audio"},
            generation_config={
                "speech_config": [
                    {"voice": voice}
                ]
            }
        )

        audio_data = interaction.output_audio.data

        if not audio_data:
            raise RuntimeError("Gemini မှ အသံဒေတာမရရှိပါ။")

        wav_bytes = base64.b64decode(audio_data)

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", "pipe:0",
                "-vn",
                "-codec:a", "libmp3lame",
                "-b:a", "128k",
                "-f", "mp3",
                "pipe:1"
            ],
            input=wav_bytes,
            capture_output=True,
            timeout=180
        )

        if result.returncode != 0 or not result.stdout:
            error = result.stderr.decode(
                "utf-8", errors="replace"
            )[-1000:]

            raise RuntimeError(
                "WAV မှ MP3 ပြောင်းမရပါ။ " + error
            )

        return StreamingResponse(
            io.BytesIO(result.stdout),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition":
                    'attachment; filename="movie-recap-voiceover.mp3"',
                "Cache-Control": "no-store"
            }
        )

    except HTTPException:
        raise

    except Exception as exc:
        print("Gemini TTS error:", repr(exc))

        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini အသံထုတ်ရာတွင် အမှားဖြစ်နေပါသည်: "
                + str(exc)[:400]
            )
        )


@app.post("/api/recap")
async def start_recap(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    mode: str = "Storytelling"
):
    filename = (video.filename or "").lower()

    if not filename.endswith(
        (".mp4", ".mov", ".mkv", ".webm")
    ):
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
                        detail=(
                            "ဗီဒီယိုဖိုင်သည် 100 MB ထက်ကြီးနေပါသည်။"
                        )
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


import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import lery
from celery import Celery
from celery.result import AsyncResult

app = FastAPI(title="Movie Recap AI Backend")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery(
    "movie_recap",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

UPLOAD_DIR = Path("/tmp/movie-recap-uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_SIZE = 100 * 1024 * 1024  # 100 MB


@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path(__file__).parent / "index.html"

    if not html_path.exists():
        raise HTTPException(
            status_code=500,
            detail="index.html file not found"
        )

    return html_path.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/recap")
async def create_recap(video: UploadFile = File(...)):
    allowed = {".mp4", ".mov", ".mkv", ".webm"}
    ext = Path(video.filename or "").suffix.lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail="MP4, MOV, MKV, or WEBM video required"
        )

    job_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{job_id}{ext}"
    size = 0

    try:
        with path.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)

                if size > MAX_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video must be 100 MB or smaller"
                    )

                out.write(chunk)

        task = celery_app.send_task(
            "tasks.process_video",
            args=[str(path), job_id]
        )

        return {
            "job_id": task.id,
            "status": "queued",
            "message": "Video accepted for processing"
        }

    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await video.close()


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    task = AsyncResult(job_id, app=celery_app)

    response = {"job_id": job_id, "status": task.status}

    if task.successful():
        response["result"] = task.result
    elif task.failed():
        response["error"] = str(task.result)

    return response

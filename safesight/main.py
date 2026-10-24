"""FastAPI application. Use one uvicorn worker (camera threads live in process)."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SAFESIGHT_DATA", str(ROOT / "data"))).resolve()
CONFIG = Path(os.environ.get("SAFESIGHT_CONFIG", str(ROOT / "config/cameras.json")))
MODE = os.environ.get("SAFESIGHT_MODE", "demo").strip().lower()
MODEL = os.environ.get("SAFESIGHT_MODEL", "")
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = Pipeline(CONFIG, DATA_DIR, MODE, MODEL)
    pipeline.start()
    try:
        yield
    finally:
        pipeline.stop()
        pipeline = None


app = FastAPI(title="SafeSight MVP", version="0.1.0", lifespan=lifespan)
app.mount("/evidence", StaticFiles(directory=str(DATA_DIR / "evidence"), check_dir=False), name="evidence")


@app.get("/")
def homepage():
    return FileResponse(ROOT / "safesight/static/index.html")


@app.get("/api/status")
def status():
    return pipeline.status()


@app.get("/api/events")
def events(camera_id: str | None = None, zone_id: str | None = None,
           violation_type: str | None = None, start: str | None = None, end: str | None = None,
           limit: int = Query(100, ge=1, le=500)):
    try:
        return pipeline.store.list_events(camera_id, zone_id, violation_type, start, end, limit)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/frame/{camera_id}")
def frame(camera_id: str):
    worker = pipeline.workers.get(camera_id)
    if worker is None:
        raise HTTPException(404, "Unknown camera")
    with worker.lock:
        jpg = worker.state.frame_jpeg
    if jpg is None:
        raise HTTPException(503, "Frame not ready")
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

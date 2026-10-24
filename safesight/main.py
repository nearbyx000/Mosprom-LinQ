"""FastAPI dashboard. Run one server process to share one camera and TPU budget."""
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent


def create_app(config_path=None, data_dir=None, mode=None, camera_factory=None, client_factory=None):
    config_path = Path(config_path or os.getenv("SAFESIGHT_CONFIG", str(ROOT / "config/cameras.json")))
    data_dir = Path(data_dir or os.getenv("SAFESIGHT_DATA", str(ROOT / "data"))).resolve()
    mode = mode or os.getenv("SAFESIGHT_MODE", "preview").strip().lower()

    @asynccontextmanager
    async def lifespan(app):
        options = {}
        if camera_factory is not None:
            options["camera_factory"] = camera_factory
        if client_factory is not None:
            options["client_factory"] = client_factory
        pipeline = Pipeline(config_path, data_dir, mode, os.getenv("SAFESIGHT_MODEL", ""), **options)
        app.state.pipeline = pipeline
        pipeline.start()
        try:
            yield
        finally:
            await asyncio.to_thread(pipeline.stop)

    app = FastAPI(title="SafeSight / LinQ", version="0.2.0", lifespan=lifespan)
    app.mount("/evidence", StaticFiles(directory=str(data_dir / "evidence"), check_dir=False), name="evidence")
    app.mount("/static", StaticFiles(directory=str(ROOT / "web")), name="static")
    app.mount("/dashboard", StaticFiles(directory=str(ROOT / "safesight/static")), name="dashboard")

    def worker_for(camera_id):
        worker = app.state.pipeline.workers.get(camera_id)
        if worker is None:
            raise HTTPException(404, "Unknown camera")
        return worker

    @app.get("/", response_class=FileResponse)
    def home():
        return FileResponse(ROOT / "safesight/static/index.html", media_type="text/html")

    @app.get("/camera", response_class=FileResponse)
    def simple_preview():
        return FileResponse(ROOT / "web/index.html", media_type="text/html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/status")
    def status():
        return app.state.pipeline.status()

    @app.get("/api/events")
    def events(camera_id: str | None = None, zone_id: str | None = None,
               violation_type: str | None = None, start: str | None = None, end: str | None = None,
               limit: int = Query(100, ge=1, le=500)):
        try:
            return app.state.pipeline.store.list_events(camera_id, zone_id, violation_type, start, end, limit)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/frame/{camera_id}")
    def frame(camera_id: str):
        worker = worker_for(camera_id)
        _, packet = worker.preview.get(timeout=0)
        if packet is None or time.perf_counter() - packet[1] > 2:
            raise HTTPException(503, "Camera unavailable")
        return Response(packet[2], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/detection/{camera_id}")
    def detection(camera_id: str):
        worker = worker_for(camera_id)
        with worker.lock:
            jpg = worker.detection_jpeg
            sequence = worker.state["last_detection_frame"]
        if jpg is None:
            raise HTTPException(503, "No detection result yet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store", "X-Frame-Sequence": str(sequence)})

    async def stream_response(request, worker):
        sequence, packet = await asyncio.to_thread(worker.preview.get, 0, 2)
        if packet is None or time.perf_counter() - packet[1] > 2:
            raise HTTPException(503, "Camera unavailable")

        async def chunks():
            nonlocal sequence, packet
            while not worker.stopped.is_set() and not await request.is_disconnected():
                if packet is not None:
                    frame_id, captured_at, jpg = packet
                    age_ms = (time.perf_counter() - captured_at) * 1000
                    if age_ms > 2000:
                        break
                    worker.metrics.observe("capture_to_stream_yield", age_ms)
                    yield (f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n"
                           f"X-Frame-Sequence: {frame_id}\r\nX-Server-Frame-Age-Ms: {age_ms:.2f}\r\n\r\n").encode() + jpg + b"\r\n"
                sequence, packet = await asyncio.to_thread(worker.preview.get, sequence, 0.3)
                if packet is None:
                    _, last = worker.preview.get(timeout=0)
                    if last is not None and time.perf_counter() - last[1] > 2:
                        break
            yield b"--frame--\r\n"

        return StreamingResponse(chunks(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store, max-age=0", "X-Accel-Buffering": "no"})

    @app.get("/api/stream/{camera_id}")
    async def stream(camera_id: str, request: Request):
        return await stream_response(request, worker_for(camera_id))

    @app.get("/video")
    async def video(request: Request):
        worker = next(iter(app.state.pipeline.workers.values()))
        return await stream_response(request, worker)

    return app


app = create_app()

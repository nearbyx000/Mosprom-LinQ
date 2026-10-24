"""FastAPI dashboard. Run one server process to share one camera and TPU budget."""
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field

from .pipeline import Pipeline
from .detections import test_message
from .exports import event_csv
from .auth import OperatorAuth
from . import __version__

ROOT = Path(__file__).resolve().parent.parent


class EventReview(BaseModel):
    decision: Literal['confirmed','false_positive','needs_review']
    note: str = Field(default='',max_length=1000)


def create_app(config_path=None, data_dir=None, mode=None, camera_factory=None, client_factory=None):
    config_path = Path(config_path or os.getenv("SAFESIGHT_CONFIG", str(ROOT / "config/cameras.json")))
    mode = mode or os.getenv("SAFESIGHT_MODE", "preview").strip().lower()
    default_data = ROOT/'data'/'demo' if mode=='demo' else ROOT/'data'
    data_dir = Path(data_dir or os.getenv("SAFESIGHT_DATA", str(default_data))).resolve()

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

    app = FastAPI(title="SafeSight / LinQ", version=__version__, lifespan=lifespan)
    app.add_middleware(OperatorAuth)
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

    @app.get("/api/warnings")
    def warnings():
        cameras = []
        for worker in app.state.pipeline.workers.values():
            state = worker.status()
            cameras.append({"camera_id": worker.id, "source": worker.mode,
                            "fresh": state["warnings_fresh"], "updated_at": state["warnings_updated_at"],
                            "safetyvision_expected": state["safetyvision_expected"],
                            "safetyvision_fresh": state["safetyvision_fresh"],
                            "frame_id": state["warnings_frame_id"], "warnings": state["active_warnings"]})
        return {"cameras": cameras, "active_count": sum(len(c["warnings"]) for c in cameras)}

    @app.get("/api/events")
    def events(camera_id: str | None = None, zone_id: str | None = None,
               violation_type: str | None = None, start: str | None = None, end: str | None = None,
               limit: int = Query(100, ge=1, le=500)):
        try:
            return app.state.pipeline.store.list_events(camera_id, zone_id, violation_type, start, end, limit)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/api/events/export.csv')
    def export_events(camera_id: str | None = None, zone_id: str | None = None,
                      violation_type: str | None = None, start: str | None = None,
                      end: str | None = None, limit: int = Query(100,ge=1,le=500)):
        try:
            rows = app.state.pipeline.store.list_events(camera_id,zone_id,violation_type,start,end,limit)
            return Response(event_csv(rows),media_type='text/csv; charset=utf-8',
                            headers={'Content-Disposition':'attachment; filename="safesight-events.csv"',
                                     'Cache-Control':'no-store'})
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc

    @app.get('/api/summary')
    def summary():
        result = app.state.pipeline.store.summary()
        result.update(source_mode=app.state.pipeline.mode,
                      outbox_pending=sum(w.recorder.pending+w.recorder.invalid for w in app.state.pipeline.workers.values()),
                      outbox_invalid=sum(w.recorder.invalid for w in app.state.pipeline.workers.values()))
        return result

    @app.patch('/api/events/{event_id}/review')
    def review_event(event_id: int,body: EventReview,request: Request):
        try:
            return app.state.pipeline.store.review(event_id,body.decision,body.note,
                                                   request.scope.get('safesight_operator','local_operator'))
        except LookupError as exc:
            raise HTTPException(404,str(exc)) from exc

    @app.get('/api/events/{event_id}/reviews')
    def event_reviews(event_id: int):
        return app.state.pipeline.store.reviews(event_id)

    @app.websocket("/ws/detections/{camera_id}")
    async def detection_stream(websocket: WebSocket, camera_id: str, test: bool = False):
        worker = app.state.pipeline.workers.get(camera_id)
        if worker is None:
            await websocket.close(code=1008, reason="Unknown camera")
            return
        await websocket.accept()
        sequence, last_sent = 0, 0.
        try:
            while not worker.stopped.is_set():
                if test:
                    sequence += 1
                    _, raw = worker.raw.get(timeout=0)
                    height, width = raw.image.shape[:2] if raw else (480, 640)
                    await websocket.send_json(test_message(camera_id, sequence, width, height))
                    await asyncio.sleep(.5)
                    continue
                sequence, update = await asyncio.to_thread(worker.detection_updates.get, sequence, .25)
                if update is not None:
                    captured_at, payload = update
                    payload = dict(payload, result_age_ms=max(0, (time.perf_counter() - captured_at) * 1000))
                    await websocket.send_json(payload)
                    last_sent = time.perf_counter()
                elif time.perf_counter() - last_sent >= 1:
                    await websocket.send_json({"type": "status", "camera_id": camera_id,
                                               "source": worker.mode,
                                               "inference_status": worker.status()["inference_status"]})
                    last_sent = time.perf_counter()
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass

    @app.get("/api/frame/{camera_id}")
    def frame(camera_id: str):
        worker = worker_for(camera_id)
        _, packet = worker.preview.get(timeout=0)
        if packet is None or time.perf_counter() - packet[1] > 2:
            raise HTTPException(503, "Camera unavailable")
        return Response(packet[2], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/detection/{camera_id}")
    def detection(camera_id: str, frame: int | None = Query(None, ge=0), generation: int | None = Query(None, ge=0)):
        worker = worker_for(camera_id)
        with worker.lock:
            jpg = worker.detection_jpeg
            sequence = worker.state["last_detection_frame"]
            if frame is not None:
                matches = [(key,image) for key,image in worker.detection_frames.items()
                           if key[1]==frame and (generation is None or key[0]==generation)]
                if not matches:
                    raise HTTPException(404, 'Requested detection frame has expired; use the saved event evidence')
                (gen,sequence),jpg = matches[-1]
            else:
                gen = next(reversed(worker.detection_frames), (0,0))[0]
        if jpg is None:
            raise HTTPException(503, "No detection result yet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store",
                        "X-Frame-Sequence": str(sequence), 'X-Frame-Generation':str(gen)})

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

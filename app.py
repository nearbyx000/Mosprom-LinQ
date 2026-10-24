"""SafeSight MVP: local/network camera preview through FastAPI.

This module deliberately does not include LinQ TPU or YOLO inference yet.
"""

from __future__ import annotations

import logging
import asyncio
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import cv2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from phone_stream import PhoneFeed, MAX_FRAME_BYTES, sanitize_jpeg

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application):
    application.state.phone_feed = PhoneFeed()
    yield

app = FastAPI(
    title="SafeSight Backend",
    description="MVP backend for previewing a webcam or an MJPEG/IP camera.",
    version="0.1.0",
    lifespan=lifespan,
)
WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def camera_source() -> int | str:
    """Camera index (e.g. 0) or a network stream URL set via CAMERA_SOURCE."""
    source = os.environ.get("CAMERA_SOURCE", "0").strip()
    if not source:
        raise ValueError("CAMERA_SOURCE must not be empty")
    if source.isdecimal():
        return int(source)
    if not source.startswith(("http://", "https://", "rtsp://")):
        raise ValueError("CAMERA_SOURCE must be a numeric camera index or an HTTP(S)/RTSP URL")
    return source


def open_camera() -> cv2.VideoCapture:
    source = camera_source()
    if isinstance(source, int) and sys.platform == "darwin":
        return cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    if isinstance(source, int) and sys.platform == "win32":
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


@app.get("/health")
def health() -> dict[str, str]:
    """Server liveness (does not validate camera access)."""
    return {"status": "ok"}


@app.get("/", response_class=FileResponse)
def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


def encode_frame(capture: cv2.VideoCapture) -> bytes:
    success, frame = capture.read()
    if not success:
        raise ValueError("Camera did not return a frame")
    if os.environ.get("CAMERA_MIRROR", "1") == "1":
        frame = cv2.flip(frame, 1)
    encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not encoded:
        raise ValueError("Could not encode camera frame")
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"


@app.get("/video")
def video() -> StreamingResponse:
    """Serve MJPEG frames. Use one viewer at a time in this MVP."""
    try:
        capture = open_camera()
    except (ValueError, cv2.error) as exc:
        logger.warning("Could not initialize the camera. Check CAMERA_SOURCE.")
        raise HTTPException(status_code=503, detail="Camera could not be initialized") from exc

    if not capture.isOpened():
        capture.release()
        logger.warning("Camera unavailable. Check camera permissions or CAMERA_SOURCE.")
        raise HTTPException(status_code=503, detail="Camera unavailable")

    # Validate a real JPEG before sending HTTP 200, so the UI can report failures.
    try:
        first_frame = encode_frame(capture)
    except (ValueError, cv2.error) as exc:
        capture.release()
        logger.warning("Camera opened but could not produce a JPEG frame.")
        raise HTTPException(status_code=503, detail="Camera could not produce a frame") from exc

    async def generate_frames() -> AsyncIterator[bytes]:
        try:
            yield first_frame
            while True:
                # Keep device reads outside the event loop so a disconnect can
                # cancel the response and other API requests remain responsive.
                yield await run_in_threadpool(encode_frame, capture)
        except (ValueError, cv2.error):
            logger.warning("Camera stream ended or failed to encode a frame")
            yield b"--frame--\r\n"
        finally:
            capture.release()

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, max-age=0"},
        # Also release if the client disconnects before the generator starts.
        background=BackgroundTask(capture.release),
    )


@app.websocket("/phone/publish")
async def publish_phone(websocket: WebSocket):
    origin = urlsplit(websocket.headers.get("origin", ""))
    if origin.scheme not in ("http", "https") or origin.netloc.lower() != websocket.headers.get("host", "").lower():
        await websocket.close(code=1008)
        return
    await websocket.accept()
    feed = websocket.app.state.phone_feed
    owner = await feed.claim()
    if owner is None:
        await websocket.close(code=1008, reason="Another phone is already publishing")
        return
    try:
        await websocket.send_json({"ready": True})
        while True:
            packet = await asyncio.wait_for(websocket.receive(), timeout=15)
            if packet["type"] == "websocket.disconnect":
                break
            body = packet.get("bytes")
            if body is None:
                await websocket.close(code=1003, reason="Expected binary JPEG")
                break
            if len(body) > MAX_FRAME_BYTES:
                await websocket.close(code=1009, reason="Frame too large")
                break
            try:
                jpeg = await run_in_threadpool(sanitize_jpeg, body)
            except Exception:
                await websocket.close(code=1003, reason="Invalid JPEG")
                break
            await feed.publish(owner, jpeg)
            await websocket.send_json({"received": feed.sequence})
            await asyncio.sleep(.1)
    except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
        pass
    finally:
        await feed.release(owner)
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.get("/phone/video")
async def phone_video(request: Request):
    feed = request.app.state.phone_feed
    first = await feed.next_frame()
    if first is None:
        raise HTTPException(503, "Phone camera is not publishing")

    async def frames():
        current = first
        while current is not None:
            sequence, jpeg = current
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            current = await feed.next_frame(sequence, timeout=15)
        yield b"--frame--\r\n"

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store, max-age=0"})

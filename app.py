"""SafeSight MVP: local/network camera preview through FastAPI.

This module deliberately does not include LinQ TPU or YOLO inference yet.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


app = FastAPI(
    title="SafeSight Backend",
    description="MVP backend for previewing a webcam or an MJPEG/IP camera.",
    version="0.1.0",
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

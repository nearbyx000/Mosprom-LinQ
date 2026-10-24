"""Tests that do not require an actual camera device."""

from unittest.mock import patch
import asyncio

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as backend


class FakeCamera:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.read_count = 0

    def isOpened(self):
        return self.opened

    def read(self):
        self.read_count += 1
        if self.read_count == 1:
            return True, np.zeros((24, 32, 3), dtype=np.uint8)
        return False, None

    def release(self):
        self.released = True


def test_health():
    with TestClient(backend.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home():
    with TestClient(backend.app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "SafeSight" in response.text
    assert 'id="source"' in response.text
    assert 'id="device-video"' in response.text
    assert 'id="reconnect"' in response.text


@pytest.mark.parametrize("path,content_type", [
    ("/static/styles.css", "text/css"),
    ("/static/app.js", "javascript"),
])
def test_frontend_assets(path, content_type):
    with TestClient(backend.app) as client:
        response = client.get(path)
    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert response.content


def test_camera_unavailable():
    fake = FakeCamera(opened=False)
    with patch.object(backend, "open_camera", return_value=fake):
        with TestClient(backend.app) as client:
            response = client.get("/video")
    assert response.status_code == 503
    assert response.json()["detail"] == "Camera unavailable"
    assert fake.released


def test_video_returns_jpeg_stream():
    fake = FakeCamera()
    with patch.object(backend, "open_camera", return_value=fake):
        with TestClient(backend.app) as client:
            response = client.get("/video")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"--frame\r\nContent-Type: image/jpeg" in response.content
    assert b"\xff\xd8" in response.content
    assert response.content.endswith(b"--frame--\r\n")
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert fake.released


def test_camera_source_index(monkeypatch):
    monkeypatch.setenv("CAMERA_SOURCE", "0")
    assert backend.camera_source() == 0


def test_camera_source_network(monkeypatch):
    monkeypatch.setenv("CAMERA_SOURCE", "http://192.168.1.105:8080/video")
    assert backend.camera_source() == "http://192.168.1.105:8080/video"


@pytest.mark.parametrize("source", ["", "  ", "-1", "camera.mp4", "ftp://camera/video"])
def test_invalid_camera_source(monkeypatch, source):
    monkeypatch.setenv("CAMERA_SOURCE", source)
    with TestClient(backend.app) as client:
        response = client.get("/video")
    assert response.status_code == 503


def test_camera_without_frames_returns_503():
    fake = FakeCamera()
    fake.read_count = 1
    with patch.object(backend, "open_camera", return_value=fake):
        with TestClient(backend.app) as client:
            response = client.get("/video")
    assert response.status_code == 503
    assert response.json()["detail"] == "Camera could not produce a frame"
    assert fake.released


@pytest.mark.parametrize("failure", [False, cv2.error("encoding failed")])
def test_encoding_failure_releases_camera(failure):
    fake = FakeCamera()
    options = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": (False, None)}
    with patch.object(backend, "open_camera", return_value=fake), patch.object(backend.cv2, "imencode", **options):
        with TestClient(backend.app) as client:
            response = client.get("/video")
    assert response.status_code == 503
    assert fake.released


@pytest.mark.parametrize("mirror,expected_calls", [("0", 0), ("1", 1)])
def test_camera_mirror(monkeypatch, mirror, expected_calls):
    monkeypatch.setenv("CAMERA_MIRROR", mirror)
    with patch.object(backend.cv2, "flip", wraps=cv2.flip) as flip:
        backend.encode_frame(FakeCamera())
    assert flip.call_count == expected_calls


@pytest.mark.parametrize("platform,backend_name", [
    ("win32", "CAP_DSHOW"),
    ("darwin", "CAP_AVFOUNDATION"),
    ("linux", None),
])
def test_native_camera_backend(monkeypatch, platform, backend_name):
    monkeypatch.setenv("CAMERA_SOURCE", "0")
    with patch.object(backend.sys, "platform", platform), patch.object(backend.cv2, "VideoCapture") as capture:
        backend.open_camera()
    expected = (0, getattr(cv2, backend_name)) if backend_name else (0,)
    capture.assert_called_once_with(*expected)


def test_windows_network_camera_uses_default_backend(monkeypatch):
    source = "rtsp://192.168.1.105/live"
    monkeypatch.setenv("CAMERA_SOURCE", source)
    with patch.object(backend.sys, "platform", "win32"), patch.object(backend.cv2, "VideoCapture") as capture:
        backend.open_camera()
    capture.assert_called_once_with(source)


def test_closing_stream_releases_camera():
    fake = FakeCamera()

    async def close_after_first_frame():
        with patch.object(backend, "open_camera", return_value=fake):
            response = backend.video()
        frame = await anext(response.body_iterator)
        assert frame.startswith(b"--frame")
        assert not fake.released
        await response.body_iterator.aclose()
        assert fake.released

    asyncio.run(close_after_first_frame())


def test_response_cleanup_releases_camera_before_iteration():
    fake = FakeCamera()
    with patch.object(backend, "open_camera", return_value=fake):
        response = backend.video()
    asyncio.run(response.background())
    assert fake.released

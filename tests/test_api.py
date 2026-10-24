"""API checks with an injected camera; no device, VPN, or TPU required."""
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from safesight.main import create_app


class FakeCamera:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        time.sleep(1 / 30)
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def app_config(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMERA_SOURCE", raising=False)
    config = tmp_path / "cameras.json"
    config.write_text(json.dumps({"cameras": [{"id": "cam-01", "name": "Test", "source": 0, "zones": []}]}))
    return config


@pytest.fixture
def client(tmp_path, app_config):
    fake = FakeCamera()
    app = create_app(app_config, tmp_path / "data", "preview", camera_factory=lambda *_: fake)
    with TestClient(app) as client:
        yield client
    assert fake.released
    assert all(not t.is_alive() for t in app.state.pipeline.workers["cam-01"].threads)


def test_dashboard_and_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="live"' in response.text
    assert 'id="inferencefps"' in response.text
    assert client.get("/camera").status_code == 200
    assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize("path,kind", [("/dashboard/dashboard.css", "text/css"),
    ("/dashboard/dashboard.js", "javascript"), ("/dashboard/stream.js", "javascript"),
    ("/static/styles.css", "text/css"), ("/static/app.js", "javascript")])
def test_assets(client, path, kind):
    response = client.get(path)
    assert response.status_code == 200
    assert kind in response.headers["content-type"]


def test_status_and_snapshots(client):
    worker = client.app.state.pipeline.workers["cam-01"]
    _, packet = worker.preview.get(timeout=2)
    assert packet is not None
    response = client.get("/api/frame/cam-01")
    assert response.status_code == 200
    assert response.content.startswith(bytes([255, 216]))
    status = client.get("/api/status").json()
    assert status["mode"] == "preview"
    assert status["cameras"][0]["inference_status"] == "disabled"
    assert status["cameras"][0]["tpu_response_received"] is False
    assert "source" not in status["cameras"][0]
    assert client.get("/api/detection/cam-01").status_code == 503
    assert client.get("/api/frame/unknown").status_code == 404
    assert client.get("/api/stream/unknown").status_code == 404


def test_events_and_validation(client):
    assert client.get("/api/events").json() == []
    assert client.get("/api/events?start=2026-10-24").status_code == 422
    assert client.get("/api/events?limit=0").status_code == 422


def test_camera_failure_is_visible_without_breaking_api(tmp_path, app_config):
    fake = FakeCamera(opened=False)
    app = create_app(app_config, tmp_path / "data", "preview", camera_factory=lambda *_: fake)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/video").status_code == 503
        status = client.get("/api/status").json()["cameras"][0]
        assert "unavailable" in status["status"]
    assert fake.released


def test_bad_linq_config_does_not_stop_preview(tmp_path, app_config, monkeypatch):
    monkeypatch.delenv("LINQ_LOGIN", raising=False)
    monkeypatch.delenv("LINQ_PROGRAM_URI", raising=False)
    app = create_app(app_config, tmp_path / "data", "linq", camera_factory=lambda *_: FakeCamera())
    with TestClient(app) as client:
        worker = app.state.pipeline.workers["cam-01"]
        _, packet = worker.preview.get(timeout=2)
        assert packet is not None
        status = client.get("/api/status").json()["cameras"][0]
        assert "LINQ_LOGIN" in status["inference_status"]
        assert status["tpu_response_received"] is False

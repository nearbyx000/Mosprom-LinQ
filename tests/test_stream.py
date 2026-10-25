"""Real localhost HTTP streaming with an injected camera, never a TPU."""
import json
import socket
import threading
import time

import httpx
import uvicorn

from safesight.main import create_app
from test_api import FakeCamera


def test_two_mjpeg_clients_share_capture_and_disconnect_cleanly(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMERA_SOURCE", raising=False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"cameras": [{"id": "cam-01", "name": "Test", "source": 0, "zones": []}]}))
    cameras = []

    def factory(*args):
        camera = FakeCamera()
        cameras.append(camera)
        return camera

    app = create_app(config, tmp_path / "data", "preview", camera_factory=factory)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
        thread.start()
        try:
            deadline = time.perf_counter() + 5
            while not server.started and time.perf_counter() < deadline:
                time.sleep(0.01)
            assert server.started
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
                with client.stream("GET", "/video") as first, client.stream("GET", "/api/stream/cam-01") as second:
                    for response in (first, second):
                        assert response.status_code == 200
                        assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
                        data = next(response.iter_bytes())
                        assert b"Content-Length:" in data
                        assert b"X-Frame-Sequence:" in data
                        assert bytes([255, 216]) in data
                assert len(cameras) == 1
                assert client.get("/health").status_code == 200
        finally:
            server.should_exit = True
            thread.join(timeout=8)
        assert not thread.is_alive()
    assert cameras[0].released

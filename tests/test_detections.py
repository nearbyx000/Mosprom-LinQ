import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from safesight.logic import Detection
from safesight.main import create_app
from safesight.runtime import Frame


class Camera:
    def isOpened(self):
        return True

    def read(self):
        time.sleep(1 / 30)
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv('CAMERA_SOURCE', raising=False)
    cfg = tmp_path / 'cameras.json'
    cfg.write_text(json.dumps({'cameras': [{'id': 'cam-01', 'name': 'test', 'source': 0, 'zones': []}]}))
    app = create_app(cfg, tmp_path / 'data', 'preview', camera_factory=lambda *_: Camera())
    with TestClient(app) as client:
        _, packet = app.state.pipeline.workers['cam-01'].preview.get(timeout=2)
        assert packet is not None
        yield client


def test_websocket_fixture_is_explicit_and_does_not_create_events(client):
    with client.websocket_connect('/ws/detections/cam-01?test=1') as ws:
        data = ws.receive_json()
        assert data['source'] == 'test-json'
        assert data['coordinate_format'] == 'xyxy'
        assert data['detections'][0]['label'] == 'TEST person'
        assert data['updated_at']
        assert client.get('/api/frame/cam-01').status_code == 200
    assert client.get('/api/events').json() == []
    assert client.get('/api/status').json()['cameras'][0]['tpu_response_received'] is False


def test_real_result_contract_survives_event_queue_failure(client):
    worker = client.app.state.pipeline.workers['cam-01']
    worker.results.put_nowait = lambda *_a: (_ for _ in ()).throw(RuntimeError('event queue failed'))
    worker.mode = 'linq'  # Inject a result at the real LinQ boundary, no hardware claim.
    frame = Frame(42, time.perf_counter(), time.time(), np.zeros((48, 64, 3), np.uint8), 1)
    with pytest.raises(RuntimeError):
        worker.enqueue_result((frame, [Detection('person', .9, (2, 3, 20, 40))], 'test-task'))
    with client.websocket_connect('/ws/detections/cam-01') as ws:
        message = ws.receive_json()
        assert message['source'] == 'linq'
        assert message['frame_id'] == 42
        assert message['task_id'] == 'test-task'
        assert message['detections'][0]['bbox'] == [2, 3, 20, 40]
    assert client.get('/api/frame/cam-01').status_code == 200


def test_latest_result_not_history_and_empty_detections_clear_boxes(client):
    worker = client.app.state.pipeline.workers['cam-01']
    for sequence in range(1, 4):
        frame = Frame(sequence, time.perf_counter(), time.time(), np.zeros((48, 64, 3), np.uint8), 1)
        worker.enqueue_result((frame, [], None))
    with client.websocket_connect('/ws/detections/cam-01') as ws:
        message = ws.receive_json()
        assert message['frame_id'] == 3
        assert message['detections'] == []


def test_idle_detector_reports_status_without_stopping_camera(client):
    with client.websocket_connect('/ws/detections/cam-01') as ws:
        assert ws.receive_json()['inference_status'] == 'disabled'
        assert client.get('/api/frame/cam-01').status_code == 200


def test_unknown_camera_rejected(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/ws/detections/unknown'):
            pass

import json
import time

import numpy as np
from fastapi.testclient import TestClient

from safesight.main import create_app


def test_requested_evidence_frame_never_returns_another_snapshot(tmp_path,monkeypatch):
    monkeypatch.setenv('CAMERA_SOURCE','synthetic')
    config = tmp_path/'cameras.json'
    config.write_text(json.dumps({'cameras':[{'id':'cam','name':'Test','source':'synthetic','zones':[]}]}))
    with TestClient(create_app(config,tmp_path/'data','preview')) as client:
        worker = client.app.state.pipeline.workers['cam']
        with worker.lock:
            worker.detection_frames = {(1,41):b'old-jpeg',(1,42):b'new-jpeg'}
            worker.detection_jpeg = b'new-jpeg'
            worker.state['last_detection_frame'] = 42
        assert client.get('/api/detection/cam?frame=41&generation=1').content == b'old-jpeg'
        response = client.get('/api/detection/cam?frame=41')
        assert response.headers['X-Frame-Sequence'] == '41'
        assert client.get('/api/detection/cam?frame=40').status_code == 404
        assert client.get('/api/detection/cam?frame=41&generation=2').status_code == 404
        assert client.get('/api/detection/cam').content == b'new-jpeg'

import json
import threading
import time

import numpy as np
from fastapi.testclient import TestClient

from safesight.logic import Detection, Zone, RuleEngine
from safesight.main import create_app
from safesight.pipeline import CameraWorker
from safesight.runtime import Frame
from safesight.storage import EventStore


def test_current_warnings_require_confirmation_and_clear_when_person_leaves():
    zone = Zone('danger', 'Danger', ((0, 0), (1, 0), (1, 1), (0, 1)))
    engine = RuleEngine(confirm_frames=3)
    for index in range(3):
        engine.process('c', [Detection('person', .95, (5, 5, 20, 30))], [zone], 40, 40)
        assert len(engine.active_violations) == (1 if index == 2 else 0)
    _, duplicate_events = engine.process('c', [Detection('person', .95, (5, 5, 20, 30))], [zone], 40, 40)
    assert len(engine.active_violations) == 1
    assert duplicate_events == []
    engine.process('c', [], [zone], 40, 40)
    assert engine.active_violations == []


def test_active_warning_clears_but_saved_backlog_remains(tmp_path):
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    worker = CameraWorker({'id': 'c', 'name': 'test', 'source': 0,
                           'zones': [{'id': 'danger', 'name': 'Danger', 'points': [[0, 0], [1, 0], [1, 1], [0, 1]]}]},
                          EventStore(tmp_path / 'events.sqlite'), evidence, mode='demo')
    thread = threading.Thread(target=worker.events_loop)
    worker.threads.append(thread)
    thread.start()
    try:
        for sequence in range(1, 4):
            packet = Frame(sequence, time.perf_counter(), time.time(), np.zeros((40, 40, 3), np.uint8))
            worker.enqueue_result((packet, [Detection('person', .95, (5, 5, 20, 30))], None))
        deadline = time.monotonic() + 2
        while not worker.store.list_events() and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(worker.status()['active_warnings']) == 1
        assert len(worker.store.list_events()) == 1
        worker.enqueue_result((Frame(4, time.perf_counter(), time.time(), np.zeros((40, 40, 3), np.uint8)), [], None))
        while worker.status()['warnings_frame_id'] != 4 and time.monotonic() < deadline:
            time.sleep(.01)
        assert worker.status()['active_warnings'] == []
        assert len(worker.store.list_events()) == 1
        with worker.lock:
            worker.warning_snapshot['captured_at'] = time.perf_counter() - 4
        assert worker.status()['warnings_fresh'] is False
    finally:
        worker.stop()


def test_warnings_api_does_not_claim_analysis_in_preview(tmp_path, monkeypatch):
    monkeypatch.setenv('CAMERA_SOURCE', 'synthetic')
    cfg = tmp_path / 'cameras.json'
    cfg.write_text(json.dumps({'cameras': [{'id': 'c', 'name': 'test', 'source': 'synthetic', 'zones': []}]}))
    with TestClient(create_app(cfg, tmp_path / 'data', 'preview')) as client:
        response = client.get('/api/warnings')
        assert response.status_code == 200
        assert response.json()['active_count'] == 0
        assert response.json()['cameras'][0]['fresh'] is False
        assert response.json()['cameras'][0]['warnings'] == []

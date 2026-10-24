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


def test_safety_events_keep_exact_frame_evidence_and_reject_stale_results(tmp_path, monkeypatch):
    monkeypatch.setenv('LINQ_SV_ROUTE', 'safesight.jpeg.sv.test')
    evidence = tmp_path/'evidence'
    evidence.mkdir()
    worker = CameraWorker({'id':'c','name':'test','source':0,'zones':[]},
                          EventStore(tmp_path/'events.sqlite'), evidence, mode='linq')
    thread = threading.Thread(target=worker.events_loop)
    worker.threads.append(thread)
    thread.start()
    captured_wall = None
    try:
        for sequence in range(1,4):
            captured_wall = time.time()
            packet = Frame(sequence,time.perf_counter(),captured_wall,np.zeros((80,100,3),np.uint8),1)
            worker.enqueue_result((packet,[Detection('person',.9,(10,0,60,75)),
                                          Detection('no_helmet',.9,(20,5,40,20)),
                                          Detection('fall_detected',.9,(5,40,95,75))],'hardware-boundary-fixture'))
        deadline = time.monotonic()+3
        while len(worker.store.list_events()) < 2 and time.monotonic() < deadline:
            time.sleep(.01)
        events = worker.store.list_events()
        assert {event['violation_type'] for event in events} == {'no_helmet','fall_detected'}
        assert len(events) == 2
        assert len(worker.status()['active_warnings']) == 2
        assert worker.status()['safetyvision_fresh'] is True
        for event in events:
            assert (evidence/event['evidence_file']).read_bytes().startswith(b'\xff\xd8')
        # A replay and an aged result cannot produce another incident or replace
        # the frame used by Warns/evidence.
        worker.enqueue_result((packet,[], 'duplicate'))
        worker.enqueue_result((Frame(4,time.perf_counter()-4,time.time(),packet.image,1),[], 'old'))
        while worker.metrics.snapshot()['counts'].get('event_results_stale',0)<2 and time.monotonic()<deadline:
            time.sleep(.01)
        assert worker.metrics.snapshot()['counts']['event_results_stale'] == 2
        assert worker.status()['warnings_frame_id'] == 3
        assert worker.status()['warnings_updated_at'] == captured_wall
        assert len(worker.store.list_events()) == 2
        worker.enqueue_result((Frame(4,time.perf_counter(),time.time(),packet.image,1,False),
                               [Detection('person',.9,(10,0,60,75))],'primary-only'))
        while worker.status()['warnings_frame_id'] != 4 and time.monotonic()<deadline:
            time.sleep(.01)
        # No SV response is unknown, not evidence that the fall/helmet incident
        # cleared. Each retained warning still points to its own original JPEG.
        warnings = worker.status()['active_warnings']
        assert len(warnings) == 2
        assert all(w['frame_id']==3 and w['evidence_frame_id']==3 for w in warnings)
        assert all((evidence/w['evidence_file']).is_file() for w in warnings)
        with worker.lock:
            worker.last_safety_at = time.perf_counter()-3
        assert worker.status()['safetyvision_fresh'] is False
    finally:
        worker.stop()


def test_evidence_write_failure_keeps_warning_and_retries_fresh_observation(tmp_path):
    evidence = tmp_path/'evidence'
    evidence.mkdir()
    worker = CameraWorker({'id':'c','name':'test','source':0,'zones':[]},
                          EventStore(tmp_path/'events.sqlite'),evidence,mode='demo')
    original = worker.recorder.record
    calls = []
    def first_write_fails(*args):
        calls.append(1)
        if len(calls)==1:
            raise OSError('disk temporarily unavailable')
        return original(*args)
    worker.recorder.record = first_write_fails
    thread = threading.Thread(target=worker.events_loop)
    worker.threads.append(thread)
    thread.start()
    try:
        for sequence in range(1,4):
            worker.enqueue_result((Frame(sequence,time.perf_counter(),time.time(),np.zeros((40,60,3),np.uint8),1),
                                   [Detection('fall_detected',.9,(0,10,50,35))],None))
        deadline = time.monotonic()+3
        while not worker.metrics.snapshot()['counts'].get('evidence_write_errors') and time.monotonic()<deadline:
            time.sleep(.01)
        assert len(worker.status()['active_warnings']) == 1
        assert 'failed' in worker.status()['event_status']
        assert worker.store.list_events() == []
        worker.enqueue_result((Frame(4,time.perf_counter(),time.time(),np.zeros((40,60,3),np.uint8),1),
                               [Detection('fall_detected',.9,(0,10,50,35))],None))
        while not worker.store.list_events() and time.monotonic()<deadline:
            time.sleep(.01)
        assert len(worker.store.list_events()) == 1
        assert worker.store.list_events()[0]['frame_id'] == 4
    finally:
        worker.stop()

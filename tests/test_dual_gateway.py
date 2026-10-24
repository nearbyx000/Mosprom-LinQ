"""Unit tests, no network or physical TPU. Run from repo root after patch."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from safesight.jpeg_gateway import JPEGSettings
from safesight.dual_gateway import DualGatewayClient, SVGatewayClient
from safesight.logic import Detection, RuleEngine
from safesight.runtime import Frame, Metrics


class Clock:
    def __init__(self):
        self.value = 100.0
    def __call__(self):
        return self.value
    def add(self, seconds):
        self.value += seconds


class FakeClient:
    def __init__(self, *, failing_connect=False):
        self.out = []
        self.sent = []
        self.failing_connect = failing_connect
        self.fail_poll = False
    def connect(self):
        if self.failing_connect:
            raise ConnectionError('not available')
    def close(self):
        pass
    def can_submit(self):
        return True
    def submit(self, frame):
        self.sent.append(frame)
        return True
    def poll(self):
        if self.fail_poll:
            raise ConnectionError('disconnected')
        out, self.out = self.out, []
        return out
    def status(self):
        return {'inflight': 0, 'max_request_fps': 2, 'error': None}


def a_frame(n):
    return Frame(n, 100.0, 1000.0, np.zeros((480, 640, 3), dtype=np.uint8), 1)


def test_exact_frame_merging(monkeypatch):
    monkeypatch.setenv('LINQ_SV_ROUTE', 'safesight.jpeg.sv.test')
    p, s, clock = FakeClient(), FakeClient(), Clock()
    dual = DualGatewayClient(JPEGSettings(route='safesight.jpeg.base'), Metrics(), clock=clock,
                             primary=p, secondary=s)
    dual.connect()
    f = a_frame(1)
    assert dual.submit(f)
    p.out.append((f, [Detection('person', 0.9, (1,2,20,50))], 'task-1'))
    assert dual.poll() == []  # Await SV for SAME frame only.
    s.out.append((f, [Detection('no_helmet', 0.8, (1,2,15,18))], 'task-2'))
    rows = dual.poll()
    assert len(rows) == 1
    assert [d.label for d in rows[0][1]] == ['person', 'no_helmet']
    assert rows[0][0].safetyvision_observed is True
    assert dual.status()['sv_joined'] == 1
    assert dual.poll() == []


def test_late_sv_never_mixed_with_next_frame(monkeypatch):
    monkeypatch.setenv('LINQ_SV_ROUTE', 'safesight.jpeg.sv.test')
    p, s, clock = FakeClient(), FakeClient(), Clock()
    dual = DualGatewayClient(JPEGSettings(route='safesight.jpeg.base'), Metrics(), clock=clock,
                             primary=p, secondary=s)
    dual.connect()
    f1, f2 = a_frame(1), a_frame(2)
    dual.submit(f1)
    p.out.append((f1, [Detection('person', .9, (1,2,20,50))], '1'))
    assert dual.poll() == []
    clock.add(.8)
    result = dual.poll()
    assert len(result) == 1 and [d.label for d in result[0][1]] == ['person']
    s.out.append((f1, [Detection('fall_detected', .9, (1,2,20,50))], 'sv-1'))
    dual.submit(f2)
    p.out.append((f2, [Detection('person', .9, (1,2,20,50))], '2'))
    dual.poll()
    clock.add(.8)
    result = dual.poll()
    assert len(result) == 1
    assert [d.label for d in result[0][1]] == ['person']
    assert dual.status()['sv_missed'] == 2


def test_secondary_loss_preserves_base(monkeypatch):
    monkeypatch.setenv('LINQ_SV_ROUTE', 'safesight.jpeg.sv.test')
    p, s, clock = FakeClient(), FakeClient(failing_connect=True), Clock()
    dual = DualGatewayClient(JPEGSettings(route='safesight.jpeg.base'), Metrics(), clock=clock,
                             primary=p, secondary=s)
    dual.connect()
    assert not dual.status()['sv_enabled']
    f = a_frame(1)
    dual.submit(f)
    assert p.sent == [f] and s.sent == []
    p.out.append((f, [], 'task'))
    rows = dual.poll()
    assert rows[0][0].sequence == f.sequence and rows[0][1:] == ([], 'task')
    assert rows[0][0].safetyvision_observed is False


def test_sv_json_contract():
    client = SVGatewayClient.__new__(SVGatewayClient)
    clock = Clock()
    f = a_frame(1)
    client.clock = clock
    client.metrics = Metrics()
    client.last_accepted = 0
    client.completed = []
    client.pending = {'t': {'frame': f, 'sent': 100.0, 'timed_out': False}}
    client.settings = SimpleNamespace(max_result_age=3)
    client.last_error = None
    doc = {'ok': True, 'frame_id': 't', 'source_width': 640, 'source_height': 480,
           'detections': [{'class_name':'fall_detected', 'confidence':.8, 'bbox':[30,40,100,150]},
                          {'class_name':'no_helmet', 'confidence':.91, 'bbox':[33,45,55,65]}]}
    client._on_response(None, None, SimpleNamespace(correlation_id='t'), json.dumps(doc).encode())
    assert [d.label for d in client.completed[0][1]] == ['fall_detected', 'no_helmet']
    assert client.last_error is None


def test_invalid_sv_label_rejected():
    client = SVGatewayClient.__new__(SVGatewayClient)
    client.clock = Clock(); client.metrics=Metrics(); client.last_accepted=0
    client.completed=[]; client.pending={'t':{'frame':a_frame(1),'sent':100.,'timed_out':False}}
    client.settings=SimpleNamespace(max_result_age=3); client.last_error=None
    doc={'ok':True,'frame_id':'t','source_width':640,'source_height':480,
         'detections':[{'class_name':'person','confidence':.8,'bbox':[1,2,3,4]}]}
    client._on_response(None,None,SimpleNamespace(correlation_id='t'),json.dumps(doc).encode())
    assert not client.completed
    assert client.metrics.snapshot()['counts']['responses_invalid'] == 1


def test_rule_engine_fall_requires_three_hits():
    e=RuleEngine(confirm_frames=3)
    detections=lambda:[Detection('fall_detected', .88, (30,40,150,180))]
    for _ in range(2):
        _, events=e.process('cam-01', detections(), [], 640, 480)
        assert events == []
    _, events=e.process('cam-01', detections(), [], 640, 480)
    assert [x.violation_type for x in events] == ['fall_detected']
    _, events=e.process('cam-01', detections(), [], 640, 480)
    assert events == []


def test_sv_pacing_never_stalls_primary(monkeypatch):
    monkeypatch.setenv('LINQ_SV_ROUTE', 'safesight.jpeg.sv.test')
    p, s, clock = FakeClient(), FakeClient(), Clock()
    dual = DualGatewayClient(JPEGSettings(route='safesight.jpeg.base'), Metrics(), clock=clock,
                             primary=p, secondary=s)
    dual.connect()
    s.can_submit = lambda: False
    assert dual.can_submit(), 'Primary keeps its own rate; absent SV is explicitly marked unknown'
    s.status = lambda: {'inflight': 1, 'error': 'TPU response timeout'}
    assert dual.can_submit(), 'Primary should remain available when SV is unhealthy'


def test_join_rechecks_age_after_wait(monkeypatch):
    monkeypatch.setenv('LINQ_SV_ROUTE', 'safesight.jpeg.sv.test')
    monkeypatch.setenv('LINQ_SV_JOIN_GRACE', '.4')
    p, s, clock = FakeClient(), FakeClient(), Clock()
    dual = DualGatewayClient(JPEGSettings(route='safesight.jpeg.base', max_result_age=.5), Metrics(),
                             clock=clock, primary=p, secondary=s)
    # A legal grace period still cannot extend the total freshness budget.
    dual.join_grace = .4
    dual.connect()
    f = a_frame(1)
    dual.submit(f)
    clock.add(.3)
    p.out.append((f, [], 'p'))
    assert dual.poll() == []
    clock.add(.3)
    s.out.append((f, [Detection('fall_detected', .9, (1,2,20,50))], 's'))
    assert dual.poll() == []
    assert dual.metrics.snapshot()['counts']['dual_results_stale'] == 1


def test_newer_primary_cannot_overtake_pending_safety_join(monkeypatch):
    monkeypatch.setenv('LINQ_SV_ROUTE', 'safesight.jpeg.sv.test')
    p, s, clock = FakeClient(), FakeClient(), Clock()
    dual = DualGatewayClient(JPEGSettings(route='safesight.jpeg.base'), Metrics(), clock=clock,
                             primary=p, secondary=s)
    dual.connect()
    f1, f2 = a_frame(1), a_frame(2)
    dual.submit(f1)
    s.can_submit = lambda: False
    dual.submit(f2)
    p.out.extend([(f1, [], 'p1'), (f2, [], 'p2')])
    assert dual.poll() == []
    s.out.append((f1, [Detection('fall_detected', .9, (1,2,20,50))], 's'))
    rows = dual.poll()
    assert [r[0].sequence for r in rows] == [1,2]
    assert [r[0].safetyvision_observed for r in rows] == [True,False]
    # Even a late duplicate primary cannot be emitted a second time.
    p.out.append((f1, [], 'p1'))
    assert dual.poll() == []

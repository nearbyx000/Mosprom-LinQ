import base64
import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from safesight.main import create_app


def application(tmp_path,monkeypatch):
    monkeypatch.setenv('CAMERA_SOURCE','synthetic')
    config = tmp_path/'camera.json'
    config.write_text(json.dumps({'cameras':[{'id':'c','name':'Test','source':'synthetic','zones':[]}]}))
    return create_app(config,tmp_path/'data','demo')


def test_review_history_keeps_original_detection_immutable(tmp_path,monkeypatch):
    with TestClient(application(tmp_path,monkeypatch)) as client:
        store = client.app.state.pipeline.store
        event_id = store.insert('c','z','no_helmet',4,.86,'original.jpg',source='demo',evidence_sha256='same-hash')
        review = {'decision':'false_positive','note':'Проверено по исходному кадру'}
        assert client.patch(f'/api/events/{event_id}/review',json=review).status_code == 200
        client.patch(f'/api/events/{event_id}/review',json=review)  # Safe retry must not duplicate audit entry.
        assert len(client.get(f'/api/events/{event_id}/reviews').json()) == 1
        client.patch(f'/api/events/{event_id}/review',json={'decision':'needs_review','note':'Второй просмотр'})
        assert len(client.get(f'/api/events/{event_id}/reviews').json()) == 2
        row = next(row for row in client.get('/api/events').json() if row['id']==event_id)
        assert row['review_status'] == 'needs_review'
        assert row['confidence'] == .86 and row['evidence_file'] == 'original.jpg'
        assert row['evidence_sha256'] == 'same-hash'
        assert client.patch('/api/events/999999/review',json=review).status_code == 404
        assert client.patch(f'/api/events/{event_id}/review',json={'decision':'delete'}).status_code == 422
        assert client.patch(f'/api/events/{event_id}/review',json={'decision':'confirmed','note':'x'*1001}).status_code == 422


def test_optional_auth_protects_data_and_websocket_but_not_liveness(tmp_path,monkeypatch):
    monkeypatch.setenv('SAFESIGHT_AUTH_USER','test-operator')
    monkeypatch.setenv('SAFESIGHT_AUTH_PASSWORD','test-only-not-a-production-password')
    header = 'Basic '+base64.b64encode(b'test-operator:test-only-not-a-production-password').decode()
    with TestClient(application(tmp_path,monkeypatch)) as client:
        assert client.get('/health').status_code == 200
        for path in ('/','/api/status','/api/events','/api/events/export.csv','/evidence/any.jpg'):
            assert client.get(path).status_code == 401
        assert client.get('/api/status',headers={'Authorization':header}).status_code == 200
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/ws/detections/c'):
                pass
        with client.websocket_connect('/ws/detections/c?test=1',headers={'Authorization':header}) as ws:
            assert ws.receive_json()['source'] == 'test-json'


def test_foreign_origin_cannot_review_or_open_detection_socket(tmp_path,monkeypatch):
    with TestClient(application(tmp_path,monkeypatch)) as client:
        assert client.patch('/api/events/1/review',json={'decision':'confirmed'},
                            headers={'Origin':'https://unrelated.invalid'}).status_code == 403
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/ws/detections/c',headers={'Origin':'https://unrelated.invalid'}):
                pass

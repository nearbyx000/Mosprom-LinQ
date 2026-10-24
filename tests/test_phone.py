import asyncio
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app as backend
from phone_stream import PhoneFeed, MAX_FRAME_BYTES, sanitize_jpeg


def jpeg():
    return cv2.imencode('.jpg', np.zeros((24, 32, 3), np.uint8))[1].tobytes()


def test_phone_frame_is_received_and_cleared_on_disconnect():
    with TestClient(backend.app) as client:
        with client.websocket_connect('/phone/publish', headers={'origin': 'http://testserver'}) as ws:
            assert ws.receive_json() == {'ready': True}
            ws.send_bytes(jpeg())
            assert ws.receive_json()['received'] == 1
            assert backend.app.state.phone_feed.jpeg.startswith(b'\xff\xd8')
        # A viewer cannot keep receiving a stale frame after publisher exit.
        assert client.get('/phone/video').status_code == 503


def test_second_publisher_rejected():
    with TestClient(backend.app) as client:
        with client.websocket_connect('/phone/publish', headers={'origin': 'http://testserver'}) as first:
            first.receive_json()
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect('/phone/publish', headers={'origin': 'http://testserver'}) as second:
                    second.receive_json()


def test_cross_origin_publisher_rejected():
    with TestClient(backend.app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/phone/publish', headers={'origin': 'https://other.example'}):
                pass


@pytest.mark.parametrize('payload', [b'not a jpeg', b'x' * (MAX_FRAME_BYTES + 1)], ids=['invalid-jpeg', 'oversize'])
def test_invalid_frame_disconnects_publisher(payload):
    with TestClient(backend.app) as client:
        with client.websocket_connect('/phone/publish', headers={'origin': 'http://testserver'}) as ws:
            ws.receive_json()
            ws.send_bytes(payload)
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()


def test_phone_viewer_receives_latest_jpeg_and_stream_ends():
    async def scenario():
        feed = PhoneFeed()
        owner = await feed.claim()
        await feed.publish(owner, sanitize_jpeg(jpeg()))
        await feed.publish(owner, sanitize_jpeg(jpeg()))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(phone_feed=feed)))
        response = await backend.phone_video(request)
        part = await anext(response.body_iterator)
        assert b'Content-Type: image/jpeg' in part
        assert b'\xff\xd8' in part
        assert feed.sequence == 2
        await feed.release(owner)
        assert await anext(response.body_iterator) == b'--frame--\r\n'
        await response.body_iterator.aclose()
    asyncio.run(scenario())


def test_phone_without_frames_times_out():
    async def scenario():
        feed = PhoneFeed()
        owner = await feed.claim()
        assert await feed.next_frame(timeout=.01) is None
        await feed.release(owner)
    asyncio.run(scenario())

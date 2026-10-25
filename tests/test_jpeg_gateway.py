"""Protocol contract tests; mock replies are NOT real TPU hardware measurements."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from safesight.jpeg_gateway import JPEGGatewayClient, JPEGSettings
from safesight.runtime import Frame, Metrics


class Clock:
    def __init__(self):
        self.now = 100.0
    def __call__(self):
        return self.now


class Channel:
    def __init__(self):
        self.sent = []
    def basic_publish(self, **kwargs):
        self.sent.append(kwargs)


def make_client():
    clock = Clock()
    metrics = Metrics()
    settings = JPEGSettings(route="safesight.jpeg.test")
    pika = SimpleNamespace(BasicProperties=lambda **kw: SimpleNamespace(**kw))
    client = JPEGGatewayClient(settings, metrics, clock=clock, pika_module=pika)
    client.channel = Channel()
    client.reply_queue = "ephemeral-response"
    client.connection = SimpleNamespace(process_data_events=lambda **kw: None)
    return client, clock, metrics


def frame(n=1):
    return Frame(n, 100.0, 1.0, np.zeros((480, 640, 3), dtype=np.uint8))


def reply_for(client, task_id, box=None):
    return json.dumps({"ok": True, "frame_id": task_id, "source_width": 640,
                       "source_height": 480, "detections": [
                           {"class_name":"person", "confidence":0.85, "bbox":box or [100, 50, 200, 250]}],
                       "stats":{"gateway_total_ms": 234.0, "tpu_worker_ms": 8.0}}).encode()


def test_jpeg_wire_and_real_tpu_provenance_contract():
    client, clock, metrics = make_client()
    assert client.submit(frame())
    sent = client.channel.sent[0]
    assert sent["exchange"] == "tpu.direct"
    assert sent["routing_key"] == "safesight.jpeg.test"
    assert sent["body"].startswith(b"\xff\xd8")
    assert len(sent["body"]) < 50000
    assert sent["properties"].content_type == "image/jpeg"
    assert sent["properties"].reply_to == "ephemeral-response"
    assert not client.can_submit()
    clock.now += .48
    client._on_response(None, None, SimpleNamespace(correlation_id=sent["properties"].correlation_id), reply_for(client, sent["properties"].correlation_id))
    items = client.poll()
    assert len(items) == 1 and items[0][1][0].label == "person"
    assert items[0][1][0].box == (100, 50, 200, 250)
    assert metrics.snapshot()["counts"]["detections"] == 1


def test_unroutable_route_fails_cleanly():
    client, _, metrics = make_client()
    client.submit(frame())
    sent = client.channel.sent[0]
    client._on_unroutable(None, None, sent["properties"], b"")
    assert client.status()["inflight"] == 0
    assert "unavailable" in client.status()["error"]


def test_rejects_mismatched_frame_id_and_invalid_box():
    client, _, metrics = make_client()
    client.submit(frame())
    props = client.channel.sent[0]["properties"]
    client._on_response(None, None, SimpleNamespace(correlation_id=props.correlation_id), reply_for(client, "wrong"))
    assert not client.poll()
    assert metrics.snapshot()["counts"]["responses_invalid"] == 1
    client, _, metrics = make_client()
    client.submit(frame())
    props = client.channel.sent[0]["properties"]
    client._on_response(None, None, SimpleNamespace(correlation_id=props.correlation_id), reply_for(client, props.correlation_id, [100, 50, 10000, 250]))
    assert not client.poll()
    assert metrics.snapshot()["counts"]["responses_invalid"] == 1


def test_stale_result_is_not_reused_for_events():
    client, clock, metrics = make_client()
    client.submit(frame())
    props = client.channel.sent[0]["properties"]
    clock.now += 4
    client._on_response(None, None, SimpleNamespace(correlation_id=props.correlation_id), reply_for(client, props.correlation_id))
    assert client.poll() == []
    assert metrics.snapshot()["counts"]["responses_stale"] == 1


def test_route_required():
    with pytest.raises(ValueError, match="LINQ_JPEG_ROUTE"):
        JPEGSettings().validate()

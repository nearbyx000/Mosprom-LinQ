"""Wire-contract tests using fakes. These are NOT hardware measurements."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from safesight.rabbitmq import LinQClient, Settings
from safesight.runtime import Frame, Metrics
from safesight.yolo import preprocess, postprocess


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Codec:
    MESSAGE_ID_ATTR_NAME = "task-id"
    RESPONSE_ROUTING_KEYS_ATTR_NAME = "responses"
    TARGET_URI_ATTR_NAME = "target"

    def PyTask(self, tensors):
        self.tensors = tensors
        return tensors

    def serialize(self, task):
        return b"test-task"

    def deserialize(self, body):
        return {"output0": self.output}, {}, {"status_code": 0, "worker_pid": 42}

    def __init__(self):
        self.output = np.zeros((1, 84, 8400), dtype=np.float32)
        self.output[0, :5, 0] = [320, 320, 100, 200, 0.9]


class Channel:
    def __init__(self):
        self.sent = []

    def basic_publish(self, **kwargs):
        self.sent.append(kwargs)


@pytest.fixture
def rpc():
    clock = Clock()
    codec = Codec()
    metrics = Metrics()
    settings = Settings(program_uri="file:///auto/cache/test/yolov8n.tpu?raw=False")
    client = LinQClient(settings, metrics, clock, SimpleNamespace(BasicProperties=lambda **kw: SimpleNamespace(**kw)), codec)
    client.channel = Channel()
    client.reply_queue = "private-response"
    client.connection = SimpleNamespace(process_data_events=lambda **kw: None)
    return client, clock, codec, metrics


def packet(sequence=1, captured_at=100.0):
    return Frame(sequence, captured_at, 1.0, np.zeros((480, 640, 3), dtype=np.uint8))


def respond(client, task_id):
    client._on_response(None, None, SimpleNamespace(headers={"task-id": task_id}), b"reply")


def test_exact_notebook_headers_and_yolo_tensor(rpc):
    client, clock, codec, metrics = rpc
    assert client.submit(packet())
    sent = client.channel.sent[0]
    assert sent["exchange"] == "tpu.direct"
    assert sent["routing_key"] == "input"
    headers = sent["properties"].headers
    assert headers["target"] == client.settings.program_uri
    assert json.loads(headers["responses"]) == [client.reply_queue]
    assert headers["task-id"] == sent["properties"].correlation_id
    assert codec.tensors["images"].shape == (1, 3, 640, 640)
    assert codec.tensors["images"].dtype == np.float32
    assert not client.submit(packet(2))
    clock.now += 0.56
    respond(client, headers["task-id"])
    results = client.poll()
    assert len(results) == 1
    assert results[0][1][0].label == "person"
    assert client.can_submit()
    assert metrics.snapshot()["counts"]["responses_received"] == 1


def test_unrelated_and_duplicate_responses_are_ignored(rpc):
    client, clock, codec, metrics = rpc
    client.submit(packet())
    task_id = next(iter(client.pending))
    respond(client, "other-task")
    assert len(client.pending) == 1
    respond(client, task_id)
    respond(client, task_id)
    assert len(client.poll()) == 1
    assert metrics.snapshot()["counts"]["responses_unmatched"] == 2


def test_timeout_pauses_submissions_until_late_reply(rpc):
    client, clock, codec, metrics = rpc
    client.submit(packet())
    task_id = next(iter(client.pending))
    clock.now += client.settings.timeout + 1
    assert client.poll() == []
    assert client.status()["paused"]
    assert not client.submit(packet(2, clock.now))
    assert len(client.channel.sent) == 1
    respond(client, task_id)
    assert client.poll() == []
    assert client.can_submit()
    assert metrics.snapshot()["counts"]["responses_stale"] == 1


def test_out_of_order_results_do_not_rewind_events(rpc):
    client, clock, codec, metrics = rpc
    client.settings = replace(client.settings, max_inflight=2)
    client.submit(packet(1))
    first = next(iter(client.pending))
    clock.now += 0.6
    client.submit(packet(2, clock.now))
    second = list(client.pending)[1]
    respond(client, second)
    respond(client, first)
    assert [result[0].sequence for result in client.poll()] == [2]


def test_wrong_model_output_is_not_accepted(rpc):
    client, clock, codec, metrics = rpc
    codec.output = np.zeros((1, 1000), dtype=np.float32)
    client.submit(packet())
    respond(client, next(iter(client.pending)))
    assert client.poll() == []
    assert metrics.snapshot()["counts"]["responses_invalid"] == 1


def test_letterbox_rgb_and_decode_coordinates():
    image = np.full((480, 640, 3), [10, 20, 30], dtype=np.uint8)
    tensor, transform = preprocess(image)
    assert np.allclose(tensor[0, :, 80, 0], np.array([30, 20, 10]) / 255)
    assert np.allclose(tensor[0, :, 0, 0], 114 / 255)
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    output[0, :5, 0] = [320, 320, 100, 200, 0.9]
    output[0, :5, 1] = [321, 321, 100, 200, 0.8]
    detections = postprocess(output, transform)
    assert len(detections) == 1
    assert detections[0].box == (270, 140, 370, 340)


def test_shared_server_budget_validation():
    with pytest.raises(ValueError, match="LINQ_MAX_INFLIGHT"):
        Settings(program_uri="file:///auto/cache/test/yolov8n.tpu?raw=False", max_inflight=30).validate()


def test_private_response_queue_matches_notebook(rpc):
    client, clock, codec, metrics = rpc
    calls = []
    channel = SimpleNamespace(
        queue_declare=lambda **kwargs: (calls.append(("declare", kwargs)) or SimpleNamespace(method=SimpleNamespace(queue="server-named"))),
        queue_bind=lambda **kwargs: calls.append(("bind", kwargs)),
        add_on_return_callback=lambda callback: None,
        basic_consume=lambda **kwargs: calls.append(("consume", kwargs)),
    )
    client.pika = SimpleNamespace(BlockingConnection=lambda parameters: SimpleNamespace(channel=lambda: channel),
                                 ConnectionParameters=lambda **kwargs: kwargs, PlainCredentials=lambda *args: args)
    client.connect()
    assert calls[0] == ("declare", {"queue": "", "exclusive": True, "auto_delete": True})
    assert calls[1] == ("bind", {"exchange": "tpu.direct", "queue": "server-named", "routing_key": "server-named"})
    assert calls[2][1]["auto_ack"] is True

"""JPEG over RabbitMQ to a separately running LinQ cloud gateway.

The gateway performs the ONLY neural inference on the remote LinQ TPU. This
adapter never imports tpu_pycloud_format or runs a local neural network.
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass

import cv2

from .logic import Detection
from .rabbitmq import LinQClient


@dataclass(frozen=True)
class JPEGSettings:
    host: str = "10.0.114.30"
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    route: str = ""
    jpeg_quality: int = 75
    max_inflight: int = 1
    max_request_fps: float = 2.0
    timeout: float = 10.0
    max_result_age: float = 3.0

    @classmethod
    def from_env(cls):
        result = cls(
            host=os.getenv("RABBITMQ_HOST", cls.host),
            port=int(os.getenv("RABBITMQ_PORT", cls.port)),
            username=os.getenv("RABBITMQ_USERNAME", cls.username),
            password=os.getenv("RABBITMQ_PASSWORD", cls.password),
            virtual_host=os.getenv("RABBITMQ_VHOST", cls.virtual_host),
            route=os.getenv("LINQ_JPEG_ROUTE", "").strip(),
            jpeg_quality=int(os.getenv("LINQ_JPEG_QUALITY", "75")),
            max_inflight=1,  # The gateway handles jobs serially; avoid stale queues.
            max_request_fps=float(os.getenv("LINQ_MAX_REQUEST_FPS", "2")),
            timeout=float(os.getenv("LINQ_TIMEOUT", "10")),
            max_result_age=float(os.getenv("LINQ_MAX_RESULT_AGE", "3")),
        )
        result.validate()
        return result

    def validate(self):
        if not self.route.startswith("safesight.jpeg.") or len(self.route) > 180 or not self.route.isascii() or any(c.isspace() for c in self.route):
            raise ValueError("LINQ_JPEG_ROUTE must equal the 'GATEWAY READY' route from the cloud process")
        if not 30 <= self.jpeg_quality <= 95:
            raise ValueError("LINQ_JPEG_QUALITY must be in [30,95]")
        if self.max_inflight != 1 or not 0 < self.max_request_fps <= 5:
            raise ValueError("JPEG gateway uses one in-flight request and at most 5 requests/s")
        if not 1<=self.port<=65535 or not all(math.isfinite(v) and v>0 for v in (self.timeout,self.max_result_age)):
            raise ValueError("Timeout and max result age must be positive")


class JPEGGatewayClient(LinQClient):
    """Implements the same connect/submit/poll/status/close interface as LinQClient.

    All Pika operations remain on the dedicated CameraWorker inference thread.
    MJPEG preview and WebSocket rendering never await the cloud result.
    """
    name = "LinQ TPU / cloud JPEG Gateway"

    def __init__(self, settings: JPEGSettings, metrics, clock=time.perf_counter, pika_module=None):
        # Supply a dummy codec to skip importing cloud-only tpu_pycloud_format.
        super().__init__(settings, metrics, clock, pika_module=pika_module, codec=object())

    @staticmethod
    def _task_id(properties):
        return getattr(properties, "correlation_id", None)

    def submit(self, frame):
        if not self.can_submit():
            return False
        start = self.clock()
        ok, jpg = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        encoded = jpg.tobytes()
        self.metrics.observe("jpeg_gateway_encode", (self.clock() - start) * 1000)
        task_id = uuid.uuid4().hex
        sent = self.clock()
        self.pending[task_id] = {"frame": frame, "sent": sent, "timed_out": False}
        self.last_sent = sent
        self.channel.basic_publish(
            exchange="tpu.direct", routing_key=self.settings.route, body=encoded,
            mandatory=True,
            properties=self.pika.BasicProperties(
                content_type="image/jpeg", correlation_id=task_id,
                reply_to=self.reply_queue, delivery_mode=1,
                expiration=str(int(self.settings.timeout * 1000)),
            ),
        )
        self.metrics.tick("requests_sent")
        self.metrics.tick("jpeg_gateway_bytes_sent", len(encoded))
        return True

    def _on_unroutable(self, channel, method, properties, body):
        if self.pending.pop(self._task_id(properties), None) is not None:
            self.last_error = "JPEG route unavailable: check remote Gateway and LINQ_JPEG_ROUTE"
            self.metrics.tick("requests_unroutable")

    def _on_response(self, channel, method, properties, body):
        task_id = self._task_id(properties)
        request = self.pending.pop(task_id, None)
        if request is None:
            self.metrics.tick("responses_unmatched")
            return
        received = self.clock()
        self.metrics.tick("responses_received")
        self.metrics.observe("rpc_round_trip", (received - request["sent"]) * 1000)
        frame = request["frame"]
        if request["timed_out"] or received - frame.captured_at > self.settings.max_result_age or frame.sequence <= self.last_accepted:
            self.metrics.tick("responses_stale")
            return
        try:
            if len(body) > 262144:
                raise ValueError("oversized JSON reply")
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("gateway result must be an object")
            height, width = frame.image.shape[:2]
            if (payload.get("ok") is not True or payload.get("frame_id") != task_id
                    or payload.get("source_width") != width or payload.get("source_height") != height):
                raise ValueError("mismatched gateway response")
            raw = payload.get("detections")
            if not isinstance(raw, list) or len(raw) > 200:
                raise ValueError("invalid detections list")
            detections = []
            for item in raw:
                if not isinstance(item, dict) or item.get("class_name") != "person":
                    raise ValueError("unexpected class")
                score = float(item["confidence"])
                box = item["bbox"]
                if (not math.isfinite(score) or not 0 <= score <= 1
                        or not isinstance(box, list) or len(box) != 4
                        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
                    raise ValueError("invalid bounding box or confidence")
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError("out-of-frame bounding box")
                detections.append(Detection("person", score, (x1, y1, x2, y2)))
            stats = payload.get("stats", {})
            if isinstance(stats, dict):
                for name, key in (("jpeg_gateway_cloud", "gateway_total_ms"), ("jpeg_gateway_tpu", "tpu_worker_ms")):
                    val = stats.get(key)
                    if isinstance(val, (int, float)) and math.isfinite(val) and val >= 0:
                        self.metrics.observe(name, val)
            self.metrics.observe("capture_to_detection", (received - frame.captured_at) * 1000)
            self.last_accepted = frame.sequence
            self.last_error = None
            self.completed.append((frame, detections, task_id))
            self.metrics.tick("detections")
        except (ValueError, KeyError, TypeError, OverflowError):
            self.last_error = "Invalid JPEG Gateway response; verify remote gateway and program"
            self.metrics.tick("responses_invalid")

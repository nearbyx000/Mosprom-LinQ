"""LinQ cloud wire protocol from the supplied working notebook.

All Pika operations stay on one inference thread. Publishing is asynchronous
with respect to responses; poll() pumps callbacks without blocking camera work.
No TPU SDK implementation or local neural model is substituted here.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit

from .yolo import preprocess, postprocess


@dataclass(frozen=True)
class Settings:
    host: str = "10.0.114.30"
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    program_uri: str = ""
    input_name: str = "images"
    output_name: str = "output0"
    max_inflight: int = 1
    max_request_fps: float = 2.0
    timeout: float = 10.0
    max_result_age: float = 3.0

    @classmethod
    def from_env(cls):
        login = os.getenv("LINQ_LOGIN", "").strip()
        uri = os.getenv("LINQ_PROGRAM_URI", "").strip()
        if not uri and login:
            uri = f"file:///auto/cache/{login}/yolov8n.tpu?raw=False"
        settings = cls(
            host=os.getenv("RABBITMQ_HOST", cls.host),
            port=int(os.getenv("RABBITMQ_PORT", str(cls.port))),
            username=os.getenv("RABBITMQ_USERNAME", cls.username),
            password=os.getenv("RABBITMQ_PASSWORD", cls.password),
            virtual_host=os.getenv("RABBITMQ_VHOST", cls.virtual_host),
            program_uri=uri,
            input_name=os.getenv("LINQ_INPUT_NAME", cls.input_name),
            output_name=os.getenv("LINQ_OUTPUT_NAME", cls.output_name),
            max_inflight=int(os.getenv("LINQ_MAX_INFLIGHT", "1")),
            max_request_fps=float(os.getenv("LINQ_MAX_REQUEST_FPS", "2")),
            timeout=float(os.getenv("LINQ_TIMEOUT", "10")),
            max_result_age=float(os.getenv("LINQ_MAX_RESULT_AGE", "3")),
        )
        settings.validate()
        return settings

    def validate(self):
        uri = urlsplit(self.program_uri)
        if uri.scheme != "file" or uri.netloc or not uri.path.startswith("/auto/cache/") or not uri.path.endswith(".tpu") or uri.query != "raw=False":
            raise ValueError("Set LINQ_LOGIN or LINQ_PROGRAM_URI=file:///auto/cache/<LOGIN>/yolov8n.tpu?raw=False")
        if not 1 <= self.max_inflight <= 2:
            raise ValueError("LINQ_MAX_INFLIGHT must be 1 or 2 for the shared broker")
        if (not 1<=self.port<=65535 or not 0 < self.max_request_fps <= 5 or
                not all(math.isfinite(v) and v>0 for v in (self.timeout,self.max_result_age))):
            raise ValueError("Request FPS must be in (0,5]; timeout and max result age must be positive")


class LinQClient:
    name = "LinQ TPU / RabbitMQ"

    def __init__(self, settings: Settings, metrics, clock=time.perf_counter, pika_module=None, codec=None):
        settings.validate()
        self.settings, self.metrics, self.clock = settings, metrics, clock
        try:
            self.pika = pika_module or importlib.import_module("pika")
            self.codec = codec or importlib.import_module("tpu_pycloud_format")
        except ImportError as exc:
            raise RuntimeError("Install requirements-linq.txt and the organizer's tpu_pycloud_format package in this Python environment") from exc
        self.connection = None
        self.channel = None
        self.reply_queue = None
        self.pending = {}
        self.completed = []
        self.last_sent = float("-inf")
        self.last_accepted = 0
        self.last_error = None

    def connect(self):
        s = self.settings
        self.connection = self.pika.BlockingConnection(self.pika.ConnectionParameters(
            host=s.host, port=s.port, virtual_host=s.virtual_host,
            credentials=self.pika.PlainCredentials(s.username, s.password),
            heartbeat=30, socket_timeout=3, stack_timeout=5,
            blocked_connection_timeout=3, connection_attempts=1,
        ))
        try:
            self.channel = self.connection.channel()
            result = self.channel.queue_declare(queue="", exclusive=True, auto_delete=True)
            self.reply_queue = result.method.queue
            # Same exchange/response-routing header convention as the notebook.
            self.channel.queue_bind(exchange="tpu.direct", queue=self.reply_queue, routing_key=self.reply_queue)
            self.channel.add_on_return_callback(self._on_unroutable)
            self.channel.basic_consume(queue=self.reply_queue, on_message_callback=self._on_response, auto_ack=True)
        except Exception:
            self.close()
            raise

    def can_submit(self):
        return (len(self.pending) < self.settings.max_inflight
                and not any(p["timed_out"] for p in self.pending.values())
                and self.clock() - self.last_sent >= 1 / self.settings.max_request_fps)

    def submit(self, frame):
        if not self.can_submit():
            return False
        started = self.clock()
        tensor, transform = preprocess(frame.image)
        pre_ms = (self.clock() - started) * 1000
        task_id = uuid.uuid4().hex
        started = self.clock()
        body = self.codec.serialize(self.codec.PyTask({self.settings.input_name: tensor}))
        serialization_ms = (self.clock() - started) * 1000
        sent = self.clock()
        self.pending[task_id] = {"frame": frame, "transform": transform, "sent": sent,
                                 "timed_out": False, "preprocessing_ms": pre_ms,
                                 "serialization_ms": serialization_ms}
        # Record before publishing: a lost connection makes execution ambiguous.
        self.last_sent = sent
        self.channel.basic_publish(
            exchange="tpu.direct", routing_key="input", body=body, mandatory=True,
            properties=self.pika.BasicProperties(
                correlation_id=task_id, reply_to=self.reply_queue, delivery_mode=1,
                expiration=str(int(self.settings.timeout * 1000)),
                headers={self.codec.TARGET_URI_ATTR_NAME: self.settings.program_uri,
                         self.codec.RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([self.reply_queue]),
                         self.codec.MESSAGE_ID_ATTR_NAME: task_id},
            ),
        )
        self.metrics.tick("requests_sent")
        return True

    def _task_id(self, properties):
        task_id = (properties.headers or {}).get(self.codec.MESSAGE_ID_ATTR_NAME)
        if isinstance(task_id, bytes):
            task_id = task_id.decode("utf-8")
        return task_id

    def _on_unroutable(self, channel, method, properties, body):
        if self.pending.pop(self._task_id(properties), None) is not None:
            self.metrics.tick("requests_unroutable")
            self.last_error = "Request unroutable: check tpu.direct/input on the cloud service"

    def _on_response(self, channel, method, properties, body):
        task_id = self._task_id(properties)
        request = self.pending.pop(task_id, None)
        if request is None:
            self.metrics.tick("responses_unmatched")
            return
        received = self.clock()
        self.metrics.tick("responses_received")
        rtt_ms = (received - request["sent"]) * 1000
        self.metrics.observe("rpc_round_trip", rtt_ms)
        frame = request["frame"]
        if request["timed_out"] or received - frame.captured_at > self.settings.max_result_age or frame.sequence <= self.last_accepted:
            self.metrics.tick("responses_stale")
            if not any(p["timed_out"] for p in self.pending.values()):
                self.last_error = None
            return
        try:
            started = self.clock()
            data, metadata, worker = self.codec.deserialize(body)
            deserialize_ms = (self.clock() - started) * 1000
            if not data or self.settings.output_name not in data:
                raise ValueError("Cloud worker returned no configured output tensor")
            started = self.clock()
            detections = postprocess(data[self.settings.output_name], request["transform"])
            post_ms = (self.clock() - started) * 1000
            timings = {"preprocessing": request["preprocessing_ms"], "serialization": request["serialization_ms"],
                       "deserialization": deserialize_ms, "postprocessing": post_ms}
            for key, value in timings.items():
                self.metrics.observe(key, value)
            self.metrics.observe("capture_to_detection", (self.clock() - frame.captured_at) * 1000)
            self.last_accepted = frame.sequence
            self.last_error = None
            self.completed.append((frame, detections, task_id))
            self.metrics.tick("detections")
        except Exception as exc:
            # Never expose worker error strings, which may include private paths.
            self.last_error = f"Invalid LinQ response ({type(exc).__name__}); check program and output tensor name"
            self.metrics.tick("responses_invalid")

    def poll(self):
        self.connection.process_data_events(time_limit=0.01)
        for request in self.pending.values():
            if not request["timed_out"] and self.clock() - request["sent"] > self.settings.timeout:
                request["timed_out"] = True
                self.metrics.tick("requests_timed_out")
                self.last_error = "TPU response timeout; new submissions paused until the late response arrives. Preview continues."
        completed, self.completed = self.completed, []
        return completed

    def status(self):
        return {"inflight": len(self.pending), "max_inflight": self.settings.max_inflight,
                "max_request_fps": self.settings.max_request_fps,
                "oldest_request_ms": round(max((self.clock() - p["sent"] for p in self.pending.values()), default=0) * 1000, 1),
                "paused": any(p["timed_out"] for p in self.pending.values()), "error": self.last_error}

    def close(self):
        if self.connection is not None and self.connection.is_open:
            self.connection.close()

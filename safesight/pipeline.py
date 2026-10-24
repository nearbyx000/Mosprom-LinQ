"""Independent capture, JPEG preview, inference, and event workers."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time
from dataclasses import asdict

import cv2

from .inference import DemoBackend, UltralyticsBackend
from .logic import Zone, RuleEngine
from .rabbitmq import LinQClient, Settings
from .jpeg_gateway import JPEGGatewayClient, JPEGSettings
from .runtime import Frame, Latest, Metrics
from .storage import EventStore
from .video import SyntheticVideo, annotate
from .detections import message as detection_message

logger = logging.getLogger(__name__)


def wait_until(stopped, deadline):
    # Event.wait on Windows can have coarse timer granularity. Python's sleep
    # uses a high-resolution waitable timer; short slices keep shutdown bounded.
    while not stopped.is_set():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def open_camera(source, fps):
    if source == "synthetic":
        return SyntheticVideo()
    if isinstance(source, str) and source.isdecimal():
        source = int(source)
    if isinstance(source, int) and sys.platform == "darwin":
        capture = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    elif isinstance(source, int) and sys.platform == "win32":
        # Preserve the team's verified Windows camera backend from WEB.
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    elif isinstance(source, str) and source.startswith(("http://", "https://", "rtsp://")):
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
    else:
        capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; drivers may ignore this.
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.getenv("CAMERA_WIDTH", "640")))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.getenv("CAMERA_HEIGHT", "480")))
        capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


class CameraWorker:
    def __init__(self, camera, store, evidence_dir, mode="preview", fps=30,
                 camera_factory=open_camera, client_factory=None, backend=None):
        self.id, self.name, self.source = camera["id"], camera["name"], camera["source"]
        self.zones = [Zone(z["id"], z["name"], tuple(tuple(p) for p in z["points"])) for z in camera.get("zones", [])]
        self.store, self.evidence_dir, self.mode = store, evidence_dir, mode
        self.fps = fps
        self.camera_factory = camera_factory
        if client_factory is not None:
            self.client_factory = client_factory
        elif os.getenv("LINQ_JPEG_ROUTE") and mode == "linq":
            self.client_factory = lambda metrics: JPEGGatewayClient(JPEGSettings.from_env(), metrics)
        else:
            self.client_factory = lambda metrics: LinQClient(Settings.from_env(), metrics)
        self.backend = backend
        self.raw, self.preview = Latest(), Latest()
        self.detection_updates = Latest()
        self.results = queue.Queue(maxsize=8)
        self.metrics = Metrics()
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.state = {"status": "starting", "inference_status": "disabled" if mode == "preview" else "starting",
                      "event_status": "ready", "rabbitmq": {}, "last_detection_frame": None,
                      "last_detection_task": None, "tpu_response_received": False}
        self.last_detection_at = None
        self.detection_jpeg = None
        self.last_event = None
        self.warning_snapshot = None
        self.threads = []

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def start(self):
        for name, target in [("capture", self.capture_loop), ("preview", self.preview_loop),
                             ("inference", self.inference_loop), ("events", self.events_loop)]:
            thread = threading.Thread(target=target, name=f"{self.id}-{name}", daemon=True)
            self.threads.append(thread)
            thread.start()

    def stop(self):
        self.stopped.set()
        self.raw.close()
        self.preview.close()
        self.detection_updates.close()
        deadline = time.perf_counter() + 6
        for thread in self.threads:
            thread.join(timeout=max(0, deadline - time.perf_counter()))
        if any(thread.is_alive() for thread in self.threads):
            logger.warning("Camera worker shutdown pending on device I/O")

    def capture_loop(self):
        sequence = 0
        generation = 0
        paced = self.source == "synthetic" or (isinstance(self.source, str) and self.source.lower().endswith((".mp4", ".avi", ".mkv")))
        while not self.stopped.is_set():
            capture = None
            try:
                capture = self.camera_factory(self.source, self.fps)
                if hasattr(capture, "isOpened") and not capture.isOpened():
                    raise RuntimeError("Camera unavailable")
                generation += 1
                next_capture_at = time.perf_counter()
                while not self.stopped.is_set():
                    started = time.perf_counter()
                    ok, image = capture.read()
                    captured_at = time.perf_counter()
                    if not ok or image is None:
                        raise RuntimeError("Camera disconnected")
                    self.metrics.observe("capture_read", (captured_at - started) * 1000)
                    if os.getenv("CAMERA_MIRROR", "1") == "1":
                        image = cv2.flip(image, 1)
                    sequence += 1
                    self.raw.put(Frame(sequence, captured_at, time.time(), image, generation))
                    self.metrics.tick("capture")
                    self.update(status="running")
                    if paced:
                        next_capture_at = max(next_capture_at + 1 / self.fps, started)
                        wait_until(self.stopped, next_capture_at)
            except Exception as exc:
                self.update(status=f"camera unavailable ({type(exc).__name__}); reconnecting")
                self.metrics.tick("capture_errors")
            finally:
                if capture is not None:
                    capture.release()
            self.stopped.wait(1)

    def preview_loop(self):
        sequence, next_frame_at = 0, time.perf_counter()
        while not self.stopped.is_set():
            wait_until(self.stopped, next_frame_at)
            if self.stopped.is_set():
                break
            new_sequence, packet = self.raw.get(sequence)
            if packet is None:
                continue
            if new_sequence > sequence + 1:
                self.metrics.tick("preview_frames_skipped", new_sequence - sequence - 1)
            sequence = new_sequence
            started = time.perf_counter()
            try:
                image = packet.image.copy()
                # Keep the shared video clean; the Web CV canvas draws zones
                # and detections without opening another capture connection.
                ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise ValueError("JPEG encoding failed")
                self.preview.put((packet.sequence, packet.captured_at, jpg.tobytes()))
                self.metrics.observe("jpeg_render", (time.perf_counter() - started) * 1000)
                self.metrics.observe("capture_to_preview", (time.perf_counter() - packet.captured_at) * 1000)
                self.metrics.tick("preview")
            except (cv2.error, ValueError):
                self.metrics.tick("preview_errors")
            # Absolute deadlines avoid accumulating the scheduler's wake-up
            # overhead every frame (visible as ~29 FPS when targeting 30).
            next_frame_at = max(next_frame_at + 1 / self.fps, started)

    def enqueue_result(self, result):
        packet, detections, task_id = result
        # Emit before rules, evidence encoding and SQLite: none of these block
        # the preview or the WebSocket detection path.
        self.detection_updates.put((packet.captured_at, detection_message(
            self.id, packet, detections, self.mode, task_id)))
        try:
            self.results.put_nowait(result)
        except queue.Full:
            try:
                self.results.get_nowait()
            except queue.Empty:
                pass
            self.metrics.tick("event_results_dropped")
            self.results.put_nowait(result)

    def inference_loop(self):
        if self.mode == "preview":
            return
        sequence = 0
        client = None
        try:
            if self.mode == "linq":
                client = self.client_factory(self.metrics)
                while not self.stopped.is_set():
                    try:
                        self.update(inference_status="connecting to RabbitMQ")
                        client.connect()
                        break
                    except Exception as exc:
                        self.update(inference_status=f"RabbitMQ connection failed ({type(exc).__name__}); retry in 5s")
                        if self.stopped.wait(5):
                            return
                while not self.stopped.is_set():
                    for packet, detections, task_id in client.poll():
                        self.enqueue_result((packet, detections, task_id))
                        self.update(tpu_response_received=True)
                    state = client.status()
                    self.update(rabbitmq=state, inference_status=state["error"] or "running")
                    if client.can_submit():
                        new_sequence, packet = self.raw.get(sequence, timeout=0.01)
                        if packet is not None and time.perf_counter() - packet.captured_at < 1:
                            if new_sequence > sequence + 1:
                                self.metrics.tick("inference_frames_skipped", new_sequence - sequence - 1)
                            sequence = new_sequence
                            client.submit(packet)
                    self.stopped.wait(0.005)
            else:
                self.update(inference_status=self.backend.name)
                while not self.stopped.is_set():
                    new_sequence, packet = self.raw.get(sequence)
                    if packet is None:
                        continue
                    if new_sequence > sequence + 1:
                        self.metrics.tick("inference_frames_skipped", new_sequence - sequence - 1)
                    sequence = new_sequence
                    started = time.perf_counter()
                    detections = self.backend.infer(packet.image)
                    self.metrics.observe("backend_call", (time.perf_counter() - started) * 1000)
                    self.metrics.observe("capture_to_detection", (time.perf_counter() - packet.captured_at) * 1000)
                    self.metrics.tick("detections")
                    self.enqueue_result((packet, detections, None))
        except Exception as exc:
            # Lost connections may have left work executing remotely. Do not
            # auto-resubmit or create more outstanding jobs on the shared TPU.
            message = str(exc) if isinstance(exc, (ValueError, RuntimeError, ImportError)) else type(exc).__name__
            self.update(inference_status=f"stopped: {message}; preview continues")
            if client is not None:
                state = client.status()
                state["paused"] = True
                state["connection_lost"] = True
                self.update(rabbitmq=state)
            self.metrics.tick("inference_errors")
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def events_loop(self):
        engine = RuleEngine()
        last_drops = 0
        generation = None
        while not self.stopped.is_set():
            try:
                packet, detections, task_id = self.results.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                drops = self.metrics.snapshot()["counts"].get("event_results_dropped", 0)
                if drops != last_drops or packet.generation != generation:
                    engine = RuleEngine()  # Missing observations cannot confirm continuity.
                    last_drops = drops
                    generation = packet.generation
                height, width = packet.image.shape[:2]
                tracked, events = engine.process(self.id, detections, self.zones, width, height)
                with self.lock:
                    self.warning_snapshot = {
                        "captured_at": packet.captured_at, "timestamp": packet.captured_wall,
                        "frame_id": packet.sequence, "source": self.mode,
                        "items": [asdict(event) for event in engine.active_violations],
                    }
                annotated = packet.image.copy()
                annotate(annotated, self.zones, tracked)
                ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise ValueError("Evidence JPEG encoding failed")
                jpeg = jpg.tobytes()
                with self.lock:
                    self.detection_jpeg = jpeg
                    self.last_detection_at = packet.captured_at
                    self.state.update(last_detection_frame=packet.sequence, last_detection_task=task_id)
                for event in events:
                    safe_camera = "".join(c for c in self.id if c.isalnum() or c in "_-")
                    filename = f"{safe_camera}-{time.time_ns()}-{event.violation_type}.jpg"
                    (self.evidence_dir / filename).write_bytes(jpeg)
                    event_id = self.store.insert(event.camera_id, event.zone_id, event.violation_type,
                                                 event.track_id, event.confidence, filename,
                                                 captured_at=packet.captured_wall)
                    with self.lock:
                        self.last_event = {"id": event_id, "camera_id": self.id, "type": event.violation_type}
                    self.metrics.tick("events")
                    self.metrics.observe("capture_to_event", (time.perf_counter() - packet.captured_at) * 1000)
                self.update(event_status="ready")
            except Exception as exc:
                self.update(event_status=f"failed ({type(exc).__name__})")
                self.metrics.tick("event_errors")
            self.metrics.observe("event_processing", (time.perf_counter() - started) * 1000)

    def status(self):
        now = time.perf_counter()
        with self.lock:
            result = dict(self.state)
            result.update(id=self.id, name=self.name,
                          last_detection_age_ms=round((now - self.last_detection_at) * 1000, 1) if self.last_detection_at else None,
                          last_event=self.last_event, zones=[{"id": z.id, "name": z.name, "points": z.points} for z in self.zones])
            warning = self.warning_snapshot
            fresh = warning is not None and now - warning["captured_at"] < 3
            result.update(warnings_fresh=fresh,
                          warnings_updated_at=warning["timestamp"] if warning else None,
                          warnings_frame_id=warning["frame_id"] if warning else None,
                          active_warnings=warning["items"] if fresh else [])
        _, preview = self.preview.get(timeout=0)
        preview_age = (now - preview[1]) * 1000 if preview else None
        result["preview_age_ms"] = round(preview_age, 1) if preview_age is not None else None
        result["preview_available"] = preview_age is not None and preview_age < 2000
        result["metrics"] = self.metrics.snapshot()
        return result


class Pipeline:
    def __init__(self, config_path, data_dir, mode="preview", model_path="", camera_factory=open_camera, client_factory=None):
        if mode not in {"preview", "linq", "demo", "local"}:
            raise ValueError("SAFESIGHT_MODE must be preview, linq, demo, or local")
        self.mode = mode
        self.evidence_dir = Path(data_dir) / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.store = EventStore(Path(data_dir) / "events.sqlite")
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        cameras = cfg["cameras"]
        # Keep a single shared-server inference budget for this MVP.
        if len(cameras) != 1:
            raise ValueError("MVP supports exactly one camera and one LinQ connection")
        backend = DemoBackend() if mode == "demo" else UltralyticsBackend(model_path) if mode == "local" else None
        self.backend_name = "LinQ TPU / RabbitMQ" if mode == "linq" else backend.name if backend else "Preview only / inference disabled"
        self.workers = {}
        fps = float(os.getenv("CAMERA_FPS", "30"))
        if not 0 < fps <= 120:
            raise ValueError("CAMERA_FPS must be in (0,120]")
        for camera in cameras:
            camera = dict(camera)
            if "CAMERA_SOURCE" in os.environ:
                camera["source"] = os.environ["CAMERA_SOURCE"]
            self.workers[camera["id"]] = CameraWorker(camera, self.store, self.evidence_dir, mode, fps,
                                                     camera_factory, client_factory, backend)

    def start(self):
        for worker in self.workers.values():
            worker.start()

    def stop(self):
        for worker in self.workers.values():
            worker.stop()

    def status(self):
        return {"mode": self.mode, "backend": self.backend_name,
                "cameras": [worker.status() for worker in self.workers.values()],
                "metric_note": "rpc_round_trip includes network, queue and TPU; separate TPU/network timings require worker instrumentation. Preview timing ends at JPEG-ready, not browser display."}

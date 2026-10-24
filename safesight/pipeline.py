"""Per-camera producer thread, visualization and storage of confirmed events."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np

from .logic import Detection, Zone, RuleEngine, Violation
from .inference import DemoBackend, UltralyticsBackend, LinQBackend, InferenceBackend
from .storage import EventStore


class SyntheticVideo:
    """Factory-only demo. Distinctly not a real industrial image source."""
    def __init__(self):
        self.frame = 0

    def read(self):
        width, height = 960, 540
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (29, 39, 45)
        for x in range(0, width, 80):
            cv2.line(canvas, (x, 0), (x, height), (43, 52, 57), 1)
        for y in range(0, height, 80):
            cv2.line(canvas, (0, y), (width, y), (43, 52, 57), 1)
        # Walk in and out of restricted zone, with magenta no-helmet label.
        # Separate tracks/episodes should create separate alarms.
        cycle = self.frame % 210
        px = 70 + int(cycle * 4.1) if cycle < 170 else -200
        py = 230
        if 0 <= px <= width - 100:
            cv2.rectangle(canvas, (px, py), (px + 80, py + 180), (0, 255, 0), -1)
            cv2.rectangle(canvas, (px + 22, py - 24), (px + 62, py - 5), (255, 0, 255), -1)
        cv2.putText(canvas, "SYNTHETIC DEMO ONLY - NO TPU", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 210, 210), 2)
        self.frame += 1
        return True, canvas

    def release(self):
        pass


@dataclass
class CameraState:
    id: str
    name: str
    source: str
    zones: list[Zone]
    frame_jpeg: bytes | None = None
    fps: float = 0.0
    last_inference_ms: float = 0.0
    status: str = "starting"
    frame_count: int = 0


class CameraWorker(threading.Thread):
    def __init__(self, state: CameraState, backend: InferenceBackend, store: EventStore,
                 evidence_dir: Path, fps_cap: float = 5.0):
        super().__init__(daemon=True, name=f"camera-{state.id}")
        self.state = state
        self.backend = backend
        self.store = store
        self.evidence_dir = evidence_dir
        self.engine = RuleEngine()
        self.fps_cap = fps_cap
        self.stopped = threading.Event()
        self.lock = threading.Lock()

    def run(self):
        reader = SyntheticVideo() if self.state.source == "synthetic" else cv2.VideoCapture(self.state.source)
        if not isinstance(reader, SyntheticVideo) and not reader.isOpened():
            with self.lock:
                self.state.status = "failed: video source could not be opened"
            return
        try:
            prev = time.monotonic()
            while not self.stopped.is_set():
                start_loop = time.monotonic()
                ok, frame = reader.read()
                if not ok or frame is None:
                    # Video-file loop only; RTSP failure should remain visible.
                    if self.state.source.lower().endswith((".mp4", ".avi", ".mkv")):
                        reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        # Reset incident state when video loops to avoid merging incidents.
                        self.engine = RuleEngine()
                        continue
                    with self.lock:
                        self.state.status = "failed: stream disconnected"
                    break
                t0 = time.perf_counter()
                detections = self.backend.infer(frame)
                inference_ms = (time.perf_counter() - t0) * 1000
                h, w = frame.shape[:2]
                tracked, events = self.engine.process(self.state.id, detections, self.state.zones, w, h)
                annotated = frame.copy()
                annotate(annotated, self.state.zones, tracked)
                for ev in events:
                    safe_cam = "".join(c for c in self.state.id if c.isalnum() or c in "_-")
                    filename = f"{safe_cam}-{time.time_ns()}-{ev.violation_type}.jpg"
                    cv2.imwrite(str(self.evidence_dir / filename), annotated)
                    self.store.insert(ev.camera_id, ev.zone_id, ev.violation_type, ev.track_id, ev.confidence, filename)
                success, data = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 76])
                dt = max(0.001, time.monotonic() - prev)
                prev = time.monotonic()
                with self.lock:
                    self.state.frame_jpeg = bytes(data) if success else None
                    self.state.last_inference_ms = round(inference_ms, 1)
                    self.state.fps = round(1 / dt, 1)
                    self.state.frame_count += 1
                    self.state.status = "running"
                self.stopped.wait(max(0, 1 / self.fps_cap - (time.monotonic() - start_loop)))
        except Exception as exc:
            with self.lock:
                self.state.status = f"failed: {type(exc).__name__}: {exc}"
        finally:
            reader.release()


def annotate(frame: np.ndarray, zones: list[Zone], detections: list[Detection]):
    h, w = frame.shape[:2]
    for zone in zones:
        points = np.array([(int(x * w), int(y * h)) for x, y in zone.points], dtype=np.int32)
        cv2.polylines(frame, [points], True, (0, 172, 255), 3)
        anchor = tuple(points[0])
        cv2.putText(frame, zone.id, (int(anchor[0]), max(20, int(anchor[1]) - 9)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 255), 2)
    for det in detections:
        x1, y1, x2, y2 = det.box
        color = (0, 0, 255) if det.label == "no_helmet" else (30, 210, 60)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{det.label} #{det.track_id} {det.confidence:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .50, color, 2)


class Pipeline:
    def __init__(self, config_path: Path, data_dir: Path, mode: str, model_path: str = ""):
        self.mode = mode
        if mode == "demo":
            backend = DemoBackend()
        elif mode == "local":
            backend = UltralyticsBackend(model_path)
        elif mode == "linq":
            backend = LinQBackend()
        else:
            raise ValueError(f"Unknown mode {mode!r}: expected demo|local|linq")
        self.backend_name = backend.name
        self.data_dir = data_dir
        self.evidence_dir = data_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.store = EventStore(data_dir / "events.sqlite")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        self.workers: dict[str, CameraWorker] = {}
        for item in cfg["cameras"]:
            zones = [Zone(id=z["id"], name=z["name"], points=tuple(tuple(float(v) for v in p) for p in z["points"])) for z in item.get("zones", [])]
            state = CameraState(id=item["id"], name=item["name"], source=item["source"], zones=zones)
            if state.id in self.workers:
                raise ValueError(f"Duplicate camera id: {state.id}")
            self.workers[state.id] = CameraWorker(state, backend, self.store, self.evidence_dir)
        # One backend object shared across camera threads is safe only for DEMO.
        # Other modes intentionally restrict to one camera for MVP.
        if mode != "demo" and len(self.workers) != 1:
            raise ValueError("MVP: local/LinQ backend supports only one camera at a time")

    def start(self):
        for worker in self.workers.values():
            worker.start()

    def stop(self):
        for worker in self.workers.values():
            worker.stopped.set()
        for worker in self.workers.values():
            worker.join(timeout=2)

    def status(self):
        cameras = []
        for worker in self.workers.values():
            with worker.lock:
                state = worker.state
                cameras.append({"id": state.id, "name": state.name, "source": state.source,
                                "fps": state.fps, "inference_ms": state.last_inference_ms,
                                "status": state.status, "frame_count": state.frame_count,
                                "zones": [{"id": z.id, "name": z.name} for z in state.zones]})
        return {"mode": self.mode, "backend": self.backend_name, "linq_tpu_verified": False,
                "cameras": cameras, "metric_note": "inference_ms measures backend call only, not end-to-end alert latency"}

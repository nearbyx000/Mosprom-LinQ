"""Concurrency checks with a slow fake backend, not a TPU benchmark."""
import threading
import time

import numpy as np

from safesight.logic import Detection
from safesight.pipeline import CameraWorker
from safesight.runtime import Latest
from safesight.storage import EventStore


class Camera:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        time.sleep(1 / 30)
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class BlockedBackend:
    name = "TEST FAKE / NOT TPU"

    def __init__(self):
        self.entered = threading.Event()
        self.unblock = threading.Event()

    def infer(self, frame):
        self.entered.set()
        if not self.unblock.wait(timeout=10):
            raise TimeoutError("Test did not release the fake backend")
        return [Detection("person", 0.9, (20, 10, 40, 40))]


def test_preview_survives_slow_inference_and_events_persist(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    camera = Camera()
    backend = BlockedBackend()
    store = EventStore(tmp_path / "events.sqlite")
    worker = CameraWorker({"id": "test", "name": "Test", "source": 0,
                           "zones": [{"id": "danger", "name": "Danger", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}]},
                          store, evidence, mode="local", backend=backend, camera_factory=lambda *_: camera)
    worker.start()
    try:
        assert backend.entered.wait(timeout=5)
        initial = worker.metrics.snapshot()["counts"]
        deadline = time.perf_counter() + 5
        while time.perf_counter() < deadline:
            counts = worker.metrics.snapshot()["counts"]
            if counts.get("preview", 0) >= initial.get("preview", 0) + 5:
                break
            time.sleep(0.01)
        # Prove actual progress while inference cannot possibly finish. CI
        # runner scheduling does not need to meet a hardware FPS benchmark.
        assert counts["capture"] >= initial.get("capture", 0) + 5
        assert counts["preview"] >= initial.get("preview", 0) + 5
        assert counts.get("detections", 0) == 0
        backend.unblock.set()
        deadline = time.monotonic() + 5
        while not store.list_events() and time.monotonic() < deadline:
            time.sleep(0.05)
        events = store.list_events()
        assert len(events) == 1
        assert (evidence / events[0]["evidence_file"]).read_bytes().startswith(bytes([255, 216]))
        assert worker.status()["tpu_response_received"] is False
    finally:
        backend.unblock.set()
        worker.stop()
    assert camera.released
    assert all(not thread.is_alive() for thread in worker.threads)


def test_latest_slot_drops_history_and_wakes_on_close():
    slot = Latest()
    for number in range(100):
        slot.put(number)
    sequence, value = slot.get()
    assert (sequence, value) == (100, 99)
    result = []
    waiter = threading.Thread(target=lambda: result.append(slot.get(sequence, timeout=5)))
    waiter.start()
    slot.close()
    waiter.join(timeout=1)
    assert result == [(100, None)]

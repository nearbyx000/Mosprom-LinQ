"""Inference backend boundary. Never silently substitute local inference for TPU."""
from __future__ import annotations

from typing import Protocol
import numpy as np
import cv2

from .logic import Detection


class InferenceBackend(Protocol):
    name: str
    def infer(self, bgr_frame: np.ndarray) -> list[Detection]: ...


class DemoBackend:
    """Only finds hand-drawn green/magenta rectangles in synthetic video.

    NOT neural inference. NOT LinQ TPU. Do not present this as a working CV model.
    """
    name = "DEMO / synthetic color extraction (NO TPU)"

    def infer(self, bgr_frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for label, low, high in (
            ("person", (0, 230, 0), (30, 255, 30)),
            ("no_helmet", (220, 0, 220), (255, 50, 255)),
        ):
            mask = cv2.inRange(bgr_frame, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            for idx in range(1, n):
                x, y, w, h, area = (int(v) for v in stats[idx])
                if area >= (280 if label == "person" else 80):
                    detections.append(Detection(label, 0.99, (x, y, x + w, y + h)))
        return detections


class UltralyticsBackend:
    name = "LOCAL Ultralytics (NO TPU)"

    def __init__(self, path: str):
        if not path:
            raise ValueError("Set SAFESIGHT_MODEL to your model file")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install: pip install -r requirements-local-ml.txt") from exc
        self.model = YOLO(path)

    def infer(self, bgr_frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(bgr_frame, verbose=False, conf=0.40)[0]
        detections: list[Detection] = []
        names = result.names
        for box in result.boxes:
            label = str(names[int(box.cls.item())]).strip().lower().replace(" ", "_").replace("-", "_")
            # Only explicit negative PPE labels, never missing helmet detection.
            if label in {"without_helmet", "no_hardhat", "no_hard_hat", "without_hardhat"}:
                label = "no_helmet"
            if label in {"person", "no_helmet"}:
                xyxy = tuple(int(round(x)) for x in box.xyxy[0].tolist())
                detections.append(Detection(label, float(box.conf.item()), xyxy))
        return detections


class LinQBackend:
    name = "LinQ TPU / NOT INTEGRATED"

    def __init__(self):
        # Must be implemented only after inspecting the real examples from LinQ.
        # Do NOT guess API methods or silently use a local model.
        raise RuntimeError(
            "LinQ API adapter not implemented: obtain SDK tpu_framework.pdf and "
            "usage_example_python_api.zip, compile ONNX to a TPU program, "
            "then implement official calls in safesight/inference.py. "
            "See docs/LINQ.md."
        )

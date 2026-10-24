"""Existing demo source and annotation helpers from LinQ-TPU."""
import cv2
import numpy as np
from .logic import Zone, Detection


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

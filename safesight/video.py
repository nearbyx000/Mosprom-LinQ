"""Existing demo source and annotation helpers from LinQ-TPU."""
import cv2
import numpy as np
from .logic import Zone, Detection


class SyntheticVideo:
    """Factory-only demo. Distinctly not a real industrial image source."""
    def __init__(self):
        self.frame = 0
        self.scenario = 'normal'

    def read(self):
        width, height = 960, 540
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (29, 39, 45)
        for x in range(0, width, 80):
            cv2.line(canvas, (x, 0), (x, height), (43, 52, 57), 1)
        for y in range(0, height, 80):
            cv2.line(canvas, (0, y), (width, y), (43, 52, 57), 1)
        # Five six-second scenes at the generated source's 30 FPS. The head is
        # inside the body bbox, matching the real person/PPE association rules.
        phase = (self.frame//180)%5
        progress = (self.frame%180)/179
        self.scenario = ('normal','zone_entry','no_helmet','fall','clear')[phase]
        if phase < 3:
            px = 160 if phase==0 else int(350+progress*240) if phase==1 else 620
            cv2.rectangle(canvas,(px,185),(px+90,465),(0,255,0),-1)
            head = (255,0,255) if phase==2 else (0,200,255)
            cv2.rectangle(canvas,(px+22,190),(px+68,225),head,-1)
        elif phase==3:
            cv2.rectangle(canvas,(565,390),(825,465),(255,120,0),-1)
        cv2.putText(canvas, 'SYNTHETIC DEMO - NO TPU / NO NEURAL INFERENCE',
                    (20,38),cv2.FONT_HERSHEY_SIMPLEX,.62,(210,210,210),1)
        cv2.putText(canvas, 'SCENE: '+self.scenario.upper(),(20,75),cv2.FONT_HERSHEY_SIMPLEX,.65,(160,180,190),1)
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

"""YOLOv8 COCO preprocessing and decode, with no local model inference."""
from dataclasses import dataclass

import cv2
import numpy as np

from .logic import Detection


@dataclass(frozen=True)
class Transform:
    scale: float
    left: int
    top: int
    width: int
    height: int


def preprocess(frame):
    height, width = frame.shape[:2]
    scale = min(640 / width, 640 / height)
    new_width, new_height = round(width * scale), round(height * scale)
    left, top = (640 - new_width) // 2, (640 - new_height) // 2
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[top:top + new_height, left:left + new_width] = cv2.resize(frame, (new_width, new_height))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0
    return tensor, Transform(scale, left, top, width, height)


def postprocess(output, transform, confidence=0.45, iou=0.45):
    output = np.asarray(output)
    if output.shape != (1, 84, 8400):
        raise ValueError(f"Expected YOLOv8 COCO output [1,84,8400], got {output.shape}")
    pred = output[0].T
    class_ids = np.argmax(pred[:, 4:], axis=1)
    scores = np.max(pred[:, 4:], axis=1)
    # This adapter is specifically a COCO person detector. PPE is a separate model.
    valid = np.flatnonzero((class_ids == 0) & (scores >= confidence) & np.isfinite(pred).all(axis=1))
    if not valid.size:
        return []
    boxes = pred[valid, :4].copy()
    scores = scores[valid]
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 0] = (boxes[:, 0] - transform.left) / transform.scale
    boxes[:, 1] = (boxes[:, 1] - transform.top) / transform.scale
    boxes[:, 2:] /= transform.scale
    x2 = np.clip(boxes[:, 0] + boxes[:, 2], 0, transform.width)
    y2 = np.clip(boxes[:, 1] + boxes[:, 3], 0, transform.height)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, transform.width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, transform.height)
    boxes[:, 2] = x2 - boxes[:, 0]
    boxes[:, 3] = y2 - boxes[:, 1]
    keep = np.flatnonzero((boxes[:, 2] > 0) & (boxes[:, 3] > 0))
    boxes, scores = boxes[keep], scores[keep]
    if not len(boxes):
        return []
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), confidence, iou)
    detections = []
    for index in np.asarray(indices).reshape(-1):
        x, y, width, height = boxes[int(index)]
        detections.append(Detection("person", float(scores[int(index)]),
                                    tuple(round(float(v)) for v in (x, y, x + width, y + height))))
    return detections

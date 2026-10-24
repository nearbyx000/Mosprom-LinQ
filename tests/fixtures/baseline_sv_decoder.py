"""SafetyVision v2 YOLOv8s raw output decoder (1,17,8400) -> pixel-space detections.

Runs after real LinQ TPU inference. Does NOT run a local model.
"""
from __future__ import annotations
import numpy as np

CLASSES = [
    'Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Mask',
    'NO-Gloves', 'NO-Goggles', 'NO-Hardhat', 'NO-Mask',
    'NO-Safety Vest', 'No_Harness', 'Person', 'Safety Vest',
]
# These canonical labels are friendly to SafeSight's RuleEngine.
NORMALIZED = [
    'fall_detected', 'gloves', 'goggles', 'hardhat', 'mask',
    'no_gloves', 'no_goggles', 'no_helmet', 'no_mask',
    'no_safety_vest', 'no_harness', 'person', 'safety_vest',
]

def _nms(boxes, scores, iou_threshold, limit):
    order = np.argsort(-scores, kind='stable')
    selected = []
    while len(order) and len(selected) < limit:
        idx = int(order[0]); selected.append(idx)
        remaining = order[1:]
        if not len(remaining):
            break
        left = np.maximum(boxes[idx, 0], boxes[remaining, 0])
        top = np.maximum(boxes[idx, 1], boxes[remaining, 1])
        right = np.minimum(boxes[idx, 2], boxes[remaining, 2])
        bottom = np.minimum(boxes[idx, 3], boxes[remaining, 3])
        intersection = np.maximum(0, right-left) * np.maximum(0, bottom-top)
        a = np.maximum(0, boxes[idx, 2]-boxes[idx, 0]) * np.maximum(0, boxes[idx, 3]-boxes[idx, 1])
        b = np.maximum(0, boxes[remaining, 2]-boxes[remaining, 0]) * np.maximum(0, boxes[remaining, 3]-boxes[remaining, 1])
        overlaps = intersection / np.maximum(a+b-intersection,1e-9)
        order = remaining[overlaps <= iou_threshold]
    return selected


def safetyvision_detections(raw, meta, confidence=0.4, iou_threshold=0.45,
                            include_person=False, max_detections=100):
    """Decode raw YOLO output with class-specific NMS.

    meta: (original_width, original_height, letterbox_scale, pad_left, pad_top)
    Note: normalized class labels still represent uncertain detector predictions.
    """
    arr = np.asarray(raw)
    if arr.shape == (1, 17, 8400):
        arr = arr[0].T
    elif arr.shape == (1, 8400, 17):
        arr = arr[0]
    else:
        raise ValueError(f'Expected SafetyVision 640 raw tensor [1,17,8400], got {arr.shape}')
    if not (0 <= confidence <= 1 and 0 < iou_threshold < 1):
        raise ValueError('Invalid detection threshold')
    original_w, original_h, ratio, left, top = meta
    if ratio <= 0 or original_w <= 0 or original_h <= 0:
        raise ValueError('Invalid letterbox metadata')
    scores_all = arr[:, 4:]
    # Multi-label candidates: an anchor may nominate several classes; NMS per class.
    valid_scores = np.isfinite(scores_all)
    ids, classes = np.nonzero(valid_scores & (scores_all >= confidence))
    if not len(ids):
        return []
    raw_boxes = arr[ids, :4].astype(np.float64)
    finite = np.isfinite(raw_boxes).all(axis=1)
    ids, classes, raw_boxes = ids[finite], classes[finite], raw_boxes[finite]
    if not include_person:
        keep = classes != 11
        ids, classes, raw_boxes = ids[keep], classes[keep], raw_boxes[keep]
    if not len(ids):
        return []
    scores = scores_all[ids, classes].astype(np.float64)
    cx, cy, w, h = raw_boxes.T
    boxes = np.stack([(cx-w/2-left)/ratio, (cy-h/2-top)/ratio,
                      (cx+w/2-left)/ratio, (cy+h/2-top)/ratio], axis=1)
    boxes[:, [0,2]] = np.clip(boxes[:, [0,2]],0,original_w)
    boxes[:, [1,3]] = np.clip(boxes[:, [1,3]],0,original_h)
    valid = (boxes[:,2]>boxes[:,0]) & (boxes[:,3]>boxes[:,1])
    boxes, classes, scores = boxes[valid], classes[valid], scores[valid]
    result=[]
    for cls in np.unique(classes):
        indices=np.flatnonzero(classes==cls)
        for local_index in _nms(boxes[indices],scores[indices],iou_threshold,max_detections):
            i=indices[local_index]
            result.append({'class_name':NORMALIZED[int(cls)],
                           'confidence':round(float(scores[i]),5),
                           'bbox':[int(round(v)) for v in boxes[i]],
                           'model':'linq-safetyvision-v2'})
    result.sort(key=lambda d:d['confidence'],reverse=True)
    return result[:max_detections]

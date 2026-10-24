"""One-to-one IoU metrics shared by detection and incident replay tools."""
from __future__ import annotations

import math
from .logic import iou


def canonical(item):
    label = item.get('label', item.get('class_name', item.get('violation_type')))
    box = item.get('box', item.get('bbox'))
    confidence = item.get('confidence', 1.0)
    if (not isinstance(label, str) or not isinstance(box, (list, tuple)) or len(box) != 4 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box) or
            box[2] <= box[0] or box[3] <= box[1] or isinstance(confidence, bool) or
            not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError('Invalid evaluation annotation or prediction')
    return {'label': label, 'box': tuple(box), 'confidence': confidence}


def rates(tp, fp, fn):
    return {'TP': tp, 'FP': fp, 'FN': fn,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'F1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def evaluate(expected, predicted, minimum_iou=.5, duration_seconds=None):
    """Each case/image must be present, including explicit empty predictions.

    Greedy confidence-ordered matching; duplicates count as false positives.
    An incident's location is compared at its first confirmation frame. For
    long moving incidents annotate that location or evaluate image detections.
    No true negatives are inferred from unlabelled material.
    """
    if set(expected) != set(predicted) or not 0 < minimum_iou <= 1:
        raise ValueError('Expected and predicted IDs must match exactly; IoU must be in (0,1]')
    counts = {}
    detected_cases = positive_cases = false_alarm_cases = 0
    for case_id, truth in expected.items():
        truth = [canonical(x) for x in truth]
        predictions = sorted((canonical(x) for x in predicted[case_id]), key=lambda x: -x['confidence'])
        used = set()
        case_tp = case_fp = 0
        for p in predictions:
            entry = counts.setdefault(p['label'], [0, 0, 0])
            matches = [(iou(p['box'], t['box']), j) for j, t in enumerate(truth)
                       if j not in used and p['label'] == t['label']]
            score, j = max(matches, default=(0, -1))
            if score >= minimum_iou:
                used.add(j)
                entry[0] += 1
                case_tp += 1
            else:
                entry[1] += 1
                case_fp += 1
        for j, t in enumerate(truth):
            entry = counts.setdefault(t['label'], [0, 0, 0])
            if j not in used:
                entry[2] += 1
        positive_cases += bool(truth)
        detected_cases += bool(case_tp)
        false_alarm_cases += bool(case_fp)
    totals = [sum(c[i] for c in counts.values()) for i in range(3)]
    if duration_seconds is not None and (not math.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError('duration_seconds must be finite and positive')
    return {'total': rates(*totals), 'per_class': {k: rates(*v) for k, v in sorted(counts.items())},
            'cases': len(expected), 'positive_cases': positive_cases,
            'detected_positive_cases': detected_cases, 'false_alarm_cases': false_alarm_cases,
            'false_alarms': totals[1], 'duration_seconds': duration_seconds,
            'false_alarms_per_hour': totals[1]*3600/duration_seconds if duration_seconds else None,
            'matching_iou': minimum_iou}

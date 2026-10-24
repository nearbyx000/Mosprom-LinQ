"""Conservative SafetyVision evidence association; never infer missing PPE.

Thresholds are initial operating points, not calibrated quality claims. Keep
this module independent of neural runtimes so replay uses the production rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os


PAIRS = {
    'no_helmet': 'hardhat', 'no_gloves': 'gloves', 'no_goggles': 'goggles',
    'no_mask': 'mask', 'no_safety_vest': 'safety_vest',
}
LABELS = frozenset((*PAIRS, *PAIRS.values(), 'no_harness', 'fall_detected'))


@dataclass
class SafetyPolicy:
    thresholds: dict[str, float] = field(default_factory=lambda: {
        label: (.45 if label in {'no_helmet', 'fall_detected'} else .40)
        for label in LABELS
    })
    conflict_margin: float = .10
    max_observation_gap: float = 2.5
    confirm_window: int = 4

    def __post_init__(self):
        if (set(self.thresholds) != LABELS or any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or not 0 <= v <= 1
                for v in self.thresholds.values())):
            raise ValueError('SafetyVision thresholds must name known classes with values in [0,1]')
        if not 0 <= self.conflict_margin <= 1 or not 0 < self.max_observation_gap < float('inf'):
            raise ValueError('Invalid SafetyVision conflict margin / observation gap')
        if isinstance(self.confirm_window, bool) or not isinstance(self.confirm_window, int) or not 1 <= self.confirm_window <= 20:
            raise ValueError('SafetyVision confirmation window must be an integer in [1,20]')

    @classmethod
    def from_env(cls):
        defaults = cls()
        overrides = json.loads(os.getenv('SAFESIGHT_SV_THRESHOLDS', '{}'))
        if not isinstance(overrides, dict) or set(overrides) - LABELS:
            raise ValueError('SAFESIGHT_SV_THRESHOLDS must be a JSON object of canonical SV labels')
        return cls({**defaults.thresholds, **overrides},
                   max_observation_gap=float(os.getenv('SAFESIGHT_SV_MAX_GAP', '2.5')),
                   confirm_window=int(os.getenv('SAFESIGHT_SV_CONFIRM_WINDOW', '4')))


def overlap(a, b):
    """Intersection over smaller area: handles a helmet inside a head box."""
    common = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = min((a[2]-a[0]) * (a[3]-a[1]), (b[2]-b[0]) * (b[3]-b[1]))
    return common / max(area, 1)


def person_for(det, people, head=False):
    """Require containment and a plausible body region; abstain if ambiguous.

    PPE is attached to COCO person tracks, not to an independent moving head
    track. A cropped head with no COCO person is deliberately unconfirmed.
    Falls are handled separately: COCO can miss a horizontal person.
    """
    x1, y1, x2, y2 = det.box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    area = max(1, (x2-x1)*(y2-y1))
    matches = []
    for person in people:
        px1, py1, px2, py2 = person.box
        pw, ph = px2-px1, py2-py1
        if pw <= 0 or ph <= 0 or not px1 <= cx <= px2:
            continue
        bottom = py1 + (.45 if head else 1.0)*ph
        if not py1 - .05*ph <= cy <= bottom:
            continue
        intersection = max(0, min(x2, px2)-max(x1, px1)) * max(0, min(y2, py2)-max(y1, py1))
        if intersection / area < .6:
            continue
        score = intersection/area - .2*abs(cx-(px1+px2)/2)/pw
        matches.append((score, person))
    matches.sort(key=lambda pair: pair[0], reverse=True)
    if not matches or (len(matches) > 1 and matches[0][0] - matches[1][0] < .1):
        return None
    return matches[0][1]


def safety_evidence(detections, people, policy):
    """Return eligible detections, with PPE IDs inherited from person tracks.

    Resolve mutually exclusive classes before applying violation thresholds;
    a weaker positive can veto a nearly equal negative. A strong negative may
    win only by the configured margin. No-harness has no positive counterpart.
    """
    eligible = [d for d in detections if d.label in LABELS and
                math.isfinite(d.confidence) and 0 <= d.confidence <= 1 and
                d.confidence >= policy.thresholds[d.label]]
    assigned = {}
    for d in eligible:
        if d.label != 'fall_detected':
            person = person_for(d, people, head=d.label in {'hardhat', 'no_helmet', 'mask', 'no_mask', 'goggles', 'no_goggles'})
            if person is not None:
                assigned[id(d)] = person.track_id
    result = []
    for d in eligible:
        if d.label != 'fall_detected' and id(d) not in assigned:
            continue
        contrary = PAIRS.get(d.label)
        if contrary:
            conflicts = [p for p in eligible if p.label == contrary and
                         assigned.get(id(p)) == assigned.get(id(d)) and
                         (d.label == 'no_helmet' or overlap(d.box, p.box) >= .3)]
            if conflicts and d.confidence - max(p.confidence for p in conflicts) < policy.conflict_margin:
                continue
        if d.label != 'fall_detected':
            d.track_id = assigned[id(d)]
        result.append(d)
    return result

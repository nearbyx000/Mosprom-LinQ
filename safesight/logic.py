"""Pure-Python geometry, lightweight tracking and one-event-per-incident logic.

IMPORTANT: centroid/IoU tracks are short-lived object continuity estimates, not
identification of a person. A production tracker should replace this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .safety import SafetyPolicy, safety_evidence


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    track_id: int | None = None


@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    points: tuple[tuple[float, float], ...]  # Normalized x/y in [0,1]


@dataclass(frozen=True)
class Violation:
    camera_id: str
    zone_id: str
    violation_type: str
    track_id: int
    confidence: float
    box: tuple[int, int, int, int]


def inside_polygon(x: float, y: float, vertices: Sequence[tuple[float, float]]) -> bool:
    """Ray-cast polygon containment; vertices are normalized image coordinates."""
    inside = False
    if len(vertices) < 3:
        return False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        # Include horizontal/non-horizontal edges within numeric tolerance.
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        if abs(cross) < 1e-10 and min(xi, xj) - 1e-10 <= x <= max(xi, xj) + 1e-10 and min(yi, yj) - 1e-10 <= y <= max(yi, yj) + 1e-10:
            return True
        if (yi > y) != (yj > y):
            cross_x = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < cross_x:
                inside = not inside
        j = i
    return inside


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    common = max(0, x2 - x1) * max(0, y2 - y1)
    if not common:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return common / (area_a + area_b - common)


@dataclass
class _Track:
    label: str
    box: tuple[int, int, int, int]
    last_frame: int


class IoUTracker:
    def __init__(self, minimum_iou: float = 0.25, max_age: int = 12):
        self.minimum_iou = minimum_iou
        self.max_age = max_age
        self.tracks: dict[int, _Track] = {}
        self.next_id = 1
        self.frame_no = 0

    def update(self, detections: list[Detection]) -> list[Detection]:
        self.frame_no += 1
        self.tracks = {k: v for k, v in self.tracks.items() if self.frame_no - v.last_frame <= self.max_age}
        candidates = []
        for idx, detection in enumerate(detections):
            for track_id, track in self.tracks.items():
                if track.label == detection.label:
                    similarity = iou(track.box, detection.box)
                    if similarity >= self.minimum_iou:
                        candidates.append((similarity, idx, track_id))
        used_detections, used_tracks = set(), set()
        for _, idx, track_id in sorted(candidates, reverse=True):
            if idx in used_detections or track_id in used_tracks:
                continue
            detections[idx].track_id = track_id
            self.tracks[track_id] = _Track(detections[idx].label, detections[idx].box, self.frame_no)
            used_detections.add(idx)
            used_tracks.add(track_id)
        for idx, detection in enumerate(detections):
            if idx not in used_detections:
                track_id = self.next_id
                self.next_id += 1
                detection.track_id = track_id
                self.tracks[track_id] = _Track(detection.label, detection.box, self.frame_no)
        return detections


@dataclass
class _Episode:
    hits: int = 0
    misses: int = 0
    emitted: bool = False
    observations: list[int] = field(default_factory=list)


class RuleEngine:
    """Confirms zone streaks and repeated SV observations, suppressing repeats.

    Event resets after `clear_after` consecutive frames without the same rule.
    Multiple people use different short-lived track IDs (no identity inference).
    """
    def __init__(self, confirm_frames: int = 3, clear_after: int = 5, min_confidence: float = 0.45,
                 safety_policy: SafetyPolicy | None = None):
        if confirm_frames < 1 or clear_after < 1:
            raise ValueError("confirm_frames and clear_after must be positive")
        self.confirm_frames = confirm_frames
        self.clear_after = clear_after
        self.min_confidence = min_confidence
        self.tracker = IoUTracker()
        self.episodes: dict[tuple[str, str, str, int], _Episode] = {}
        self.active_violations: list[Violation] = []
        self.safety_policy = safety_policy or SafetyPolicy()
        self.last_observed_at = None
        self.last_safety_observed_at = None

    def break_unconfirmed_continuity(self):
        """Dropped observations cannot confirm a streak or erase a saved incident.

        Keep emitted episode/track identity until actual clear observations or
        the wall-time gap expires. Otherwise storage backpressure re-emits the
        same person as a new incident every time the bounded queue overflows.
        """
        for episode in self.episodes.values():
            if not episode.emitted:
                episode.hits = episode.misses = 0
                episode.observations.clear()

    def process(self, camera_id: str, detections: list[Detection], zones: list[Zone], image_w: int, image_h: int,
                *, safetyvision_observed: bool = True, observed_at: float | None = None) -> tuple[list[Detection], list[Violation]]:
        if observed_at is not None:
            if self.last_observed_at is not None:
                if observed_at <= self.last_observed_at:
                    return detections, []  # Replays cannot count as new evidence.
                if observed_at - self.last_observed_at > self.safety_policy.max_observation_gap:
                    self.episodes.clear()
                    # Track IDs are also evidence-link keys. Expire continuity
                    # without reusing an earlier ID in this camera generation.
                    self.tracker.tracks.clear()
            self.last_observed_at = observed_at
            if (self.last_safety_observed_at is not None and
                    observed_at - self.last_safety_observed_at > self.safety_policy.max_observation_gap):
                self.episodes = {k: v for k, v in self.episodes.items() if k[2] == 'restricted_zone'}
            if safetyvision_observed:
                self.last_safety_observed_at = observed_at
        # Person and fall continuity are tracked independently. PPE inherits the
        # person's track rather than letting head jitter create a new incident.
        anchors = [d for d in detections if d.label == 'person' or (
            safetyvision_observed and d.label == 'fall_detected' and
            d.confidence >= self.safety_policy.thresholds['fall_detected'])]
        self.tracker.update(anchors)
        people = [d for d in anchors if d.label == 'person' and d.confidence >= self.min_confidence]
        evidence = safety_evidence(detections, people, self.safety_policy) if safetyvision_observed else []
        tracked = [*people, *evidence]
        candidates: dict[tuple[str, str, str, int], Violation] = {}
        for det in tracked:
            if det.track_id is None:
                continue
            if det.label == "person":
                # Person's foot point, not the center of the bbox.
                foot_x = ((det.box[0] + det.box[2]) / 2) / image_w
                foot_y = det.box[3] / image_h
                for zone in zones:
                    if inside_polygon(foot_x, foot_y, zone.points):
                        event = Violation(camera_id, zone.id, "restricted_zone", det.track_id, det.confidence, det.box)
                        candidates[(camera_id, zone.id, event.violation_type, det.track_id)] = event
            elif det.label == "no_helmet":
                # Only an explicit positive 'no_helmet' class is admissible.
                # A missing helmet detection cannot establish this violation.
                event = Violation(camera_id, "any", "no_helmet", det.track_id, det.confidence, det.box)
                candidates[(camera_id, "any", event.violation_type, det.track_id)] = event
            elif det.label == "fall_detected":
                # SV positively detected a possible fall; still require repeated observations.
                event = Violation(camera_id, "any", "fall_detected", det.track_id, det.confidence, det.box)
                candidates[(camera_id, "any", event.violation_type, det.track_id)] = event
        emitted = []
        for key, episode in list(self.episodes.items()):
            if key not in candidates:
                if key[2] in {'no_helmet', 'fall_detected'} and not safetyvision_observed:
                    # No SV reply is unknown, not a negative observation. Only
                    # actual SV observations count; wall-time gaps expire above.
                    continue
                episode.misses += 1
                if key[2] in {'no_helmet', 'fall_detected'}:
                    episode.observations.append(0)
                    episode.observations = episode.observations[-max(self.confirm_frames, self.safety_policy.confirm_window):]
                if episode.misses >= self.clear_after:
                    del self.episodes[key]
        for key, event in candidates.items():
            episode = self.episodes.setdefault(key, _Episode())
            if key[2] in {'no_helmet', 'fall_detected'}:
                episode.observations.append(1)
                episode.observations = episode.observations[-max(self.confirm_frames, self.safety_policy.confirm_window):]
                episode.hits = sum(episode.observations)
            else:
                episode.hits = episode.hits + 1 if episode.misses == 0 else 1
            episode.misses = 0
            if episode.hits >= self.confirm_frames and not episode.emitted:
                episode.emitted = True
                emitted.append(event)
        self.active_violations = [event for key, event in candidates.items()
                                  if self.episodes[key].emitted]
        return tracked, emitted

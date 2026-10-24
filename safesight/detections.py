"""JSON contract shared by test packets and real inference results."""
from datetime import datetime, timezone
import time


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def message(camera_id, packet, detections, source, task_id=None):
    height, width = packet.image.shape[:2]
    return {
        "type": "detections", "camera_id": camera_id, "source": source,
        "frame_id": packet.sequence, "generation": packet.generation,
        "frame_width": width, "frame_height": height, "coordinate_format": "xyxy",
        "captured_at": iso(packet.captured_wall), "updated_at": iso(time.time()),
        "result_age_ms": max(0, (time.perf_counter() - packet.captured_at) * 1000),
        "task_id": task_id,
        "detections": [{"label": d.label, "confidence": float(d.confidence),
                        "bbox": [int(x) for x in d.box], "track_id": d.track_id} for d in detections],
    }


def test_message(camera_id, index, width=640, height=480):
    """Explicit UI fixture, never fed into rule engine or event storage."""
    return {
        "type": "detections", "camera_id": camera_id, "source": "test-json",
        "frame_id": index, "generation": 0,
        "frame_width": width, "frame_height": height, "coordinate_format": "xyxy",
        "captured_at": iso(time.time()), "updated_at": iso(time.time()), "result_age_ms": 0,
        "task_id": None,
        "detections": [{"label": "TEST person", "confidence": .95,
                        "bbox": [int(width * .2), int(height * .15), int(width * .55), int(height * .85)],
                        "track_id": 1}],
    }

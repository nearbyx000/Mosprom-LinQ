"""Atomic JPEG evidence and durable event outbox, independent of video capture.

An accepted event is replayable after a database outage or process restart.
Retries concern SQLite only; inference tasks are never resent.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def atomic_write(path: Path, content: bytes):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class EvidenceRecorder:
    def __init__(self, store, evidence_dir, clock=time.monotonic):
        self.store = store
        self.evidence_dir = Path(evidence_dir)
        self.outbox = self.evidence_dir.parent/'event-outbox'
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.retry_at = 0.
        self.retry_delay = 1.
        self.last_error = None

    @property
    def pending(self):
        return sum(1 for _ in self.outbox.glob('*.json'))

    @property
    def invalid(self):
        return sum(1 for _ in (self.outbox/'invalid').glob('*.json'))

    def _persist(self, intent, intent_path):
        filename = intent['evidence_file']
        if Path(filename).name != filename or '/' in filename or '\\' in filename:
            raise ValueError('Invalid evidence filename in outbox')
        image = self.evidence_dir/filename
        if not image.is_file() or hashlib.sha256(image.read_bytes()).hexdigest() != intent['evidence_sha256']:
            raise ValueError('Evidence missing or checksum mismatch')
        event = intent['event']
        event_id = self.store.insert(event['camera_id'], event['zone_id'], event['violation_type'],
                                     event['track_id'], event['confidence'], filename,
                                     captured_at=intent['captured_at'], event_key=intent['event_key'],
                                     source=intent['source'], frame_id=intent['frame_id'],
                                     generation=intent['generation'], evidence_sha256=intent['evidence_sha256'])
        # If removal fails after commit, the unique event_key makes recovery
        # idempotent instead of inserting the same incident a second time.
        intent_path.unlink(missing_ok=True)
        return {'id':event_id, 'camera_id':event['camera_id'], 'type':event['violation_type'],
                'evidence_file':filename, 'persisted':True, 'event_key':intent['event_key']}

    def record(self, event, jpeg, packet, source):
        key = uuid.uuid4().hex
        camera = ''.join(c for c in event.camera_id if c.isalnum() or c in '_-')[:64]
        filename = f'{camera}-{key}-{event.violation_type}.jpg'
        intent = {'version':1, 'event_key':key, 'event':asdict(event), 'evidence_file':filename,
                  'evidence_sha256':hashlib.sha256(jpeg).hexdigest(), 'source':source,
                  'captured_at':packet.captured_wall, 'frame_id':packet.sequence, 'generation':packet.generation}
        atomic_write(self.evidence_dir/filename, jpeg)
        intent_path = self.outbox/f'{key}.json'
        atomic_write(intent_path, json.dumps(intent, ensure_ascii=False, allow_nan=False).encode('utf-8'))
        try:
            result = self._persist(intent, intent_path)
            self.last_error = None
            return result
        except Exception as exc:
            self.last_error = type(exc).__name__
            self.retry_at = self.clock() + self.retry_delay
            return {'id':None, 'camera_id':event.camera_id, 'type':event.violation_type,
                    'evidence_file':filename, 'persisted':False, 'event_key':key}

    def recover(self):
        if self.clock() < self.retry_at:
            return []
        saved = []
        for path in sorted(self.outbox.glob('*.json'))[:10]:
            try:
                if path.stat().st_size > 32768:
                    raise ValueError('Oversized outbox record')
                intent = json.loads(path.read_text(encoding='utf-8'))
                saved.append(self._persist(intent, path))
                self.last_error = None
                self.retry_delay = 1.
            except Exception as exc:
                self.last_error = type(exc).__name__
                if isinstance(exc,(ValueError,KeyError,TypeError)):
                    # Preserve invalid metadata for an operator, while allowing
                    # unrelated valid events to recover. Never delete a JPEG.
                    quarantine = self.outbox/'invalid'
                    quarantine.mkdir(exist_ok=True)
                    path.replace(quarantine/f'{path.stem}-{uuid.uuid4().hex[:8]}.json')
                    continue
                self.retry_at = self.clock() + self.retry_delay
                self.retry_delay = min(30., self.retry_delay*2)
                break
        return saved

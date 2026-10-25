import hashlib
import json
import sqlite3
from types import SimpleNamespace

import cv2
import numpy as np

from safesight.evidence import EvidenceRecorder
from safesight.logic import Violation
from safesight.storage import EventStore


def sample():
    ok,jpg = cv2.imencode('.jpg',np.zeros((30,40,3),np.uint8))
    assert ok
    return (Violation('cam','zone','restricted_zone',7,.9,(0,0,20,20)),jpg.tobytes(),
            SimpleNamespace(captured_wall=1792890000.,sequence=42,generation=3))


def test_failed_database_write_recovers_after_restart_with_exact_evidence(tmp_path):
    evidence = tmp_path/'evidence'
    evidence.mkdir()
    store = EventStore(tmp_path/'events.sqlite')
    original = store.insert
    store.insert = lambda *_a,**_kw: (_ for _ in ()).throw(sqlite3.OperationalError('database is locked'))
    recorder = EvidenceRecorder(store,evidence,clock=lambda:100.)
    event,jpeg,packet = sample()
    assert recorder.record(event,jpeg,packet,'linq')['persisted'] is False
    assert recorder.pending == 1
    intent = json.loads(next(recorder.outbox.glob('*.json')).read_text())
    assert (evidence/intent['evidence_file']).read_bytes() == jpeg
    assert store.list_events() == []
    store.insert = original
    # New recorder represents process restart; the pending intent lives on disk.
    restarted = EvidenceRecorder(store,evidence)
    assert len(restarted.recover()) == 1
    assert restarted.pending == 0
    assert restarted.recover() == []
    rows = store.list_events()
    assert len(rows) == 1
    assert rows[0]['frame_id'] == 42 and rows[0]['generation'] == 3
    assert rows[0]['source'] == 'linq'
    assert rows[0]['evidence_sha256'] == hashlib.sha256(jpeg).hexdigest()


def test_retry_after_database_commit_is_idempotent(tmp_path):
    evidence = tmp_path/'evidence'
    evidence.mkdir()
    store = EventStore(tmp_path/'events.sqlite')
    recorder = EvidenceRecorder(store,evidence,clock=lambda:100.)
    original = store.insert
    def committed_then_failed(*args,**kwargs):
        original(*args,**kwargs)
        raise OSError('Simulate failure after committed transaction')
    store.insert = committed_then_failed
    assert recorder.record(*sample(),'demo')['persisted'] is False
    assert len(store.list_events()) == 1
    store.insert = original
    restarted = EvidenceRecorder(store,evidence)
    assert len(restarted.recover()) == 1
    assert len(store.list_events()) == 1


def test_changed_evidence_is_not_inserted_as_valid_event(tmp_path):
    evidence = tmp_path/'evidence'
    evidence.mkdir()
    store = EventStore(tmp_path/'events.sqlite')
    original = store.insert
    store.insert = lambda *_a,**_kw: (_ for _ in ()).throw(sqlite3.OperationalError())
    recorder = EvidenceRecorder(store,evidence)
    recorder.record(*sample(),'demo')
    next(evidence.glob('*.jpg')).write_bytes(b'corrupted')
    store.insert = original
    restarted = EvidenceRecorder(store,evidence)
    assert restarted.recover() == []
    assert restarted.pending == 0 and restarted.invalid == 1
    assert restarted.last_error == 'ValueError'
    assert store.list_events() == []


def test_invalid_intent_cannot_starve_other_pending_events(tmp_path):
    evidence = tmp_path/'evidence'
    evidence.mkdir()
    store = EventStore(tmp_path/'events.sqlite')
    original = store.insert
    store.insert = lambda *_a,**_kw: (_ for _ in ()).throw(sqlite3.OperationalError())
    recorder = EvidenceRecorder(store,evidence)
    recorder.record(*sample(),'demo')
    (recorder.outbox/'000000-invalid.json').write_text('{broken json')
    store.insert = original
    restarted = EvidenceRecorder(store,evidence)
    assert len(restarted.recover()) == 1
    assert len(store.list_events()) == 1
    assert restarted.invalid == 1 and restarted.pending == 0


def test_legacy_database_migration_preserves_existing_rows(tmp_path):
    path = tmp_path/'events.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp TEXT NOT NULL,'
                   'camera_id TEXT NOT NULL,zone_id TEXT NOT NULL,violation_type TEXT NOT NULL,'
                   'track_id INTEGER NOT NULL,confidence REAL NOT NULL,evidence_file TEXT NOT NULL)')
        db.execute("INSERT INTO events VALUES(1,'2026-10-24T12:00:00+00:00','c','z','no_helmet',7,.9,'old.jpg')")
    store = EventStore(path)
    assert store.schema_backup is not None and store.schema_backup.is_file()
    with sqlite3.connect(store.schema_backup) as backup:
        assert 'event_key' not in {r[1] for r in backup.execute('PRAGMA table_info(events)')}
        assert backup.execute('SELECT evidence_file FROM events').fetchone()[0] == 'old.jpg'
    rows = store.list_events()
    assert len(rows) == 1 and rows[0]['evidence_file'] == 'old.jpg'
    assert rows[0]['source'] == 'legacy' and rows[0]['event_key'] is None
    EventStore(path)  # Migration must also be safe on the next application start.
    assert len(store.list_events()) == 1

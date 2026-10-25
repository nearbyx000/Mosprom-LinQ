"""SQLite-backed evidence and event log; one connection per operation."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone, timedelta
import uuid


class EventStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema_backup = None
        if self.path.is_file() and self.path.stat().st_size:
            with sqlite3.connect(self.path,timeout=2) as existing:
                old_columns = {row[1] for row in existing.execute('PRAGMA table_info(events)')}
                if old_columns and 'event_key' not in old_columns:
                    backup = self.path.with_name(f'{self.path.stem}.pre-v03-{uuid.uuid4().hex[:10]}.sqlite')
                    with sqlite3.connect(backup) as destination:
                        existing.backup(destination)
                    self.schema_backup = backup
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                zone_id TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                confidence REAL NOT NULL,
                evidence_file TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_events_when ON events(timestamp)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id)")
            columns = {row[1] for row in db.execute('PRAGMA table_info(events)')}
            # Additive migration preserves every legacy event and JPEG name.
            for name, definition in (
                ('event_key', 'TEXT'), ('source', "TEXT NOT NULL DEFAULT 'legacy'"),
                ('frame_id', 'INTEGER'), ('generation', 'INTEGER'), ('evidence_sha256', 'TEXT'),
            ):
                if name not in columns:
                    db.execute(f'ALTER TABLE events ADD COLUMN {name} {definition}')
            db.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_events_key ON events(event_key)')
            db.execute('''CREATE TABLE IF NOT EXISTS event_reviews(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL REFERENCES events(id),
                decision TEXT NOT NULL CHECK(decision IN ('confirmed','false_positive','needs_review')),
                note TEXT NOT NULL DEFAULT '', reviewer TEXT NOT NULL,
                reviewed_at TEXT NOT NULL)''')
            db.execute('CREATE INDEX IF NOT EXISTS idx_reviews_event ON event_reviews(event_id,id)')

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=2)
        try:
            with db:
                yield db
        finally:
            db.close()

    def insert(self, camera_id: str, zone_id: str, violation_type: str, track_id: int, confidence: float,
               evidence_file: str, captured_at: float | None = None, *, event_key: str | None = None,
               source: str = 'legacy', frame_id: int | None = None, generation: int | None = None,
               evidence_sha256: str | None = None) -> int:
        now = (datetime.fromtimestamp(captured_at, timezone.utc) if captured_at is not None else datetime.now(timezone.utc)).isoformat(timespec="seconds")
        with self._connect() as db:
            cursor = db.execute(
                """INSERT INTO events(timestamp,camera_id,zone_id,violation_type,track_id,confidence,evidence_file,
                   event_key,source,frame_id,generation,evidence_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(event_key) DO NOTHING""",
                (now, camera_id, zone_id, violation_type, track_id, confidence, evidence_file,
                 event_key, source, frame_id, generation, evidence_sha256),
            )
            if event_key is not None:
                return int(db.execute('SELECT id FROM events WHERE event_key=?', (event_key,)).fetchone()[0])
            return int(cursor.lastrowid)

    def list_events(self, camera_id: str | None = None, zone_id: str | None = None,
                    violation_type: str | None = None, start: str | None = None,
                    end: str | None = None, limit: int = 100) -> list[dict]:
        clauses, params = [], []
        for key, val in (("camera_id", camera_id), ("zone_id", zone_id), ("violation_type", violation_type)):
            if val:
                clauses.append(f"{key}=?")
                params.append(val)
        for op, val in ((">=", start), ("<=", end)):
            if val:
                # timestamps stored as UTC, normalize input before comparison.
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    raise ValueError("start/end require explicit timezone offset")
                clauses.append(f"timestamp{op}?")
                params.append(dt.astimezone(timezone.utc).isoformat(timespec="seconds"))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(min(max(limit, 1), 500))
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute('''SELECT events.*, COALESCE(review.decision,'unreviewed') AS review_status,
                review.note AS review_note, review.reviewed_at AS reviewed_at, review.reviewer AS reviewer
                FROM events LEFT JOIN event_reviews review ON review.id=(
                    SELECT MAX(id) FROM event_reviews WHERE event_id=events.id)'''
                + where + ' ORDER BY events.id DESC LIMIT ?',params).fetchall()
            return [dict(row) for row in rows]

    def review(self,event_id,decision,note='',reviewer='local_operator'):
        if decision not in {'confirmed','false_positive','needs_review'} or len(note)>1000:
            raise ValueError('Invalid review decision or note')
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            if db.execute('SELECT 1 FROM events WHERE id=?',(event_id,)).fetchone() is None:
                raise LookupError('Event not found')
            latest = db.execute('SELECT * FROM event_reviews WHERE event_id=? ORDER BY id DESC LIMIT 1',(event_id,)).fetchone()
            if latest is not None and latest['decision']==decision and latest['note']==note and latest['reviewer']==reviewer:
                return dict(latest)
            now = datetime.now(timezone.utc).isoformat(timespec='seconds')
            cursor = db.execute('INSERT INTO event_reviews(event_id,decision,note,reviewer,reviewed_at) VALUES(?,?,?,?,?)',
                                (event_id,decision,note,reviewer,now))
            return dict(db.execute('SELECT * FROM event_reviews WHERE id=?',(cursor.lastrowid,)).fetchone())

    def reviews(self,event_id):
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute('SELECT * FROM event_reviews WHERE event_id=? ORDER BY id',(event_id,)).fetchall()]

    def summary(self):
        since = (datetime.now(timezone.utc)-timedelta(hours=24)).isoformat(timespec='seconds')
        with self._connect() as db:
            rows = db.execute('SELECT violation_type,COUNT(*) FROM events WHERE timestamp>=? GROUP BY violation_type',
                              (since,)).fetchall()
            latest = db.execute('SELECT id,timestamp FROM events ORDER BY id DESC LIMIT 1').fetchone()
        return {'period':'last_24_hours','since':since,'total':sum(row[1] for row in rows),
                'by_type':dict(rows),'last_event_at':latest[1] if latest else None,
                'last_event_id':latest[0] if latest else None}

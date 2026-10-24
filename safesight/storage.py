"""SQLite-backed evidence and event log; one connection per operation."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone


class EventStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
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

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def insert(self, camera_id: str, zone_id: str, violation_type: str, track_id: int, confidence: float, evidence_file: str, captured_at: float | None = None) -> int:
        now = (datetime.fromtimestamp(captured_at, timezone.utc) if captured_at is not None else datetime.now(timezone.utc)).isoformat(timespec="seconds")
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO events(timestamp,camera_id,zone_id,violation_type,track_id,confidence,evidence_file) VALUES(?,?,?,?,?,?,?)",
                (now, camera_id, zone_id, violation_type, track_id, confidence, evidence_file),
            )
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
            rows = db.execute("SELECT * FROM events" + where + " ORDER BY id DESC LIMIT ?", params).fetchall()
            return [dict(row) for row in rows]

import tempfile
import unittest
from pathlib import Path
from safesight.storage import EventStore

class StorageTests(unittest.TestCase):
    def test_insert_and_filter(self):
        with tempfile.TemporaryDirectory() as path:
            db = EventStore(Path(path) / "events.sqlite")
            db.insert("cam", "a", "restricted_zone", 17, .94, "image.jpg")
            self.assertEqual(len(db.list_events()), 1)
            self.assertEqual(len(db.list_events(camera_id="other")), 0)
            self.assertEqual(db.list_events()[0]["evidence_file"], "image.jpg")

if __name__ == "__main__":
    unittest.main()

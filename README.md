# SafeSight | LinQ TPU industrial safety hackathon (starter MVP)

**Prototype only. Demo mode does NOT use neural inference or LinQ TPU.**

## Start (Linux, macOS, Windows PowerShell)

Requirements: Python 3.11 or 3.12. After cloning/unzipping:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn safesight.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ . The default synthetic source produces a demo green 'person' rectangle and a magenta explicit `no_helmet` indicator; the `DemoBackend` only color-segments these rectangles (no AI, no TPU). Incident snapshots persist under `data/evidence/` and the event log at `data/events.sqlite`.

### Real local inference (not eligible as LinQ TPU evidence)

```bash
python -m pip install -r requirements-local-ml.txt
# Linux/macOS:
export SAFESIGHT_MODE=local
export SAFESIGHT_MODEL=/absolute/path/to/your/detector.pt
# Windows PowerShell:
# $env:SAFESIGHT_MODE="local"
# $env:SAFESIGHT_MODEL="C:\\models\\detector.pt"
python -m uvicorn safesight.main:app --host 127.0.0.1 --port 8000
```

In `config/cameras.json`, replace `source: "synthetic"` with an actual local `.mp4` filepath or RTSP URL (avoid committing RTSP secrets). A COCO person detector can trigger `restricted_zone`, but **cannot** detect `no_helmet` unless the model was trained for that explicit class.

The dashboard reports backend and camera status. No mode may silently substitute local inference when `SAFESIGHT_MODE=linq` is selected. See `docs/LINQ.md`.

### Test

```bash
python -m unittest discover -s tests -v
```

### Components

- `safesight/inference.py`: clear inference abstraction: color-only demo, local Ultralytics, LinQ placeholder.
- `safesight/logic.py`: normalized zones, footpoint rule, explicit `no_helmet` class, IoU association and per-episode deduplication.
- `safesight/pipeline.py`: per-camera thread, frames, evidence and performance counters.
- `safesight/storage.py`: SQLite event log + indexed queries and filters.
- `safesight/main.py`: API and static web UI.
- `config/cameras.json`: camera source and normalized polygon ROI.
- `docs/TZ.md`: official requirement checklist.

### API

- `GET /api/status` — current backend identity, camera state, FPS, backend call latency.
- `GET /api/frame/cam-01` — annotated frame as JPEG.
- `GET /api/events?camera_id=cam-01&zone_id=restricted-a&violation_type=restricted_zone` — indexed SQLite log.
- `GET /evidence/{filename}` — persisted event screenshot.

Run uvicorn with **one worker**. RTSP disconnect detection exists but no automated reconnect yet. Do not serve this dev app to an untrusted network; there is no authentication or authorization. Do not commit VPN config or credentials.

## Next required gate

1. Run the organizer's ResNet50 TPU example via VPN and Jupyter.
2. Download SDK PDF and Python API example from organizer links.
3. Choose a compatible detection model, quantize+compile ONNX with the official SDK.
4. Implement proper adapter using the *actual API example*, verify postprocessing and real TPU FPS.
5. Test on official incident videos and adjust rule/thresholds. Never claim this has been done until verified.

See `docs/LINQ.md` and `docs/TZ.md` for the exact roadmap and risks.

### Benchmark (not proof of TPU usage)

```bash
python scripts/benchmark.py --mode demo --frames 100
# after providing an actual local model:
# python scripts/benchmark.py --mode local --model /path/to/model.pt --source /path/to/video.mp4 --frames 100
```

Benchmarks print median, p95, mean latency and throughput. Hardware provenance must be independently confirmed when the real LinQ adapter is implemented.

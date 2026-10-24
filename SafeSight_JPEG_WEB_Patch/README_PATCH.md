# SafeSight WEB + LinQ JPEG Gateway — integration patch

Applies to https://github.com/nearbyx000/Mosprom-LinQ branch `WEB`, read on 2026-10-24. No TPU recompilation.

A GitHub App write attempt returned HTTP 403. Therefore **nothing was pushed** to GitHub. Integrate locally on a feature branch and send your teammate a PR.

## On Windows / PowerShell

1. Ensure `Mosprom-LinQ` is cloned and synchronized. Make sure your own modifications are committed/stashed; don't overwrite teammates' work.
2. From repository directory:

```powershell
git fetch origin WEB
git switch -c feature/linq-jpeg-web origin/WEB
```

3. Unzip `SafeSight_JPEG_WEB_Patch.zip` into any location (e.g. Downloads). Then use:

```powershell
py -3.12 SafeSight_JPEG_WEB_Patch\apply_integration.py .
py -3.12 -m pip install -r requirements-dev.txt
py -3.12 -m pytest tests/test_jpeg_gateway.py -q
```

If the patch is unpacked in `Downloads`, provide its full Windows path to `apply_integration.py`, keeping trailing `.` as argument.

4. Keep existing LinQ cloud Gateway running in Jupyter Terminal/Notebook; record the `GATEWAY READY | route: safesight.jpeg....` string. Do not commit live route/secrets.
5. In the WEB server's PowerShell:

```powershell
$env:LINQ_JPEG_ROUTE='safesight.jpeg.REPLACE_WITH_CURRENT_GATEWAY_ROUTE'
py -3.12 -m pip install -r requirements-linq.txt
py -3.12 run.py --mode linq --source 0
```

Open `http://127.0.0.1:8000/` full dashboard. Switch `Detection source` away from test JSON to real results if needed.

## WSL equivalent

```bash
git fetch origin WEB
git switch -c feature/linq-jpeg-web origin/WEB
python /path/to/SafeSight_JPEG_WEB_Patch/apply_integration.py .
python -m pip install -r requirements-linq.txt
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_jpeg_gateway.py -q
export LINQ_JPEG_ROUTE='safesight.jpeg.REPLACE_WITH_CURRENT_GATEWAY_ROUTE'
python run.py --mode linq --source 0
```

Use `--source synthetic` only to verify UI pipeline; it cannot establish real persons in an arbitrary test scene. For live human detections, use an actual camera or a video file.

## What changes

- New `safesight/jpeg_gateway.py` implements the existing asynchronous client contract (`connect/submit/poll/status/close`) and sends JPEG to a remote route, accepting JSON with original frame coordinates.
- `safesight/pipeline.py` selects this adapter when `LINQ_JPEG_ROUTE` is configured and `mode=linq`. Otherwise retains the original direct Float32 LinQ transport.
- The current WebSocket overlays, live preview, EventStore/RuleEngine and status do not change.
- `tests/test_jpeg_gateway.py` uses mocks (no GPU/TPU required).

## Important

The remote gateway must be running on a permitted machine **inside the LinQ network**; running it on your WSL host eliminates the main network benefit. The gateway uses a private ephemeral route, not the official `input` TPU route for JPEG. If it restarts, the route changes; update `LINQ_JPEG_ROUTE` and restart local WEB. Existing LIVE camera preview remains separate from inference.

Do not equate 30 preview FPS with 30 TPU detector FPS. As of Oct 24, 10 JPEG test requests all found 4 people with RTT 324–675 ms, median 488 ms, from WSL to gateway.

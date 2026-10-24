#!/usr/bin/env python3
"""Apply additive JPEG Gateway integration on the exact team's WEB code layout.

Run from a local clone on a NEW branch: python path/to/apply_integration.py .
Never runs Git commands, never changes network settings or remote branches.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

PATCH = Path(__file__).resolve().parent


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
    pipeline = root / 'safesight/pipeline.py'
    if not pipeline.is_file() or not (root / 'safesight/main.py').is_file():
        raise SystemExit('ERROR: run from the root of Mosprom-LinQ WEB checkout')
    old = pipeline.read_text(encoding='utf-8')
    # Strict anchors prevent silently patching a teammate's changed implementation.
    replacements = [
        ('from .rabbitmq import LinQClient, Settings\n',
         'from .rabbitmq import LinQClient, Settings\nfrom .jpeg_gateway import JPEGGatewayClient, JPEGSettings\n'),
        ('        self.client_factory = client_factory or (lambda metrics: LinQClient(Settings.from_env(), metrics))\n',
         '        if client_factory is not None:\n'
         '            self.client_factory = client_factory\n'
         '        elif os.getenv("LINQ_JPEG_ROUTE") and mode == "linq":\n'
         '            self.client_factory = lambda metrics: JPEGGatewayClient(JPEGSettings.from_env(), metrics)\n'
         '        else:\n'
         '            self.client_factory = lambda metrics: LinQClient(Settings.from_env(), metrics)\n'),
    ]
    for needle, value in replacements:
        if old.count(needle) != 1:
            raise SystemExit(f'ERROR: WEB pipeline has changed; expected exactly one anchor: {needle.strip()!r}. No files modified.')
        old = old.replace(needle, value)
    dest_module = root / 'safesight/jpeg_gateway.py'
    dest_test = root / 'tests/test_jpeg_gateway.py'
    if dest_module.exists() or dest_test.exists():
        raise SystemExit('ERROR: integration files already exist. No files modified (patch possibly already applied).')
    # Make targeted edits only; do not overwrite any existing frontend JS/routes.
    additions = {
        root / '.env.example': '''\n# JPEG Gateway transport (set route printed by remote GATEWAY READY).\n# When LINQ_JPEG_ROUTE is nonempty and mode=linq, tensors stay in the LinQ cloud.\n# LINQ_JPEG_ROUTE=safesight.jpeg.REPLACE_WITH_CURRENT_GATEWAY_ROUTE\n# LINQ_JPEG_QUALITY=75\n''',
        root / 'README.md': '''\n\n## JPEG Gateway: actual LinQ TPU detections in WEB\n\nStart `SafeSight_JPEG_Gateway/gateway.py` inside the LinQ cloud environment,\nwith existing `/auto/cache/<LOGIN>/yolov8n.tpu`. Keep that process running.\nCopy its unique `GATEWAY READY` route. This gateway uses real LinQ TPU\n(`PyTask` inside the cloud), **not** local or synthetic model inference.\n\nIn the **local WEB server** environment, install dependencies:\n\n```bash\npython -m pip install -r requirements-linq.txt\n```\n\nSet `LINQ_JPEG_ROUTE` to the route from the remote process. Start as usual\nwith `--mode linq`. For example on WSL/Linux:\n\n```bash\nexport LINQ_JPEG_ROUTE=safesight.jpeg.REPLACE_WITH_CURRENT_GATEWAY_ROUTE\npython run.py --mode linq --source 0\n```\n\nOr set `LINQ_JPEG_ROUTE=...` in a local `.env` (do not commit it), and run\n`python run.py --mode linq --source 0 --env-file .env`. Open `/` (full\nDashboard), not `/camera` (minimal preview). If the remote gateway is stopped,\nthe UI will report no fresh detection; clean video continues.\n\nThe existing `linq` transport remains available by unsetting\n`LINQ_JPEG_ROUTE` and configuring `LINQ_LOGIN`/`LINQ_PROGRAM_URI` and\n`tpu_pycloud_format`. JPEG transport needs only `pika` locally. Preview\nis asynchronous and independent of cloud RTT; expect ~2 detection updates/s\nunder previously measured ~488 ms sequential median, **not** 30 TPU FPS.\nTest with `python -m pytest tests/test_jpeg_gateway.py -q` (fake replies).\n''',
    }
    for path in additions:
        if not path.is_file():
            raise SystemExit(f'ERROR: expected WEB file not found: {path}; no files modified')
    pipeline.write_text(old, encoding='utf-8')
    shutil.copyfile(PATCH / 'safesight/jpeg_gateway.py', dest_module)
    shutil.copyfile(PATCH / 'tests/test_jpeg_gateway.py', dest_test)
    for path, section in additions.items():
        with path.open('a', encoding='utf-8') as f:
            f.write(section)
    print('OK: JPEG Gateway adapter added to WEB')
    print('Updated:', pipeline.relative_to(root), '.env.example', 'README.md')
    print('Created:', dest_module.relative_to(root), dest_test.relative_to(root))
    print('No frontend files, database, or Git branches were changed.')


if __name__ == '__main__':
    main()

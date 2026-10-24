"""Cross-platform development launcher: python run.py."""

import argparse
import os
from pathlib import Path

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the SafeSight camera web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Reload after source changes")
    parser.add_argument("--mode", choices=("preview", "linq", "demo", "local"), help="Inference mode; default: SAFESIGHT_MODE or preview")
    parser.add_argument("--source", help="Override camera index, network URL, or video file")
    parser.add_argument('--data-dir', help='Separate event/evidence directory (demo defaults to data/demo)')
    parser.add_argument('--config', help='Camera and zone JSON configuration')
    parser.add_argument("--env-file", help="Load environment file (requires python-dotenv from requirements-linq.txt)")
    args = parser.parse_args()
    if args.env_file and not Path(args.env_file).is_file():
        parser.error('Environment file does not exist')
    if args.mode:
        os.environ["SAFESIGHT_MODE"] = args.mode
    if args.source is not None:
        os.environ["CAMERA_SOURCE"] = args.source
    if args.data_dir:
        os.environ['SAFESIGHT_DATA'] = args.data_dir
    if args.config:
        os.environ['SAFESIGHT_CONFIG'] = args.config
    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload, env_file=args.env_file)


if __name__ == "__main__":
    main()

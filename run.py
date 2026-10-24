"""Cross-platform development launcher: python run.py."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the SafeSight camera web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Reload after source changes")
    args = parser.parse_args()
    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

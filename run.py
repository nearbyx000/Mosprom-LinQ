"""Cross-platform development launcher: python run.py."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the SafeSight camera web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Reload after source changes")
    parser.add_argument("--ssl-certfile", help="HTTPS certificate trusted by the phone")
    parser.add_argument("--ssl-keyfile", help="Private key for the HTTPS certificate")
    args = parser.parse_args()
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("Provide both --ssl-certfile and --ssl-keyfile")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload,
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile,
                ws_max_size=1024 * 1024)


if __name__ == "__main__":
    main()

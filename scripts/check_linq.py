"""Small explicit real-cloud probe; never replaces LinQ with local inference."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import cv2

from safesight.rabbitmq import LinQClient, Settings
from safesight.runtime import Frame, Metrics


def main():
    parser = argparse.ArgumentParser(description="Send a fixed number of real YOLO requests to LinQ through VPN")
    parser.add_argument("--image", required=True)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--output", default="data/linq-probe.json")
    parser.add_argument("--env-file", help="Load local settings, e.g. .env")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)
    if not 1 <= args.requests <= 100:
        parser.error("requests must be 1..100; start with one on the shared server")
    image = cv2.imread(args.image)
    if image is None:
        parser.error("Image could not be read")
    metrics = Metrics()
    settings = Settings.from_env()
    # A fixed test image does not become stale; live preview retains its own
    # stricter age bound. A slow model load should still be visible in a probe.
    client = LinQClient(replace(settings, max_result_age=settings.timeout + 1), metrics)
    sent, received, results = 0, 0, []
    started = time.perf_counter()
    try:
        client.connect()
        while received < args.requests:
            for frame, detections, task_id in client.poll():
                received += 1
                print(f"response {received}/{args.requests}, task={task_id}, people={len(detections)}")
                results.append({"task_id": task_id, "people": [{"confidence": d.confidence, "box": d.box} for d in detections]})
            state = client.status()
            counts = metrics.snapshot()["counts"]
            if state["paused"] or any(counts.get(key, 0) for key in ("responses_invalid", "responses_stale", "requests_unroutable")):
                raise RuntimeError(state["error"] or "Response rejected; see counts in the saved report")
            if sent < args.requests and client.can_submit():
                sent += 1
                now = time.perf_counter()
                client.submit(Frame(sent, now, time.time(), image))
            time.sleep(0.005)
    finally:
        report = {"mode": "linq", "real_responses_decoded": received,
                  "elapsed_s": round(time.perf_counter() - started, 3), "metrics": metrics.snapshot(),
                  "results": results, "tpu_ms": None,
                  "note": "RTT includes network + queue + TPU. No separate TPU timing is supplied by the notebook protocol."}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        client.close()
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()

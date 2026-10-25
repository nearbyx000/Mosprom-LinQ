#!/usr/bin/env python3
"""Resolution-parametrized WSL -> RabbitMQ -> remote LinQ TPU smoke test.

Adapted from the official organizer RabbitMQ sample and the previously
validated YOLOv8n Jupyter notebook. Does not access local pytpu.Device.
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def preprocess(image: Image.Image, side: int = 640):
    """YOLOv8 letterbox: RGB NCHW float32, values in [0,1]."""
    image = image.convert("RGB")
    width, height = image.size
    ratio = min(side / width, side / height)
    new_width = round(width * ratio)
    new_height = round(height * ratio)
    left = round((side - new_width) / 2 - 0.1)
    top = round((side - new_height) / 2 - 0.1)
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (side, side), (114, 114, 114))
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32)
    tensor = np.ascontiguousarray(array.transpose(2, 0, 1)[None] / 255.0)
    return tensor, (ratio, left, top)


def remote_infer(tensor, host: str, port: int, program_path: str, timeout: float):
    import pika
    from tpu_pycloud_format import (
        MESSAGE_ID_ATTR_NAME,
        RESPONSE_ROUTING_KEYS_ATTR_NAME,
        TARGET_URI_ATTR_NAME,
        PyTask,
        deserialize,
        serialize,
    )

    task_id = uuid.uuid4().hex
    reply_queue = f"notebook_{task_id[:16]}"
    credentials = pika.PlainCredentials(os.environ.get("LINQ_MQ_USER", "guest"),
                                        os.environ.get("LINQ_MQ_PASSWORD", "guest"))
    params = pika.ConnectionParameters(host=host, port=port, credentials=credentials,
                                        socket_timeout=10, blocked_connection_timeout=timeout)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    try:
        channel.queue_declare(queue=reply_queue, auto_delete=True)
        channel.queue_bind(exchange="tpu.direct", queue=reply_queue, routing_key=reply_queue)
        headers = {
            TARGET_URI_ATTR_NAME: f"file://{program_path}?raw=False",
            RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([reply_queue]),
            MESSAGE_ID_ATTR_NAME: task_id,
        }
        start = time.monotonic()
        channel.basic_publish(
            exchange="tpu.direct", routing_key="input",
            body=serialize(PyTask({"images": tensor})),
            properties=pika.BasicProperties(headers=headers),
        )
        print(f"Task sent: {task_id}", flush=True)
        response = None
        for method, props, body in channel.consume(reply_queue, auto_ack=True, inactivity_timeout=1):
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"No reply in {timeout} seconds")
            if method is None:
                continue
            if not props or not props.headers or props.headers.get(MESSAGE_ID_ATTR_NAME) != task_id:
                print("Ignoring unmatched response", flush=True)
                continue
            response = body
            break
        channel.cancel()
        if response is None:
            raise RuntimeError("No matching TPU response received")
        rtt_ms = (time.monotonic() - start) * 1000
        data, metadata, worker = deserialize(response)
        if worker.get("status_code") != 0 or not data:
            raise RuntimeError(f"TPU worker error: {worker}")
        return data, worker, rtt_ms
    finally:
        if connection.is_open:
            connection.close()


def find_people(raw: np.ndarray, width: int, height: int, meta, confidence: float):
    """YOLOv8 COCO: channel 4 contains 'person' class confidence."""
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 84:
        arr = arr[0].T
    elif arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 84:
        arr = arr[0]
    else:
        raise ValueError(f"Unexpected model output: {arr.shape}; expected [1,84,N] or [1,N,84]")
    candidates = arr[np.isfinite(arr[:, 4]) & (arr[:, 4] >= confidence)]
    if not len(candidates):
        return [], float(np.max(arr[:, 4]))
    xywh = candidates[:, :4]
    boxes = np.column_stack([xywh[:, 0] - xywh[:, 2] / 2,
                             xywh[:, 1] - xywh[:, 3] / 2,
                             xywh[:, 0] + xywh[:, 2] / 2,
                             xywh[:, 1] + xywh[:, 3] / 2])
    scores = candidates[:, 4]
    order = np.argsort(scores)[::-1]
    kept = []
    while len(order) and len(kept) < 100:
        i = int(order[0])
        kept.append(i)
        other = order[1:]
        if not len(other):
            break
        inter_w = np.maximum(0, np.minimum(boxes[i, 2], boxes[other, 2]) - np.maximum(boxes[i, 0], boxes[other, 0]))
        inter_h = np.maximum(0, np.minimum(boxes[i, 3], boxes[other, 3]) - np.maximum(boxes[i, 1], boxes[other, 1]))
        inter = inter_w * inter_h
        area_i = max(0, boxes[i, 2] - boxes[i, 0]) * max(0, boxes[i, 3] - boxes[i, 1])
        area_other = np.maximum(0, boxes[other, 2] - boxes[other, 0]) * np.maximum(0, boxes[other, 3] - boxes[other, 1])
        iou = inter / np.maximum(area_i + area_other - inter, 1e-9)
        order = other[iou <= 0.45]
    ratio, left, top = meta
    result = []
    for i in kept:
        x1, y1, x2, y2 = boxes[i]
        coords = [int(round(np.clip((x - left) / ratio, 0, width))) for x in (x1, x2)]
        ys = [int(round(np.clip((y - top) / ratio, 0, height))) for y in (y1, y2)]
        bbox = (coords[0], ys[0], coords[1], ys[1])
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            result.append((float(scores[i]), bbox))
    return result, float(np.max(arr[:, 4]))


def main():
    parser = argparse.ArgumentParser(description="Single-image LinQ TPU cloud smoke test from WSL")
    parser.add_argument("--image", required=True, help="local JPG image path")
    parser.add_argument("--program", default="/auto/cache/hackathon16045/yolov8n.tpu",
                        help="EXISTING program path on REMOTE LinQ cloud server")
    parser.add_argument("--host", default="10.0.114.30")
    parser.add_argument("--port", type=int, default=5672)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--side", type=int, choices=(320, 416, 640), default=640, help="Match this to the compiled TPU model input size")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--out", default="person_wsl.jpg")
    parser.add_argument("--raw", default="output_wsl.npz")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    tensor, meta = preprocess(image, side=args.side)
    print(f"Input: images shape={tensor.shape} dtype={tensor.dtype}", flush=True)
    print(f"Remote program path: {args.program}", flush=True)
    data, worker, rtt_ms = remote_infer(tensor, args.host, args.port, args.program, args.timeout)
    print(f"RTT: {rtt_ms:.1f} ms")
    print("Worker status:", worker.get("status_code"), "TPU time:", worker.get("inference_time"), "s")
    print("Output keys:", list(data))
    for k, v in data.items():
        print(f"  {k}: {np.asarray(v).shape} {np.asarray(v).dtype}")
    np.savez_compressed(args.raw, **data)
    output = data.get("output0")
    if output is None:
        if len(data) != 1:
            raise RuntimeError("Can't identify YOLO tensor among outputs")
        output = next(iter(data.values()))
    people, max_conf = find_people(output, *image.size, meta, args.confidence)
    print(f"Person max confidence: {max_conf:.4f}")
    print(f"Detected people: {len(people)}")
    draw = ImageDraw.Draw(image)
    for conf, bbox in people:
        print(f"  person {conf:.3f} bbox={bbox}")
        draw.rectangle(bbox, outline=(0, 255, 0), width=3)
        draw.text((bbox[0], max(0, bbox[1] - 14)), f"Person {conf:.2f}", fill=(0, 255, 0))
    image.save(args.out)
    print(f"Saved: {args.out} and {args.raw}")


if __name__ == "__main__":
    main()

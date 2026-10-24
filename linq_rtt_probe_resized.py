#!/usr/bin/env python3
"""Measure WSL->RabbitMQ->LinQ TPU round-trip without recreating the connection.

Runs sequential probes by default to avoid loading a shared hackathon broker.
Use only on the authorized organizers' infrastructure.
"""
import argparse
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from linq_wsl_smoke_resized import preprocess


def milliseconds(a, b):
    return round((b - a) * 1000.0, 1)


def percentile(values, p):
    a = sorted(values)
    if not a:
        return float('nan')
    idx = (len(a) - 1) * p / 100.0
    low = int(idx)
    high = min(low + 1, len(a) - 1)
    weight = idx - low
    return round(a[low] * (1 - weight) + a[high] * weight, 1)


def main():
    p = argparse.ArgumentParser(description='Measure end-to-end LinQ cloud RTT by stage')
    p.add_argument('--image', default='calib/t3.jpg')
    p.add_argument('--count', type=int, default=5)
    p.add_argument('--timeout', type=float, default=40)
    p.add_argument('--side', type=int, choices=(320, 416, 640), default=640, help='Match compiled model input size')
    p.add_argument('--host', default='10.0.114.30')
    p.add_argument('--port', type=int, default=5672)
    p.add_argument('--program', default='/auto/cache/hackathon16045/yolov8n.tpu')
    args = p.parse_args()
    if not (1 <= args.count <= 20):
        p.error('count must be 1..20')
    if not Path(args.image).is_file():
        p.error(f'Image not found: {args.image}')

    import pika
    from tpu_pycloud_format import (
        MESSAGE_ID_ATTR_NAME,
        RESPONSE_ROUTING_KEYS_ATTR_NAME,
        TARGET_URI_ATTR_NAME,
        PyTask,
        deserialize,
        serialize,
    )

    t0 = time.perf_counter()
    image = Image.open(args.image).convert('RGB')
    tensor, _ = preprocess(image, side=args.side)
    t1 = time.perf_counter()
    print('Preprocess:', milliseconds(t0, t1), 'ms', 'tensor bytes:', tensor.nbytes, flush=True)

    credentials = pika.PlainCredentials(
        os.environ.get('LINQ_MQ_USER', 'guest'),
        os.environ.get('LINQ_MQ_PASSWORD', 'guest'),
    )
    params = pika.ConnectionParameters(
        host=args.host,
        port=args.port,
        credentials=credentials,
        socket_timeout=15,
        blocked_connection_timeout=args.timeout,
        heartbeat=120,
    )
    conn_t0 = time.perf_counter()
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    reply_queue = 'bench_' + uuid.uuid4().hex[:16]
    channel.queue_declare(queue=reply_queue, auto_delete=True)
    channel.queue_bind(exchange='tpu.direct', queue=reply_queue, routing_key=reply_queue)
    conn_t1 = time.perf_counter()
    print('Connect + queue setup:', milliseconds(conn_t0, conn_t1), 'ms', flush=True)
    latencies = []
    worker_times = []
    try:
        # One server consumer and one connection for all requests.
        consumer = channel.consume(reply_queue, auto_ack=True, inactivity_timeout=0.25)
        for n in range(1, args.count + 1):
            task_id = uuid.uuid4().hex
            headers = {
                TARGET_URI_ATTR_NAME: f'file://{args.program}?raw=False',
                RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([reply_queue]),
                MESSAGE_ID_ATTR_NAME: task_id,
            }
            a = time.perf_counter()
            payload = serialize(PyTask({'images': tensor}))
            b = time.perf_counter()
            channel.basic_publish(
                exchange='tpu.direct',
                routing_key='input',
                body=payload,
                properties=pika.BasicProperties(headers=headers),
            )
            c = time.perf_counter()
            deadline = c + args.timeout
            while True:
                if time.perf_counter() >= deadline:
                    raise TimeoutError(f'Task {n}: no response in {args.timeout}s')
                method, props, body = next(consumer)
                if method is None:
                    continue
                if not props or not props.headers or props.headers.get(MESSAGE_ID_ATTR_NAME) != task_id:
                    print('WARNING: ignoring unmatched response', flush=True)
                    continue
                d = time.perf_counter()
                data, metadata, worker = deserialize(body)
                e = time.perf_counter()
                if worker.get('status_code') != 0 or not data:
                    raise RuntimeError(f'Task {n}: worker failed: {worker}')
                response_size = len(body)
                break
            rtt = milliseconds(b, d)
            wt = 1000.0 * float(worker.get('inference_time', 0.0))
            latencies.append(rtt)
            worker_times.append(wt)
            print(
                f'#{n}: serialize={milliseconds(a,b)} ms | '
                f'publish={milliseconds(b,c)} ms | '
                f'wait={milliseconds(c,d)} ms | '
                f'decode={milliseconds(d,e)} ms | '
                f'RTT={rtt} ms | TPU={wt:.2f} ms | '
                f'wire request={len(payload)} B response={response_size} B',
                flush=True,
            )
        print('SUMMARY:', 'n=', len(latencies),
              'RTT p50=', percentile(latencies, 50),
              'p95=', percentile(latencies, 95),
              'min=', round(min(latencies), 1),
              'max=', round(max(latencies), 1),
              'TPU mean=', round(statistics.mean(worker_times), 2),
              'ms', flush=True)
    finally:
        if connection.is_open:
            try:
                channel.cancel()
            except Exception:
                pass
            connection.close()


if __name__ == '__main__':
    main()

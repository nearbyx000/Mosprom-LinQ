#!/usr/bin/env python3
"""Run on LOCAL Windows/WSL. Send JPEG over AMQP and receive compact JSON.

No tpu_pycloud_format required here. Can save person boxes on source image.
"""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import statistics
import time
import uuid

import pika
from PIL import Image, ImageDraw

EXCHANGE = 'tpu.direct'


def connect(host, port, timeout):
    return pika.BlockingConnection(pika.ConnectionParameters(
        host=host, port=port,
        credentials=pika.PlainCredentials(os.getenv('LINQ_MQ_USER', 'guest'),
                                          os.getenv('LINQ_MQ_PASSWORD', 'guest')),
        socket_timeout=10, blocked_connection_timeout=timeout, heartbeat=120,
    ))


def request(channel, consumer, queue, route, jpeg, timeout, frame_id=None):
    task_id = frame_id or uuid.uuid4().hex
    t0 = time.perf_counter()
    channel.basic_publish(exchange=EXCHANGE, routing_key=route, body=jpeg, mandatory=True,
                          properties=pika.BasicProperties(content_type='image/jpeg',
                                                          correlation_id=task_id,
                                                          reply_to=queue))
    publish_ms = (time.perf_counter() - t0) * 1000
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        method, properties, payload = next(consumer)
        if method is None:
            continue
        if properties and properties.correlation_id == task_id:
            result = json.loads(payload.decode('utf-8'))
            result['stats_client'] = {
                'rabbitmq_round_trip_ms': round((time.perf_counter() - t0) * 1000, 2),
                'publish_ms': round(publish_ms, 2),
                'jpeg_sent_bytes': len(jpeg),
                'json_received_bytes': len(payload),
            }
            return result
    raise TimeoutError(f'No JPEG gateway reply within {timeout}s; check remote process and route')


def save_overlay(image, result, filename):
    canvas = image.copy().convert('RGB')
    draw = ImageDraw.Draw(canvas)
    for det in result.get('detections', []):
        bbox = tuple(det['bbox'])
        draw.rectangle(bbox, outline=(0, 255, 0), width=3)
        draw.text((bbox[0], max(0, bbox[1] - 16)), f"{det['class_name']} {det['confidence']:.2f}", fill=(0, 255, 0))
    canvas.save(filename)


def main():
    p = argparse.ArgumentParser(description='Local JPEG -> cloud gateway -> LinQ -> JSON')
    p.add_argument('--image', required=True)
    p.add_argument('--route', required=True, help='Copy GATEWAY READY route from the REMOTE terminal')
    p.add_argument('--host', default='10.0.114.30')
    p.add_argument('--port', type=int, default=5672)
    p.add_argument('--count', type=int, default=1, help='Sequential calls (1..10)')
    p.add_argument('--quality', type=int, default=75)
    p.add_argument('--timeout', type=float, default=40)
    p.add_argument('--out', default='person_gateway.jpg')
    args = p.parse_args()
    if not 1 <= args.count <= 10:
        p.error('count must be 1..10')
    if not 30 <= args.quality <= 95:
        p.error('quality must be 30..95')
    if not Path(args.image).is_file():
        p.error(f'Missing input image: {args.image}')
    image = Image.open(args.image).convert('RGB')
    buffer = BytesIO()
    encode_start = time.perf_counter()
    image.save(buffer, format='JPEG', quality=args.quality)
    jpeg = buffer.getvalue()
    print(f'Image {image.width}x{image.height} | JPEG {len(jpeg)} B | '
          f'encode {(time.perf_counter()-encode_start)*1000:.1f}ms', flush=True)
    connection = connect(args.host, args.port, args.timeout)
    channel = connection.channel()
    response = channel.queue_declare(queue='', exclusive=True, auto_delete=True)
    response_queue = response.method.queue
    channel.queue_bind(exchange=EXCHANGE, queue=response_queue, routing_key=response_queue)
    # Mandatory publishing signals if gateway queue isn't bound to this route.
    channel.confirm_delivery()
    consumer = channel.consume(response_queue, auto_ack=True, inactivity_timeout=0.25)
    timings = []
    try:
        for n in range(args.count):
            result = request(channel, consumer, response_queue, args.route, jpeg, args.timeout)
            if not result.get('ok'):
                raise RuntimeError(f'Gateway failed: {result.get("error")}')
            metrics = result['stats_client']
            timings.append(metrics['rabbitmq_round_trip_ms'])
            print(f'#{n+1} | RTT {metrics["rabbitmq_round_trip_ms"]}ms | '
                  f'publish {metrics["publish_ms"]}ms | '
                  f'gateway {result["stats"]["gateway_total_ms"]}ms | '
                  f'TPU {result["stats"]["tpu_worker_ms"]}ms | '
                  f'people {len(result["detections"])} | '
                  f'wire JPEG {metrics["jpeg_sent_bytes"]}B JSON {metrics["json_received_bytes"]}B', flush=True)
            for d in result['detections']:
                print(f'  {d["class_name"]}: {d["confidence"]:.3f} {d["bbox"]}')
            if n == 0:
                save_overlay(image, result, args.out)
                Path('detections_gateway.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f'RTT min={min(timings):.1f}ms median={statistics.median(timings):.1f}ms '
              f'max={max(timings):.1f}ms', flush=True)
        print(f'Saved {args.out} and detections_gateway.json')
    finally:
        if connection.is_open:
            try:
                channel.cancel()
            except Exception:
                pass
            connection.close()


if __name__ == '__main__':
    main()

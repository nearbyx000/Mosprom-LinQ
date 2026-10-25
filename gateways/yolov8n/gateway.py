#!/usr/bin/env python3
"""Run NEAR the TPU broker, not on the user's WSL machine.

Receive JPEG from a private AMQP route, invoke official PyTask TPU inference,
return only compact JSON. The existing 640x640 TPU program is reused.
"""
import argparse
import json
import os
import secrets
import time
import uuid

import pika
import importlib
if __package__:
    from .preprocessing import jpeg_to_tensor, person_detections
else:
    from preprocessing import jpeg_to_tensor, person_detections

EXCHANGE = 'tpu.direct'
TPU_INPUT_ROUTE = 'input'


def ms(start):
    return round((time.perf_counter() - start) * 1000.0, 2)


def connect(host, port, timeout):
    return pika.BlockingConnection(pika.ConnectionParameters(
        host=host, port=port,
        credentials=pika.PlainCredentials(os.getenv('LINQ_MQ_USER', 'guest'),
                                          os.getenv('LINQ_MQ_PASSWORD', 'guest')),
        socket_timeout=10, blocked_connection_timeout=timeout, heartbeat=120,
    ))


class TPUClient:
    """Persistent private AMQP connection + reply consumer for TPU calls."""
    def __init__(self, host, port, program, timeout):
        self.codec = importlib.import_module("tpu_pycloud_format")
        self.program, self.timeout = program, timeout
        self.connection = connect(host, port, timeout)
        self.channel = self.connection.channel()
        q = self.channel.queue_declare(queue='', exclusive=True, auto_delete=True)
        self.reply_queue = q.method.queue
        self.channel.queue_bind(exchange=EXCHANGE, queue=self.reply_queue, routing_key=self.reply_queue)
        self.consumer = self.channel.consume(self.reply_queue, auto_ack=True, inactivity_timeout=0.2)

    def infer(self, tensor):
        task_id = uuid.uuid4().hex
        headers = {
            self.codec.TARGET_URI_ATTR_NAME: f'file://{self.program}?raw=False',
            self.codec.RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([self.reply_queue]),
            self.codec.MESSAGE_ID_ATTR_NAME: task_id,
        }
        payload = self.codec.serialize(self.codec.PyTask({'images': tensor}))
        a = time.perf_counter()
        self.channel.basic_publish(exchange=EXCHANGE, routing_key=TPU_INPUT_ROUTE, body=payload,
                                   properties=pika.BasicProperties(headers=headers))
        publish_ms = ms(a)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            method, properties, body = next(self.consumer)
            if method is None:
                continue
            if not properties or not properties.headers or properties.headers.get(self.codec.MESSAGE_ID_ATTR_NAME) != task_id:
                continue
            data, metadata, worker = self.codec.deserialize(body)
            if not data or worker.get('status_code') != 0:
                raise RuntimeError(f'TPU worker: {worker}')
            raw = data.get('output0')
            if raw is None:
                raise RuntimeError(f'output0 missing: {list(data)}')
            return raw, worker, publish_ms, len(payload), len(body)
        raise TimeoutError(f'TPU did not respond within {self.timeout}s')

    def close(self):
        if self.connection.is_open:
            try:
                self.channel.cancel()
            except Exception:
                pass
            self.connection.close()


def handle_jpeg(jpeg, tpu, frame_id, confidence):
    t0 = time.perf_counter()
    tensor, meta = jpeg_to_tensor(jpeg)
    prep_ms = ms(t0)
    raw, worker, tpu_publish_ms, tpu_request_bytes, tpu_response_bytes = tpu.infer(tensor)
    infer_return_ms = ms(t0)
    dets = person_detections(raw, meta, confidence)
    return {
        'ok': True,
        'frame_id': frame_id,
        'source_width': meta[0], 'source_height': meta[1],
        'detections': dets,
        'stats': {
            'gateway_total_ms': ms(t0),
            'preprocess_ms': prep_ms,
            'tpu_rpc_ms': round(infer_return_ms - prep_ms, 2),
            'tpu_publish_ms': tpu_publish_ms,
            'tpu_worker_ms': round(float(worker.get('inference_time', 0.0)) * 1000, 2),
            'internal_tensor_request_bytes': tpu_request_bytes,
            'internal_tensor_response_bytes': tpu_response_bytes,
        }
    }


def main():
    p = argparse.ArgumentParser(description='Remote JPEG -> LinQ TPU -> JSON bridge; no recompilation')
    p.add_argument('--host', default='10.0.114.30')
    p.add_argument('--port', type=int, default=5672)
    p.add_argument('--program', required=True, help='Existing worker-side /auto/cache/.../yolov8n.tpu')
    p.add_argument('--route', default=os.getenv('SAFESIGHT_ROUTE') or 'safesight.jpeg.' + secrets.token_hex(12))
    p.add_argument('--timeout', type=float, default=40.0)
    p.add_argument('--confidence', type=float, default=0.25)
    args = p.parse_args()
    if args.route == TPU_INPUT_ROUTE or len(args.route) > 180:
        p.error('Invalid route')
    ingress = connect(args.host, args.port, args.timeout)
    tpu = TPUClient(args.host, args.port, args.program, args.timeout)
    channel = ingress.channel()
    # Temporary queue is deleted with gateway connection; old camera frames are dropped.
    reply = channel.queue_declare(queue='', exclusive=True, auto_delete=True,
                                  arguments={'x-max-length': 1, 'x-message-ttl': 3000})
    gateway_queue = reply.method.queue
    channel.queue_bind(exchange=EXCHANGE, queue=gateway_queue, routing_key=args.route)
    channel.basic_qos(prefetch_count=1)
    print(f'GATEWAY READY | route: {args.route}', flush=True)
    print(f'TPU program: {args.program}', flush=True)
    print('Keep this process running while testing. Press Ctrl+C to stop.', flush=True)
    try:
        for method, props, jpeg in channel.consume(gateway_queue, auto_ack=False, inactivity_timeout=1.0):
            if method is None:
                continue
            frame_id = getattr(props, 'correlation_id', None) or 'unknown'
            reply_to = getattr(props, 'reply_to', None)
            if not reply_to or len(reply_to) > 255 or len(jpeg) > 2_000_000:
                channel.basic_ack(method.delivery_tag)
                continue
            try:
                result = handle_jpeg(jpeg, tpu, frame_id, args.confidence)
            except Exception as exc:
                result = {'ok': False, 'frame_id': frame_id, 'error': f'{type(exc).__name__}: {str(exc)[:400]}'}
            output = json.dumps(result, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
            try:
                channel.basic_publish(exchange=EXCHANGE, routing_key=reply_to, body=output,
                    properties=pika.BasicProperties(content_type='application/json', correlation_id=frame_id))
                channel.basic_ack(method.delivery_tag)
            except Exception:
                channel.basic_nack(method.delivery_tag, requeue=False)
                raise
            if result.get('ok'):
                print(f'frame={frame_id[:12]} | {len(result["detections"])} people | '
                      f'cloud={result["stats"]["gateway_total_ms"]}ms | '
                      f'TPU={result["stats"]["tpu_worker_ms"]}ms | '
                      f'in={len(jpeg)}B out={len(output)}B', flush=True)
            else:
                print(f'FAILED frame={frame_id[:12]} | {result["error"]}', flush=True)
    except KeyboardInterrupt:
        print('\nGateway stopped.', flush=True)
    finally:
        tpu.close()
        if ingress.is_open:
            try:
                channel.cancel()
            except Exception:
                pass
            ingress.close()


if __name__ == '__main__':
    main()

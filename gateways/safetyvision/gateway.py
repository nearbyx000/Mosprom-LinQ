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
    from .preprocessing import jpeg_to_tensor
    from .decoder import safetyvision_detections
else:
    from preprocessing import jpeg_to_tensor
    from decoder import safetyvision_detections

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
    """Single TPU RPC in flight; reconnect after a lost connection.

    Never resend an ambiguous task: a disconnected client cannot know whether
    the TPU worker finished it. The failing request is reported as an error,
    and a *new* later frame gets a fresh correlation ID.
    """
    def __init__(self, host, port, program, timeout):
        self.codec = importlib.import_module("tpu_pycloud_format")
        self.host, self.port = host, port
        self.program, self.timeout = program, timeout
        self.connection = None
        self.channel = None
        self.consumer = None
        self.reply_queue = None
        self.next_attempt_at = 0.0

    def _open(self):
        connection = connect(self.host, self.port, self.timeout)
        try:
            channel = connection.channel()
            q = channel.queue_declare(queue='', exclusive=True, auto_delete=True)
            reply_queue = q.method.queue
            channel.queue_bind(exchange=EXCHANGE, queue=reply_queue, routing_key=reply_queue)
            consumer = channel.consume(reply_queue, auto_ack=True, inactivity_timeout=0.2)
        except Exception:
            try:
                connection.close()
            except Exception:
                pass
            raise
        self.connection, self.channel = connection, channel
        self.reply_queue, self.consumer = reply_queue, consumer
        print('TPU RPC connection ready', flush=True)

    def _ensure(self):
        if (self.connection is not None and self.connection.is_open and
                self.channel is not None and self.channel.is_open):
            return
        if time.monotonic() < self.next_attempt_at:
            raise RuntimeError('TPU RPC connection recovering; next frame will retry')
        self.close()
        try:
            self._open()
        except Exception:
            self.next_attempt_at = time.monotonic() + 2.0
            raise

    def infer(self, tensor):
        self._ensure()
        task_id = uuid.uuid4().hex
        headers = {
            self.codec.TARGET_URI_ATTR_NAME: f'file://{self.program}?raw=False',
            self.codec.RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([self.reply_queue]),
            self.codec.MESSAGE_ID_ATTR_NAME: task_id,
        }
        payload = self.codec.serialize(self.codec.PyTask({'images': tensor}))
        a = time.perf_counter()
        try:
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
                    raise RuntimeError(f'TPU worker error/status: {worker}')
                raw = data.get('output0')
                if raw is None:
                    raise RuntimeError(f'output0 missing: {list(data)}')
                return raw, worker, publish_ms, len(payload), len(body)
            raise TimeoutError(f'TPU did not respond within {self.timeout}s')
        except Exception:
            # A failed RPC may have run remotely. Do not retry the same tensor.
            # Recreate both reply queue and channel for the *next* frame.
            self.close()
            self.next_attempt_at = time.monotonic() + 1.0
            raise

    def close(self):
        connection, channel = self.connection, self.channel
        self.connection = None
        self.channel = None
        self.consumer = None
        self.reply_queue = None
        if connection is not None:
            try:
                if connection.is_open:
                    if channel is not None and channel.is_open:
                        try:
                            channel.cancel()
                        except Exception:
                            pass
                    connection.close()
            except Exception:
                pass


def handle_jpeg(jpeg, tpu, frame_id, confidence, class_thresholds=None):
    t0 = time.perf_counter()
    tensor, meta = jpeg_to_tensor(jpeg, backend='opencv')
    prep_ms = ms(t0)
    raw, worker, tpu_publish_ms, tpu_request_bytes, tpu_response_bytes = tpu.infer(tensor)
    infer_return_ms = ms(t0)
    dets = safetyvision_detections(raw, meta, confidence, class_thresholds=class_thresholds)
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
    p.add_argument('--program', required=True, help='Existing worker-side /auto/cache/.../*.tpu; never uploaded or replaced')
    p.add_argument('--route', default=os.getenv('SAFESIGHT_ROUTE') or 'safesight.jpeg.sv.' + secrets.token_hex(12))
    p.add_argument('--timeout', type=float, default=40.0)
    p.add_argument('--confidence', type=float, default=0.4)
    p.add_argument('--class-thresholds', type=json.loads, default={},
                   help='JSON canonical-label thresholds; keep positives low for contradiction checks')
    args = p.parse_args()
    if args.route == TPU_INPUT_ROUTE or len(args.route) > 180:
        p.error('Invalid route')
    # Keep routing key stable during automatic reconnects.
    # Only one gateway process should bind this route at any time.
    tpu = TPUClient(args.host, args.port, args.program, args.timeout)
    print(f'GATEWAY ROUTE | {args.route}', flush=True)
    print(f'TPU program: {args.program}', flush=True)
    try:
        while True:
            ingress = None
            channel = None
            try:
                ingress = connect(args.host, args.port, args.timeout)
                channel = ingress.channel()
                reply = channel.queue_declare(queue='', exclusive=True, auto_delete=True,
                                              arguments={'x-max-length': 1, 'x-message-ttl': 3000})
                gateway_queue = reply.method.queue
                channel.queue_bind(exchange=EXCHANGE, queue=gateway_queue, routing_key=args.route)
                channel.basic_qos(prefetch_count=1)
                print(f'GATEWAY READY | route: {args.route}', flush=True)
                for method, props, jpeg in channel.consume(gateway_queue, auto_ack=False, inactivity_timeout=1.0):
                    if method is None:
                        continue
                    frame_id = getattr(props, 'correlation_id', None) or 'unknown'
                    reply_to = getattr(props, 'reply_to', None)
                    if not reply_to or len(reply_to) > 255 or len(jpeg) > 2_000_000:
                        channel.basic_ack(method.delivery_tag)
                        continue
                    try:
                        result = handle_jpeg(jpeg, tpu, frame_id, args.confidence, args.class_thresholds)
                    except Exception as exc:
                        result = {'ok': False, 'frame_id': frame_id,
                                  'error': f'{type(exc).__name__}: {str(exc)[:300]}'}
                    output = json.dumps(result, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
                    # If ingress died as well, restart it. Do not resubmit the frame.
                    channel.basic_publish(exchange=EXCHANGE, routing_key=reply_to, body=output,
                        properties=pika.BasicProperties(content_type='application/json', correlation_id=frame_id))
                    channel.basic_ack(method.delivery_tag)
                    if result.get('ok'):
                        print(f'frame={frame_id[:12]} | {len(result["detections"])} SV detections | '
                              f'cloud={result["stats"]["gateway_total_ms"]}ms | '
                              f'TPU={result["stats"]["tpu_worker_ms"]}ms | '
                              f'in={len(jpeg)}B out={len(output)}B', flush=True)
                    else:
                        print(f'FAILED frame={frame_id[:12]} | {result["error"]}', flush=True)
            except Exception as exc:
                print(f'INGRESS LOST: {type(exc).__name__}: {str(exc)[:200]} | reconnecting in 2s', flush=True)
                time.sleep(2.0)
            finally:
                if ingress is not None:
                    try:
                        if ingress.is_open:
                            if channel is not None and channel.is_open:
                                try:
                                    channel.cancel()
                                except Exception:
                                    pass
                            ingress.close()
                    except Exception:
                        pass
    except KeyboardInterrupt:
        print('Gateway stopped.', flush=True)
    finally:
        tpu.close()


if __name__ == '__main__':
    main()

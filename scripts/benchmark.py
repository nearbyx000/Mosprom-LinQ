"""Backend benchmark (no hidden CPU fallback, no simulated TPU metrics).

Example:
 python scripts/benchmark.py --mode local --model models/person.pt --source video.mp4 --frames 100
Use --mode linq only after implementing the official LinQ adapter.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safesight.inference import DemoBackend, LinQBackend, UltralyticsBackend
from safesight.pipeline import SyntheticVideo


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('demo','local','linq'), default='demo')
    p.add_argument('--model', default='')
    p.add_argument('--source', default='synthetic')
    p.add_argument('--frames', type=int, default=100)
    p.add_argument('--warmup', type=int, default=10)
    args = p.parse_args()
    if args.frames <= 0 or args.warmup < 0:
        p.error('frames must be >0 and warmup >=0')
    backend = {'demo': DemoBackend, 'local': lambda: UltralyticsBackend(args.model), 'linq': LinQBackend}[args.mode]()
    cap = SyntheticVideo() if args.source == 'synthetic' else cv2.VideoCapture(args.source)
    measurements = []
    total_start = time.perf_counter()
    try:
        for i in range(args.frames + args.warmup):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Could not read frame {i}; choose a longer video or a stream')
            start = time.perf_counter()
            backend.infer(frame)
            delta_ms = (time.perf_counter() - start) * 1000
            if i >= args.warmup:
                measurements.append(delta_ms)
    finally:
        cap.release()
    times = sorted(measurements)
    total = time.perf_counter() - total_start
    print(f'MODE: {args.mode}; BACKEND: {backend.name}')
    print(f'LinQ TPU result: {"NO — synthetic demo" if args.mode == "demo" else "NOT VERIFIED — confirm via official SDK" if args.mode == "linq" else "NO — local inference"}')
    print(f'n={len(times)}, mean={statistics.mean(times):.2f}ms, median={statistics.median(times):.2f}ms, p95={times[min(len(times)-1,int(len(times)*0.95))]:.2f}ms')
    print(f'backend-only theoretical throughput = {1000/statistics.mean(times):.2f} inferences/s; end-to-end including frame capture = {(args.frames+args.warmup)/total:.2f} frames/s')
    print('NOTE: This is NOT an accelerator hardware benchmark unless verified with the official LinQ SDK.')

if __name__ == '__main__':
    main()

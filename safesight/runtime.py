"""Bounded latest-value slots and thread-safe rolling measurements."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    sequence: int
    captured_at: float
    captured_wall: float
    image: np.ndarray
    generation: int = 0
    safetyvision_observed: bool = True


class Latest:
    """One value per stage. Slow readers skip old frames without growing a queue."""
    def __init__(self):
        self.condition = threading.Condition()
        self.value = None
        self.sequence = 0
        self.closed = False

    def put(self, value):
        with self.condition:
            if self.closed:
                return
            self.sequence += 1
            self.value = value
            self.condition.notify_all()

    def get(self, after=0, timeout=0.2):
        with self.condition:
            self.condition.wait_for(lambda: self.sequence > after or self.closed, timeout)
            if self.sequence <= after:
                return after, None
            return self.sequence, self.value

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class Metrics:
    """At most 300 observations per metric; rates use the last ten seconds."""
    def __init__(self):
        self.lock = threading.Lock()
        self.samples = defaultdict(lambda: deque(maxlen=300))
        self.ticks = defaultdict(lambda: deque(maxlen=600))
        self.counts = defaultdict(int)
        self.started = time.perf_counter()

    def observe(self, name, milliseconds):
        with self.lock:
            self.samples[name].append(float(milliseconds))

    def tick(self, name, count=1):
        with self.lock:
            self.counts[name] += count
            if count == 1:
                self.ticks[name].append(time.perf_counter())

    def snapshot(self):
        now = time.perf_counter()
        with self.lock:
            latencies = {name: {"samples": len(values),
                                "p50": round(float(np.percentile(values, 50)), 2),
                                "p95": round(float(np.percentile(values, 95)), 2)}
                         for name, values in self.samples.items() if values}
            rates = {}
            for name, ticks in self.ticks.items():
                recent = [t for t in ticks if t >= now - 10]
                rates[name] = round(max(0, len(recent) - 1) / max(0.001, now - recent[0]), 2) if recent else 0.0
            return {"latency_ms": latencies, "fps": rates, "counts": dict(self.counts)}

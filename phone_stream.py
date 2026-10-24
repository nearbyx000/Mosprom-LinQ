"""One browser camera publisher, shared in-memory JPEG feed, no recordings."""
import asyncio
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

MAX_FRAME_BYTES = 1024 * 1024


def sanitize_jpeg(body: bytes) -> bytes:
    if len(body) > MAX_FRAME_BYTES:
        raise ValueError("Frame exceeds 1 MiB")
    with Image.open(BytesIO(body)) as header:
        if header.format != "JPEG" or header.width * header.height > 1920 * 1080:
            raise ValueError("Expected JPEG up to 1920x1080 pixels")
    image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid JPEG")
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return encoded.tobytes()


class PhoneFeed:
    def __init__(self):
        self.condition = asyncio.Condition()
        self.owner = None
        self.jpeg = None
        self.sequence = 0

    async def claim(self):
        async with self.condition:
            if self.owner is not None:
                return None
            self.owner = object()
            self.jpeg = None
            return self.owner

    async def publish(self, owner, jpeg):
        async with self.condition:
            if self.owner is not owner:
                return
            self.jpeg = jpeg
            self.sequence += 1
            self.condition.notify_all()

    async def release(self, owner):
        async with self.condition:
            if self.owner is owner:
                self.owner = None
                self.jpeg = None
                self.condition.notify_all()

    async def next_frame(self, after=-1, timeout=5):
        async with self.condition:
            try:
                await asyncio.wait_for(self.condition.wait_for(
                    lambda: self.owner is None or (self.jpeg is not None and self.sequence != after)
                ), timeout)
            except asyncio.TimeoutError:
                return None
            if self.owner is None or self.jpeg is None:
                return None
            return self.sequence, self.jpeg

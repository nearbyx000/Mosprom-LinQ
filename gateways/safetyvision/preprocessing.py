"""YOLOv8n 640x640 preprocessing and COCO person decoding.

Compatible with the existing quantized LinQ yolov8n.tpu program.
"""
from io import BytesIO
import numpy as np
from PIL import Image, UnidentifiedImageError


def jpeg_to_tensor(jpeg_bytes: bytes, side: int = 640, backend: str = 'pil'):
    """Decode JPEG, RGB letterbox, return (float32 NCHW tensor, (w,h,ratio,left,top))."""
    if not jpeg_bytes or len(jpeg_bytes) > 2_000_000:
        raise ValueError("JPEG size must be 1..2,000,000 bytes")
    try:
        with Image.open(BytesIO(jpeg_bytes)) as source:
            if source.format != 'JPEG':
                raise ValueError('Only JPEG input is accepted')
            if max(source.size) > 4096 or min(source.size) < 1:
                raise ValueError('Invalid image dimensions (max 4096 px per side)')
            image = source.convert('RGB')
    except UnidentifiedImageError as exc:
        raise ValueError('Invalid JPEG data') from exc
    width, height = image.size
    if backend not in {'pil', 'opencv'}:
        raise ValueError('Unknown resize backend')
    if backend == 'opencv':
        import cv2
        bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('Invalid JPEG data')
        # OpenCV applies EXIF orientation, as does the calibration compiler.
        # Camera JPEGs have no EXIF; imported calibration photos may have it.
        height, width = bgr.shape[:2]
    ratio = min(side / width, side / height)
    new_width, new_height = round(width * ratio), round(height * ratio)
    left = round((side - new_width) / 2 - 0.1)
    top = round((side - new_height) / 2 - 0.1)
    if backend == 'opencv':
        # SafetyVision calibration uses OpenCV INTER_LINEAR. Pillow's
        # antialiased downsampling is numerically different. Keep the original
        # PIL default for the already working YOLOv8n gateway.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas = np.full((side, side, 3), 114, dtype=np.uint8)
        canvas[top:top+new_height, left:left+new_width] = cv2.resize(
            rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        arr = canvas.astype(np.float32)
    else:
        resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (side, side), (114, 114, 114))
        canvas.paste(resized, (left, top))
        arr = np.asarray(canvas, dtype=np.float32)
    tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None] / 255.0)
    assert tensor.shape == (1, 3, side, side)
    return tensor, (width, height, ratio, left, top)


def person_detections(raw, meta, confidence: float = 0.25, iou_threshold: float = 0.45):
    """Return list of dicts with bbox in ORIGINAL JPEG pixel coordinates."""
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f'Unsupported YOLO shape: {arr.shape}')
    if arr.shape[0] == 84:
        arr = arr.T
    elif arr.shape[1] != 84:
        raise ValueError(f'Expected 84 features, received {arr.shape}')
    scores_all = arr[:, 4]
    valid = np.isfinite(scores_all) & (scores_all >= confidence) & np.all(np.isfinite(arr[:, :4]), axis=1)
    candidates = arr[valid]
    if len(candidates) == 0:
        return []
    xywh = candidates[:, :4].astype(np.float64)
    scores = candidates[:, 4].astype(np.float64)
    boxes = np.column_stack((
        xywh[:, 0] - xywh[:, 2] / 2,
        xywh[:, 1] - xywh[:, 3] / 2,
        xywh[:, 0] + xywh[:, 2] / 2,
        xywh[:, 1] + xywh[:, 3] / 2,
    ))
    order = np.argsort(scores)[::-1]
    kept = []
    while order.size and len(kept) < 100:
        i = int(order[0])
        kept.append(i)
        others = order[1:]
        if not len(others):
            break
        iw = np.maximum(0, np.minimum(boxes[i, 2], boxes[others, 2]) - np.maximum(boxes[i, 0], boxes[others, 0]))
        ih = np.maximum(0, np.minimum(boxes[i, 3], boxes[others, 3]) - np.maximum(boxes[i, 1], boxes[others, 1]))
        inter = iw * ih
        a = max(0.0, boxes[i, 2] - boxes[i, 0]) * max(0.0, boxes[i, 3] - boxes[i, 1])
        b = np.maximum(0, boxes[others, 2] - boxes[others, 0]) * np.maximum(0, boxes[others, 3] - boxes[others, 1])
        overlaps = inter / np.maximum(a + b - inter, 1e-9)
        order = others[overlaps <= iou_threshold]
    width, height, ratio, left, top = meta
    detections = []
    for i in kept:
        x1, y1, x2, y2 = boxes[i]
        box = [
            int(round(np.clip((x1 - left) / ratio, 0, width))),
            int(round(np.clip((y1 - top) / ratio, 0, height))),
            int(round(np.clip((x2 - left) / ratio, 0, width))),
            int(round(np.clip((y2 - top) / ratio, 0, height))),
        ]
        if box[2] > box[0] and box[3] > box[1]:
            detections.append({"class_name": "person", "confidence": round(float(scores[i]), 5), "bbox": box, "model": "linq-yolov8n"})
    return detections

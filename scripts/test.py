"""Check SafetyVision v2 ONNX on an image using ONNX Runtime.

Example:
    python test_safetyvision_onnx.py --image test.jpg --model models/v2/best_640.onnx
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASSES = [
    "Fall-Detected", "Gloves", "Goggles", "Hardhat", "Mask",
    "NO-Gloves", "NO-Goggles", "NO-Hardhat", "NO-Mask",
    "NO-Safety Vest", "No_Harness", "Person", "Safety Vest",
]


def letterbox(image: np.ndarray, size: int = 640):
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, scale, left, top


def postprocess(output, scale, left, top, image_shape, conf_threshold=0.35, iou_threshold=0.45):
    pred = output[0].T  # (8400, 17) for the 640px model
    if pred.ndim != 2 or pred.shape[1] != 4 + len(CLASSES):
        raise ValueError(f"Unexpected model output: {pred.shape}")
    class_ids = np.argmax(pred[:, 4:], axis=1)
    scores = np.max(pred[:, 4:], axis=1)
    valid = np.where(scores >= conf_threshold)[0]
    if valid.size == 0:
        return []
    boxes = pred[valid, :4].copy()
    class_ids, scores = class_ids[valid], scores[valid]

    # Model outputs (cx, cy, w, h) in letterboxed pixel coordinates.
    boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - left) / scale
    boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - top) / scale
    boxes[:, 2] /= scale
    boxes[:, 3] /= scale
    h, w = image_shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - boxes[:, 0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - boxes[:, 1])

    detections = []
    for class_id in np.unique(class_ids):
        subset = np.flatnonzero(class_ids == class_id)
        indices = cv2.dnn.NMSBoxes(
            boxes[subset].tolist(), scores[subset].tolist(),
            score_threshold=conf_threshold, nms_threshold=iou_threshold
        )
        for j in np.asarray(indices).reshape(-1):
            i = subset[int(j)]
            x, y, bw, bh = boxes[i]
            if bw <= 0 or bh <= 0:
                continue
            detections.append((int(class_id), float(scores[i]),
                               int(x), int(y), int(x + bw), int(y + bh)))
    return sorted(detections, key=lambda d: d[1], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/v2/best_640.onnx")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="result.jpg")
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()
    if not Path(args.model).is_file():
        parser.error(f"Model not found: {args.model}")
    image = cv2.imread(args.image)
    if image is None:
        parser.error(f"Cannot read image: {args.image}")

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    tensor, scale, left, top = letterbox(image)
    start = time.perf_counter()
    prediction = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
    person_scores = prediction[0, 15, :]
    print("Person max confidence:", person_scores.max())
    print("Person candidates > 0.15:", (person_scores > 0.15).sum())
    elapsed_ms = (time.perf_counter() - start) * 1000
    detections = postprocess(prediction, scale, left, top, image.shape,
                             conf_threshold=args.confidence)
    for cid, score, x1, y1, x2, y2 in detections:
        name = CLASSES[cid]
        print(f"{name}: {score:.2f}  box=({x1},{y1},{x2},{y2})")
        color = (0, 0, 255) if name.startswith("NO-") or name in ("Fall-Detected", "No_Harness") else (0, 180, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{name} {score:.2f}", (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
    if not cv2.imwrite(args.out, image):
        raise RuntimeError(f"Failed to save {args.out}")
    print(f"Inference: {elapsed_ms:.1f} ms | Detections: {len(detections)} | Saved: {args.out}")


if __name__ == "__main__":
    main()

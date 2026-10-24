#!/usr/bin/env python3
"""Compile an ONNX object detector to LinQ H1M .tpu using tpu_framework 2026.9.45.

The vendor SDK is REQUIRED for compilation. API calls follow the official
`tpu_framework.pdf`, Chapter 5 (End-To-End Test), pp. 27, 30-31.
This is a project adaptation, not a vendor-tested YOLOv8n configuration.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_yolo(path: Path, height: int, width: int) -> np.ndarray:
    """Letterbox RGB NCHW float32 [0,1], pad=114, like YOLOv8 predict(imgsz)."""
    # imdecode/fromfile supports paths containing Cyrillic characters on Windows.
    file_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    ih, iw = image.shape[:2]
    r = min(height / ih, width / iw)
    new_w, new_h = round(iw * r), round(ih * r)
    dw, dh = (width - new_w) / 2, (height - new_h) / 2
    if new_w != iw or new_h != ih:
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(
        image, round(dh - 0.1), round(dh + 0.1),
        round(dw - 0.1), round(dw + 0.1),
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    if image.shape[:2] != (height, width):
        raise RuntimeError(f"Unexpected letterbox shape: {image.shape}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(
        image_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    )


def get_input(model: onnx.ModelProto) -> tuple[str, int, int]:
    inputs = list(model.graph.input)
    if len(inputs) != 1:
        raise ValueError(f"Expected 1 ONNX input, got {len(inputs)}")
    first = inputs[0]
    shape = [d.dim_value if d.HasField("dim_value") else None
             for d in first.type.tensor_type.shape.dim]
    if len(shape) != 4 or shape[:2] != [1, 3] or not all(
        isinstance(x, int) and x > 0 for x in shape[2:]
    ):
        raise ValueError(f"Expected static NCHW [1,3,H,W], got {shape}")
    return first.name, shape[2], shape[3]


def inspect(model: onnx.ModelProto) -> None:
    opsets = [(imp.domain or "ai.onnx", imp.version) for imp in model.opset_import]
    print("ONNX opset:", opsets)
    for label, tensors in (("Inputs", model.graph.input), ("Outputs", model.graph.output)):
        print(label + ":")
        for t in tensors:
            dims = [d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in t.type.tensor_type.shape.dim]
            print("  ", t.name, dims)
    print("Graph nodes:", len(model.graph.node))


def compile_model(args: argparse.Namespace, model: onnx.ModelProto) -> None:
    input_name, h, w = get_input(model)
    if not args.calib_dir:
        raise ValueError("--calib-dir is required in compile mode")
    base = Path(args.calib_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"No calibration directory: {base}")
    paths = sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"Calibration directory has no images: {base}")
    paths = paths[:args.limit]
    print(f"Calibration: {len(paths)} images, {h}x{w}, input='{input_name}'", flush=True)
    samples = np.concatenate([preprocess_yolo(p, h, w) for p in paths], axis=0)
    print("Calibration tensor:", samples.shape, samples.dtype, flush=True)

    # These names are copied from the vendor's End-To-End example (PDF pp. 19, 27, 30-31).
    try:
        from tpu_framework import (RegularModel, Network, TPU_H1M, O1,
                                   TpuProgram, compiler)
    except ImportError as exc:
        raise RuntimeError(
            "LinQ SDK is not installed in this Python environment. "
            "Use Ubuntu + Python 3.12.11 and the vendor SDK package."
        ) from exc

    print("Stage 1/4: convert ONNX into RegularModel...", flush=True)
    regular = RegularModel(model=model)
    anchors_mapping = regular.anchors_mapping
    if input_name not in anchors_mapping:
        raise KeyError(
            f"Input {input_name!r} absent from SDK anchors_mapping: {anchors_mapping}"
        )
    sdk_input_name = anchors_mapping[input_name]
    print(f"Anchor mapping: {input_name} -> {sdk_input_name}", flush=True)

    print("Stage 2/4: calibrate + quantize...", flush=True)
    calibration_dict = {sdk_input_name: samples}
    start = time.perf_counter()
    thresholds = regular.calibrate(calibration_dict)
    print(f"Calibration finished in {time.perf_counter()-start:.2f}s", flush=True)
    if args.thresholds_out:
        out_thresholds = Path(args.thresholds_out)
        out_thresholds.parent.mkdir(parents=True, exist_ok=True)
        RegularModel.dump_thresholds(str(out_thresholds), thresholds)
        print("Thresholds:", out_thresholds, flush=True)
    quantized_model = regular.quantize(thresholds)

    print("Stage 3/4: build network + compile H1M, preset O1...", flush=True)
    network, renaming = Network.from_quantized_model(quantized_model)
    network.set_batch(1)
    executable, tensor_descriptions = compiler.compile_(TPU_H1M, network, O1)

    print("Stage 4/4: save TPU program...", flush=True)
    program = TpuProgram.from_executable(executable, tensor_descriptions)
    path_out = Path(args.out)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    program.to_file(str(path_out))
    if not path_out.is_file() or path_out.stat().st_size == 0:
        raise RuntimeError("Compiler returned, but no nonempty .tpu program was created")
    print(f"SUCCESS: {path_out.resolve()} ({path_out.stat().st_size:,} bytes)", flush=True)
    print("NOTE: output is not yet validated on physical/remote LinQ TPU.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to exported ONNX model")
    p.add_argument("--mode", choices=("inspect", "compile"), default="inspect")
    p.add_argument("--calib-dir", help="Directory of calibration photos")
    p.add_argument("--limit", type=int, default=16, help="Number of calibration images (default 16)")
    p.add_argument("--out", default="artifacts/yolov8n_h1m_o1.tpu")
    p.add_argument("--thresholds-out", default="artifacts/yolov8n_thresholds.json")
    args = p.parse_args()
    if args.limit < 1:
        p.error("--limit must be >= 1")
    model_path = Path(args.model)
    if not model_path.is_file():
        p.error(f"ONNX file not found: {model_path}")
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    inspect(model)
    if args.mode == "compile":
        compile_model(args, model)
    else:
        print("ONNX validated. For .tpu generation, use --mode compile --calib-dir ...")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        raise

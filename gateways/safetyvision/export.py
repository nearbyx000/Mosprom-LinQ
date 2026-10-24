#!/usr/bin/env python3
"""Download SafetyVision v2 PT (if missing), export static ONNX opset 11 and verify I/O.

Run in the Windows Python environment that has Ultralytics/PyTorch; no LinQ SDK needed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import tempfile
import os
import onnx

REPO = 'ayushgupta7777/safetyvision-yolov8'


def shape(value):
    return [d.dim_value if d.HasField('dim_value') else (d.dim_param or '?')
            for d in value.type.tensor_type.shape.dim]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', default='models/v2/best.pt', help='Existing .pt or destination for HF download')
    p.add_argument('--output', default='models/safetyvision_candidate_op11.onnx',help='New output only; existing files are never replaced')
    p.add_argument('--skip-download', action='store_true')
    args = p.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        p.error('Output already exists; choose a new candidate filename')
    output.parent.mkdir(parents=True,exist_ok=True)
    weights = Path(args.weights).resolve()
    if not weights.exists():
        if args.skip_download:
            p.error(f'Missing weights {weights}')
        if weights.as_posix().endswith('/v2/best.pt'):
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError('Install huggingface_hub: python -m pip install huggingface_hub') from exc
            print('Downloading official v2/best.pt from Hugging Face ...', flush=True)
            weights = Path(hf_hub_download(repo_id=REPO, filename='v2/best.pt', local_dir=str(weights.parents[1]))).resolve()
        else:
            p.error('For auto-download use --weights v2/best.pt or supply an existing path')
    os.environ.setdefault('YOLO_CONFIG_DIR',str(output.parent/'export-settings'))
    from ultralytics import YOLO
    print('Loaded weights:', weights, flush=True)
    model = YOLO(str(weights))
    print('Classes:', model.names, flush=True)
    expected = ['Fall-Detected','Gloves','Goggles','Hardhat','Mask','NO-Gloves',
                'NO-Goggles','NO-Hardhat','NO-Mask','NO-Safety Vest',
                'No_Harness','Person','Safety Vest']
    actual = [model.names.get(i) for i in range(13)]
    if actual != expected:
        raise ValueError(f'Unexpected model class mapping: {actual}. Expected {expected}. Wrong weights?')
    # Ultralytics writes beside its source weights. Export a temporary copy to
    # avoid overwriting an existing best.onnx belonging to the working model.
    with tempfile.TemporaryDirectory(prefix='sv-export-',dir=output.parent) as workspace:
        temporary_weights = Path(workspace)/'candidate.pt'
        shutil.copy2(weights,temporary_weights)
        exported = Path(YOLO(str(temporary_weights)).export(format='onnx',imgsz=640,batch=1,opset=11,
                                                         dynamic=False,simplify=False,nms=False)).resolve()
        model_graph = onnx.load(str(exported))
        onnx.checker.check_model(model_graph)
        if (len(model_graph.graph.input)!=1 or len(model_graph.graph.output)!=1 or
                shape(model_graph.graph.input[0])!=[1,3,640,640] or shape(model_graph.graph.output[0])!=[1,17,8400]):
            raise ValueError('Unexpected candidate I/O; output was not saved')
        with output.open('xb') as stream:
            stream.write(exported.read_bytes())
    onnx.checker.check_model(model_graph)
    inp, out = model_graph.graph.input, model_graph.graph.output
    if len(inp) != 1 or len(out) != 1:
        raise ValueError(f'Unexpected I/O count: {len(inp)} inputs, {len(out)} outputs')
    ishape, oshape = shape(inp[0]), shape(out[0])
    print('ONNX opset:', [(i.domain or 'ai.onnx',i.version) for i in model_graph.opset_import])
    print('INPUT:', inp[0].name, ishape)
    print('OUTPUT:', out[0].name, oshape)
    if ishape != [1,3,640,640] or oshape != [1,17,8400]:
        raise ValueError('Unexpected model dimensions. Do not compile without inspecting output format.')
    print(f'EXPORT OK: {output} ({output.stat().st_size:,} bytes)', flush=True)

if __name__ == '__main__':
    main()

"""Explicit offline ONNX / real remote LinQ comparison on identical tensors.

No local fallback in TPU mode, no program upload, no production route changes.
Output directories must be new. JPEGs and raw tensors stay in local artifacts.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from gateways.safetyvision.decoder import CLASSES, safetyvision_detections
from gateways.safetyvision.preprocessing import jpeg_to_tensor
from safesight.evaluation import evaluate
from scripts.safetyvision_eval import read, write

CALIBRATION_TAGS = {'helmet', 'no_helmet', 'fall', 'normal', 'negative', 'low_light', 'small_person'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def onnx_session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    names = ast.literal_eval(session.get_modelmeta().custom_metadata_map.get('names', '{}'))
    if (len(inputs) != 1 or inputs[0].name != 'images' or inputs[0].shape != [1, 3, 640, 640] or
            inputs[0].type != 'tensor(float)' or len(outputs) != 1 or outputs[0].shape != [1, 17, 8400] or
            names != dict(enumerate(CLASSES))):
        raise ValueError('Wrong ONNX input / output / class order; refusing comparison')
    return session


def prepare_inputs(manifest_path, output, backend='opencv'):
    manifest = read(manifest_path)
    rows, seen = [], set()
    for index, entry in enumerate(manifest['images']):
        image_id = entry['id']
        if not isinstance(image_id, str) or image_id in seen:
            raise ValueError('Duplicate or invalid image ID')
        seen.add(image_id)
        path = Path(manifest_path).resolve().parent / entry['path']
        jpeg = path.read_bytes()
        tensor, meta = jpeg_to_tensor(jpeg, backend=backend)
        tensor_name = f'tensor_{index:04d}.npy'
        np.save(output / tensor_name, tensor, allow_pickle=False)
        (output / f'image_{index:04d}.jpg').write_bytes(jpeg)
        rows.append({'id': image_id, 'jpeg_sha256': digest(jpeg), 'tensor_sha256': digest(tensor.tobytes()),
                     'tensor': tensor_name, 'meta': list(meta), 'preprocess': backend,
                     'split': entry.get('split'), 'group': entry.get('group'), 'tags': entry.get('tags', [])})
    if not rows:
        raise ValueError('Manifest contains no images')
    return {'kind': manifest['kind'], 'images': rows}


def load_tensor(base, entry):
    path = (Path(base) / entry['tensor']).resolve()
    if not path.is_relative_to(Path(base).resolve()):
        raise ValueError('Tensor path escapes input bundle')
    tensor = np.load(path, allow_pickle=False)
    if (tensor.shape != (1, 3, 640, 640) or tensor.dtype != np.float32 or not np.isfinite(tensor).all() or
            tensor.min() < 0 or tensor.max() > 1 or digest(tensor.tobytes()) != entry['tensor_sha256']):
        raise ValueError('Input tensor shape / dtype / hash mismatch')
    return tensor


def infer(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    remote = None
    if args.inputs:
        index = read(Path(args.inputs) / 'index.json')
        base = Path(args.inputs)
    elif args.manifest and args.backend == 'onnx':
        index = prepare_inputs(args.manifest, output, args.preprocess)
        base = output
    else:
        raise ValueError('ONNX requires --manifest or --inputs; TPU requires --inputs from ONNX run')
    if args.backend == 'onnx':
        if not args.onnx:
            raise ValueError('--onnx is required for explicitly local reference inference')
        session = onnx_session(args.onnx)
        model = {'backend': 'LOCAL ONNX CPU / NOT TPU', 'onnx_sha256': digest(Path(args.onnx).read_bytes())}
    else:
        if not args.program or not args.program.startswith('/auto/cache/') or not args.program.endswith('.tpu') or '?' in args.program:
            raise ValueError('Set an existing absolute worker --program /auto/cache/.../candidate.tpu')
        if args.codec_path:
            sys.path.insert(0, str(Path(args.codec_path).resolve()))
        from gateways.safetyvision.gateway import TPUClient
        remote = TPUClient(args.host, args.port, args.program, args.timeout)
        model = {'backend': 'REMOTE LinQ TPU / PyTask raw=False', 'program': args.program,
                 'program_sha256': args.program_sha256, 'program_hash_verified_remotely': False}
    result_rows, predictions = [], {}
    try:
        for i, entry in enumerate(index['images']):
            tensor = load_tensor(base, entry)
            started = time.perf_counter()
            if remote:
                raw, worker, *_ = remote.infer(tensor)
                worker_ms = float(worker.get('inference_time', 0))*1000
            else:
                raw = session.run(None, {'images': tensor})[0]
                worker_ms = None
            elapsed = (time.perf_counter()-started)*1000
            raw = np.asarray(raw)
            if raw.shape != (1, 17, 8400) or not np.isfinite(raw).all():
                raise ValueError('Nonfinite or unexpected raw model output')
            name = f'raw_{i:04d}.npy'
            np.save(output / name, raw, allow_pickle=False)
            dets = safetyvision_detections(raw, entry['meta'], include_person=True)
            for detection in dets:
                detection['model'] = 'linq-safetyvision-v2' if remote else 'LOCAL-ONNX-safetyvision-v2-NOT-TPU'
            predictions[entry['id']] = dets
            result_rows.append({**entry, 'raw': name, 'raw_sha256': digest(raw.tobytes()),
                                'elapsed_ms': elapsed, 'worker_ms': worker_ms})
            print(f"{args.backend}: {entry['id']}: {len(dets)} detections, {elapsed:.1f}ms", flush=True)
    finally:
        if remote:
            remote.close()
    write(output / 'index.json', {'kind': index['kind'], 'images': result_rows, 'model': model})
    write(output / 'predictions.json', {'kind': index['kind'], 'model': model, 'predictions': predictions})


def compare(args):
    a, b = read(Path(args.reference)/'index.json'), read(Path(args.candidate)/'index.json')
    left = {r['id']: r for r in a['images']}
    right = {r['id']: r for r in b['images']}
    if not left or set(left) != set(right) or len(left) != len(a['images']) or len(right) != len(b['images']):
        raise ValueError('Missing or duplicate paired inputs')
    differences = []
    for image_id, x in left.items():
        y = right[image_id]
        if any(x[k] != y[k] for k in ('tensor_sha256', 'jpeg_sha256', 'meta', 'preprocess')):
            raise ValueError(f'Different input / preprocessing for {image_id}')
        raws = []
        for base, row in ((args.reference, x), (args.candidate, y)):
            raw_path = (Path(base)/row['raw']).resolve()
            if not raw_path.is_relative_to(Path(base).resolve()):
                raise ValueError('Raw path escapes result bundle')
            raw = np.load(raw_path, allow_pickle=False)
            if raw.shape != (1,17,8400) or not np.isfinite(raw).all() or digest(raw.tobytes()) != row['raw_sha256']:
                raise ValueError('Corrupt raw tensor')
            raws.append(raw.astype(np.float64))
        delta = np.abs(raws[0]-raws[1])
        differences.append({'id': image_id, 'coordinate_MAE': float(delta[:, :4].mean()),
                            'score_MAE': float(delta[:, 4:].mean()),
                            'score_max_abs_error': float(delta[:, 4:].max()),
                            'per_class_score_MAE': dict(zip(CLASSES, delta[0,4:].mean(axis=1).tolist()))})
    reference = read(Path(args.reference)/'predictions.json')['predictions']
    candidate = read(Path(args.candidate)/'predictions.json')['predictions']
    result = {'kind': a['kind'], 'note': 'Agreement against ONNX reference, NOT accuracy against ground truth',
              'reference': a.get('model'), 'candidate': b.get('model'), 'raw_differences': differences,
              'detection_agreement': evaluate(reference, candidate)}
    write(args.output, result)
    print(json.dumps(result['detection_agreement'], indent=2))


def calibration(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    index = prepare_inputs(args.manifest, output)
    rows = index['images']
    tags = Counter(t for row in rows for t in row['tags'])
    hashes = {r['jpeg_sha256'] for r in rows}
    tensor_hashes = {r['tensor_sha256'] for r in rows}
    if (len(rows) < args.min_images or len(hashes) != len(rows) or len(tensor_hashes) != len(rows) or
            any(r['split'] != 'calibration' or not r['group'] for r in rows) or CALIBRATION_TAGS-set(tags)):
        raise ValueError('Calibration needs unique images/tensors, calibration split, capture group, and all required tags')
    if args.eval_manifest:
        evaluation_manifest = read(args.eval_manifest)
        base = Path(args.eval_manifest).resolve().parent
        eval_rows = evaluation_manifest['images']
        if not eval_rows or any(not r.get('group') or r.get('split') not in {'validation', 'test'} for r in eval_rows):
            raise ValueError('Held-out images need capture groups and validation/test splits')
        eval_hashes = {digest((base/r['path']).read_bytes()) for r in eval_rows}
        eval_groups = {r['group'] for r in eval_rows}
        if hashes & eval_hashes or {r['group'] for r in rows} & eval_groups:
            raise ValueError('Calibration / evaluation image or capture-group leakage')
    index['coverage_metadata_only'] = True
    index['coverage_tags'] = dict(tags)
    index['heldout_leakage_checked'] = bool(args.eval_manifest)
    write(output/'index.json', index)
    print(f'Prepared {len(rows)} calibration tensors. Coverage tags require manual review: {dict(tags)}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('infer')
    r.add_argument('--backend', choices=['onnx', 'tpu'], required=True)
    group = r.add_mutually_exclusive_group(required=True)
    group.add_argument('--manifest')
    group.add_argument('--inputs')
    r.add_argument('--onnx')
    r.add_argument('--preprocess', choices=['opencv', 'pil'], default='opencv')
    r.add_argument('--program')
    r.add_argument('--program-sha256', help='Operator-supplied hash; not remotely verified')
    r.add_argument('--codec-path', help='Organizer-provided codec directory or zip, if not installed')
    r.add_argument('--host', default='10.0.114.30')
    r.add_argument('--port', type=int, default=5672)
    r.add_argument('--timeout', type=float, default=12)
    r.add_argument('--output', required=True)
    c = sub.add_parser('compare')
    c.add_argument('--reference', required=True)
    c.add_argument('--candidate', required=True)
    c.add_argument('--output', required=True)
    k = sub.add_parser('calibration')
    k.add_argument('--manifest', required=True)
    k.add_argument('--eval-manifest')
    k.add_argument('--min-images', type=int, default=50)
    k.add_argument('--output', required=True)
    args = p.parse_args()
    if args.command == 'infer':
        infer(args)
    elif args.command == 'compare':
        compare(args)
    else:
        if args.min_images < 1:
            p.error('--min-images must be positive')
        calibration(args)


if __name__ == '__main__':
    main()

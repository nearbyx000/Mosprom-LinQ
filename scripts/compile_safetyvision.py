"""Compile a NEW SafetyVision candidate using exact gateway calibration tensors.

Linux + vendor SDK only. API matches the supplied LinQ build_yolo_linq.py.
Does not deploy, replace any program, or change worker configuration.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from scripts.safetyvision_bench import CALIBRATION_TAGS, digest, load_tensor, onnx_session
from scripts.safetyvision_eval import read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--output', required=True, help='NEW build directory; must not already exist')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        p.error('Output already exists; use a new candidate build directory')
    index = read(Path(args.calibration)/'index.json')
    rows = index['images']
    if (len(rows) < 50 or not index.get('heldout_leakage_checked') or
            CALIBRATION_TAGS-set(index.get('coverage_tags', {})) or
            any(r.get('preprocess') != 'opencv' or r.get('split') != 'calibration' or not r.get('group') for r in rows) or
            len({r['tensor_sha256'] for r in rows}) != len(rows)):
        p.error('Need >=50 distinct gateway tensors, coverage tags and held-out leakage check')
    onnx_session(args.onnx)  # Validate class mapping and static model contract.
    samples = np.concatenate([load_tensor(args.calibration, r) for r in rows], axis=0)
    import onnx
    from tpu_framework import RegularModel, Network, TPU_H1M, O1, TpuProgram, compiler
    output.mkdir(parents=True, exist_ok=False)
    model = RegularModel(model=onnx.load(args.onnx))
    sdk_input = model.anchors_mapping['images']
    thresholds = model.calibrate({sdk_input: samples})
    RegularModel.dump_thresholds(str(output/'thresholds.json'), thresholds)
    quantized = model.quantize(thresholds)
    quantized.save(file_dir=str(output), file_name='safetyvision.quant.onnx')
    network, _ = Network.from_quantized_model(quantized)
    network.set_batch(1)
    executable, descriptions = compiler.compile_(TPU_H1M, network, O1)
    program = TpuProgram.from_executable(executable, descriptions)
    target = output/'safetyvision_candidate.tpu'
    program.to_file(str(target))
    if not target.is_file() or not target.stat().st_size:
        raise RuntimeError('Compiler produced no program')
    write(output/'build.json', {'onnx_sha256': digest(Path(args.onnx).read_bytes()),
                               'calibration_index_sha256': digest((Path(args.calibration)/'index.json').read_bytes()),
                               'program_sha256': digest(target.read_bytes()), 'calibration_images': len(rows),
                               'hardware': 'H1M', 'preset': 'O1', 'validated_on_TPU': False})
    print(f'Candidate created: {target}. NOT deployed or validated on TPU.')


if __name__ == '__main__':
    main()

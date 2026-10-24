"""Offline labelled evaluation and deterministic production RuleEngine replay.

Run from repository root: python -m scripts.safetyvision_eval --help
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

from safesight.evaluation import evaluate
from safesight.logic import Detection, RuleEngine
from safesight.safety import SafetyPolicy


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def cases_digest(cases):
    """Stable across Windows/Linux line endings without changing annotations."""
    return hashlib.sha256(json.dumps(cases,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def write(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A report is an artifact: accidentally overwriting a baseline must fail.
    with path.open('x', encoding='utf-8') as out:
        json.dump(doc, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')


def load_engine(path):
    spec = importlib.util.spec_from_file_location('audit_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RuleEngine


def replay(cases, engine_class=RuleEngine, policy=None):
    policy = policy or SafetyPolicy()
    if not cases.get('cases'):
        raise ValueError('Replay requires nonempty labelled cases')
    current = 'safety_policy' in inspect.signature(engine_class).parameters
    expected, predicted = {}, {}
    for case in cases['cases']:
        case_id = case['id']
        if case_id in expected:
            raise ValueError('Duplicate case ID')
        expected[case_id] = case['expected']
        kwargs = {'confirm_frames': 3, 'clear_after': 5}
        if current:
            kwargs['safety_policy'] = policy
        engine = engine_class(**kwargs)
        events = []
        for frame in case['frames']:
            detections = [Detection(d['label'], d['confidence'], tuple(d['box'])) for d in frame['detections']]
            observation = {'safetyvision_observed': frame.get('safetyvision_observed', True),
                           'observed_at': frame['at']} if current else {}
            _, emitted = engine.process('evaluation', detections, [], case.get('width', 640),
                                        case.get('height', 480), **observation)
            events.extend(asdict(event) for event in emitted)
        predicted[case_id] = events
    return {'kind': cases['kind'], 'predictions': predicted,
            'configuration': {'confirm_frames': 3, 'clear_after': 5,
                              'safety_policy': asdict(policy) if current else None},
            'metrics': evaluate(expected, predicted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    r = commands.add_parser('replay')
    r.add_argument('--cases', required=True)
    r.add_argument('--engine-file', help='Explicit frozen pre-patch module; offline comparison only')
    r.add_argument('--output', required=True)
    s = commands.add_parser('score')
    s.add_argument('--manifest', required=True)
    s.add_argument('--predictions', required=True)
    s.add_argument('--iou', type=float, default=.5)
    s.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'replay':
        engine_class = RuleEngine
        if args.engine_file:
            engine_class = load_engine(args.engine_file)
        result = replay(read(args.cases), engine_class, SafetyPolicy.from_env())
        result['cases_sha256'] = cases_digest(read(args.cases))
        result['cases_hash_format'] = 'canonical_json_v1'
        if not args.engine_file:
            result['source_sha256'] = {name: hashlib.sha256(Path(name).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
                                       for name in ('safesight/logic.py', 'safesight/safety.py')}
            result['source_hash_format'] = 'utf8_lf_v1'
        if args.engine_file:
            result['engine_sha256'] = hashlib.sha256(Path(args.engine_file).read_bytes()).hexdigest()
    else:
        manifest = read(args.manifest)
        rows = manifest['images']
        if not rows or any('annotations' not in row for row in rows):
            raise ValueError('Ground truth required for every image, including explicit [] for negative scenes')
        expected = {row['id']: row['annotations'] for row in rows}
        if len(expected) != len(rows):
            raise ValueError('Duplicate image ID')
        predictions = read(args.predictions)
        result = {'kind': manifest['kind'],
                  'metrics': evaluate(expected, predictions['predictions'], args.iou,
                                      manifest.get('duration_seconds'))}
    write(args.output, result)
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()

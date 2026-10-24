"""Exploratory candidate comparison on the frozen internet mini-set; NOT TPU.

This is development/selection evidence, not an independent final test. Weights
must be supplied explicitly. Nothing is downloaded or deployed by this script.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gateways.safetyvision.preprocessing import jpeg_to_tensor
from scripts.research.helmet import helmet_detections,validate_helmet_session
from scripts.internet_eval20 import montage,task_scores,presence_scores
from scripts.safetyvision_bench import digest
from scripts.safetyvision_eval import read,write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--data',type=Path,default=Path('data/internet-eval-20'))
    parser.add_argument('--manifest',type=Path,default=Path('data/internet-eval-20/results-reviewed/manifest_reviewed.json'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(args.model,sess_options=options,providers=['CPUExecutionProvider'])
    validate_helmet_session(session)
    manifest = read(args.manifest)
    args.output.mkdir(parents=True,exist_ok=False)
    predictions = {}
    for i,row in enumerate(manifest['images']):
        jpeg = (args.data/row['path']).read_bytes()
        if digest(jpeg)!=row['jpeg_sha256']:
            raise ValueError('Frozen image has changed')
        tensor,meta = jpeg_to_tensor(jpeg,backend='opencv')
        raw = session.run(None,{'images':tensor})[0]
        np.save(args.output/f'raw_{i:02d}.npy',raw,allow_pickle=False)
        predictions[row['id']] = helmet_detections(raw,meta)
        for d in predictions[row['id']]:
            d['model'] = 'LOCAL-ONNX-HELMET-CANDIDATE-NOT-TPU'
    report = {'backend':'LOCAL ONNX CPU / NOT TPU','status':'exploratory candidate, not deployed',
              'model_sha256':digest(Path(args.model).read_bytes()),
              'manifest_sha256':digest(args.manifest.read_bytes()),
              'thresholds':{'hardhat':.40,'no_helmet':.45},
              'ppe_10':task_scores(manifest['images'],predictions,'ppe',{'hardhat','no_helmet'},'ppe'),
              'no_helmet_presence_10':presence_scores(manifest['images'],predictions,'ppe','no_helmet','ppe'),
              'licensing':'Verify model-weight rights before redistribution or commercial rollout; no weights shipped in Git'}
    write(args.output/'report.json',report)
    write(args.output/'predictions.json',{'backend':report['backend'],'predictions':predictions})
    montage(manifest['images'],args.data,args.output/'candidate.jpg',predictions)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

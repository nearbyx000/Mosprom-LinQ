"""Exploratory LOCAL ONNX ROI comparison; never selected by production modes.

Uses frozen YOLOv8n predictions, not ground-truth person boxes. Compares two
fixed hypotheses without threshold tuning. The same diagnostic set is NOT an
independent acceptance test. No weights, routes or TPU programs are changed.
"""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from gateways.safetyvision.decoder import safetyvision_detections, _nms
from gateways.safetyvision.preprocessing import jpeg_to_tensor
from scripts.internet_eval20 import task_scores, presence_scores, montage
from scripts.safetyvision_bench import onnx_session, digest
from scripts.safetyvision_eval import read, write


def crop_boxes(people, width, height, head):
    for person in people[:10]:
        if person['class_name']!='person' or person['confidence']<.45:
            continue
        x1,y1,x2,y2 = person['bbox']
        w,h = x2-x1,y2-y1
        if min(w,h)<=0:
            continue
        bottom = y1+.45*h if head else y2
        box = (max(0,int(x1-.1*w)),max(0,int(y1-.05*h)),
               min(width,int(np.ceil(x2+.1*w))),min(height,int(np.ceil(bottom+.05*h))))
        if min(box[2]-box[0],box[3]-box[1])>=16:
            yield box


def merge(detections):
    result = []
    for label in ('hardhat','no_helmet'):
        objects = [d for d in detections if d['class_name']==label]
        if objects:
            boxes = np.array([d['bbox'] for d in objects],dtype=float)
            scores = np.array([d['confidence'] for d in objects])
            result.extend(objects[i] for i in _nms(boxes,scores,.45,100))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--data',type=Path,default=Path('data/internet-eval-20'))
    parser.add_argument('--baseline',type=Path,default=Path('data/internet-eval-20/results-reviewed'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    session = onnx_session(args.model)
    manifest = read(args.baseline/'manifest_reviewed.json')
    primary = read(args.baseline/'YOLOv8n_predictions.json')['predictions']
    original = read(args.baseline/'SafetyVision_predictions.json')['predictions']
    if read(args.baseline/'report.json')['model_sha256']['SafetyVision'] != digest(Path(args.model).read_bytes()):
        raise ValueError('ROI experiment must use the same frozen SafetyVision model')
    args.output.mkdir(parents=True,exist_ok=False)
    report = {'backend':'LOCAL ONNX CPU / NOT TPU','status':'exploratory ROI; not deployed',
              'model_sha256':digest(Path(args.model).read_bytes()),
              'manifest_sha256':digest((args.baseline/'manifest_reviewed.json').read_bytes()),
              'primary_predictions_sha256':digest((args.baseline/'YOLOv8n_predictions.json').read_bytes()),
              'protocol':{'person_confidence':.45,'max_people':10,'head_fraction':.45,
                          'horizontal_padding':.10,'vertical_padding':.05,'min_crop_side':16,
                          'jpeg_quality':95,'hardhat_confidence':.40,'no_helmet_confidence':.45},'variants':{}}
    for variant in ('body','head'):
        predictions,regions = {},{}
        for i,row in enumerate(manifest['images']):
            jpeg = (args.data/row['path']).read_bytes()
            if digest(jpeg)!=row['jpeg_sha256']:
                raise ValueError('Frozen image changed')
            image = cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
            h,w = image.shape[:2]
            objects = copy.deepcopy(original[row['id']])
            regions[row['id']] = list(crop_boxes(primary[row['id']],w,h,variant=='head'))
            for n,(x1,y1,x2,y2) in enumerate(regions[row['id']]):
                ok,encoded = cv2.imencode('.jpg',image[y1:y2,x1:x2],[cv2.IMWRITE_JPEG_QUALITY,95])
                if not ok:
                    raise ValueError('Crop JPEG encode failed')
                tensor,meta = jpeg_to_tensor(encoded.tobytes(),backend='opencv')
                raw = session.run(None,{'images':tensor})[0]
                np.save(args.output/f'{variant}_{i:02d}_{n:02d}.npy',raw,allow_pickle=False)
                for d in safetyvision_detections(raw,meta,class_thresholds={'no_helmet':.45,'fall_detected':.45}):
                    if d['class_name'] in {'hardhat','no_helmet'}:
                        a,b,c,e = d['bbox']
                        d['bbox'] = [a+x1,b+y1,c+x1,e+y1]
                        objects.append(d)
            predictions[row['id']] = merge(objects)
            for d in predictions[row['id']]:
                d['model'] = f'LOCAL-ONNX-SV-{variant}-ROI-NOT-TPU'
        report['variants'][variant] = {
            'ppe_10':task_scores(manifest['images'],predictions,'ppe',{'hardhat','no_helmet'},'ppe'),
            'no_helmet_presence_10':presence_scores(manifest['images'],predictions,'ppe','no_helmet','ppe'),
            'crops':sum(len(v) for v in regions.values()),
            'note':'Full-frame predictions plus per-person ROI; class-aware NMS, no extra conflict/temporal filtering'}
        write(args.output/f'{variant}_predictions.json',{'backend':report['backend'],'predictions':predictions,'regions':regions})
        montage(manifest['images'],args.data,args.output/f'{variant}.jpg',predictions)
    write(args.output/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

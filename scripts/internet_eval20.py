"""Reproducible external-image diagnostic, explicitly LOCAL ONNX / NOT TPU.

20 shared images: 10 PPE test (5 with no-helmet, 5 helmet-only), 10 fall
validation (5 fall, 3 walking, 2 sitting). Selection precedes all inference.
No claim that these public sources are absent from model training data.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import cv2
import numpy as np

from gateways.safetyvision.decoder import safetyvision_detections
from gateways.safetyvision.preprocessing import jpeg_to_tensor, person_detections
from safesight.evaluation import evaluate, rates
from scripts.safetyvision_bench import digest, onnx_session
from scripts.safetyvision_eval import read, write

SOURCES = {
    'ppe': {'url':'https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip',
            'page':'https://docs.ultralytics.com/datasets/detect/construction-ppe', 'split':'test'},
    'fall': {'url':'https://www.kaggle.com/api/v1/datasets/download/uttejkumarkandagatla/fall-detection-dataset',
             'page':'https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset', 'split':'val'},
}
SEED = 'safesight-internet20-v1'


def download(base):
    base.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        target = base/f'{name}.zip'
        if not target.exists():
            request = urllib.request.Request(source['url'],headers={'User-Agent':'SafeSight-evaluation/1.0'})
            with urllib.request.urlopen(request,timeout=60) as response, target.open('xb') as out:
                while chunk := response.read(1024*1024):
                    out.write(chunk)
        if not zipfile.is_zipfile(target):
            raise ValueError(f'Invalid archive: {target}')


def prepare(base):
    if (base/'manifest.json').exists():
        raise FileExistsError('Selection already frozen; use another dataset directory')
    download(base)
    images = base/'images'
    images.mkdir(exist_ok=True)
    rows = []
    for source in SOURCES:
        with zipfile.ZipFile(base/f'{source}.zip') as archive:
            paths = {p for p in archive.namelist() if p.lower().endswith(('.jpg','.jpeg','.png'))}
            labels = [p for p in archive.namelist() if '/labels/' in '/'+p and
                      f"/{SOURCES[source]['split']}/" in p and p.endswith('.txt')]
            labels.sort(key=lambda p:digest(f'{SEED}:{source}:{p}'.encode()))
            parsed = {p:[list(map(float,line.split())) for line in archive.read(p).decode().splitlines()
                         if line.strip()] for p in labels}
            classes = {p:{int(line[0]) for line in lines} for p,lines in parsed.items()}
            buckets = [(5,lambda ids:7 in ids),(5,lambda ids:0 in ids and 7 not in ids)] if source=='ppe' else [
                (5,lambda ids:0 in ids),(3,lambda ids:1 in ids and 0 not in ids),
                (2,lambda ids:2 in ids and 0 not in ids)]
            used = set()
            for count, condition in buckets:
                chosen = [p for p in labels if p not in used and condition(classes[p])][:count]
                if len(chosen) != count:
                    raise ValueError('Insufficient samples in prespecified bucket')
                for label_path in chosen:
                    used.add(label_path)
                    stem = ('/'+label_path).replace('/labels/','/images/').lstrip('/').rsplit('.',1)[0]
                    matches = sorted(p for p in paths if p.rsplit('.',1)[0]==stem)
                    if len(matches)!=1:
                        raise ValueError(f'Image pair missing: {label_path}')
                    image_path = matches[0]
                    original = archive.read(image_path)
                    bgr = cv2.imdecode(np.frombuffer(original,np.uint8),cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ValueError('Image decode failed')
                    height,width = bgr.shape[:2]
                    image_id = f'{source}-{Path(image_path).stem}'
                    jpeg = original
                    if Path(image_path).suffix.lower() not in {'.jpg','.jpeg'}:
                        ok,encoded = cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
                        if not ok:
                            raise ValueError('JPEG conversion failed')
                        jpeg = encoded.tobytes()
                    (images/f'{image_id}.jpg').write_bytes(jpeg)
                    truth = {'person':[], 'ppe':[], 'fall':[]}
                    mapping = {0:'hardhat',7:'no_helmet',6:'person'} if source=='ppe' else {
                        0:'fall_detected',1:'person',2:'person'}
                    for line in parsed[label_path]:
                        cls,cx,cy,w,h = line[:5]
                        box = [(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height]
                        box = [max(0,min(v,width if i%2==0 else height)) for i,v in enumerate(box)]
                        label = mapping.get(int(cls))
                        if label is None:
                            continue
                        obj = {'label':label,'box':box}
                        if label=='person':
                            truth['person'].append(obj)
                        elif label in {'hardhat','no_helmet'}:
                            truth['ppe'].append(obj)
                        else:
                            truth['fall'].append(obj)
                            truth['person'].append({'label':'person','box':box})
                    rows.append({'id':image_id,'path':f'images/{image_id}.jpg','source':source,
                                 'source_image':image_path,'source_label':label_path,
                                 'original_sha256':digest(original),'jpeg_sha256':digest(jpeg),
                                 'label_sha256':digest(archive.read(label_path)),
                                 'width':width,'height':height,'truth':truth})
    if len(rows)!=20 or len({r['jpeg_sha256'] for r in rows})!=20:
        raise ValueError('Expected exactly 20 unique images')
    write(base/'manifest.json',{'kind':'public-internet-mini-diagnostic-NOT-independent-heldout',
                               'selection_seed':SEED,'sources':SOURCES,
                               'archive_sha256':{name:digest((base/f'{name}.zip').read_bytes()) for name in SOURCES},
                               'images':rows})
    montage(rows,base,base/'ground_truth.jpg',None)
    print('Frozen 20 images before inference:',base/'manifest.json')


def montage(rows,base,target,predictions):
    tiles = []
    colors = {'person':(0,220,220),'no_helmet':(0,0,255),'hardhat':(0,210,0),'fall_detected':(220,0,220)}
    for row in rows:
        image = cv2.imdecode(np.frombuffer((base/row['path']).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        scale = min(480/image.shape[1],320/image.shape[0])
        image = cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
        tile = np.full((360,480,3),35,np.uint8)
        tile[35:35+image.shape[0],:image.shape[1]] = image
        objects = predictions[row['id']] if predictions else [d for group in row['truth'].values() for d in group]
        for obj in objects:
            label = obj.get('class_name',obj.get('label'))
            if label not in colors:
                continue
            box = obj.get('bbox',obj.get('box'))
            x1,y1,x2,y2 = [int(v*scale) for v in box]
            cv2.rectangle(tile,(x1,y1+35),(x2,y2+35),colors[label],2)
            cv2.putText(tile,label,(x1,max(47,y1+32)),cv2.FONT_HERSHEY_SIMPLEX,.38,colors[label],1)
        cv2.putText(tile,row['id'],(5,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
        tiles.append(tile)
    grid = np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
    ok, encoded = cv2.imencode('.jpg',grid,[cv2.IMWRITE_JPEG_QUALITY,90])
    if not ok:
        raise ValueError('Montage encode failed')
    target.write_bytes(encoded.tobytes())


def task_scores(rows,predictions,task,labels,source=None):
    selected = [r for r in rows if source is None or r['source']==source]
    truth = {r['id']:r['truth'][task] for r in selected}
    preds = {r['id']:[p for p in predictions[r['id']] if p.get('class_name',p.get('label')) in labels]
             for r in selected}
    return evaluate(truth,preds)


def presence_scores(rows,predictions,task,label,source=None):
    """Image-level presence, separate from box localization and video alarms."""
    tp = fp = fn = tn = 0
    for row in rows:
        if source is not None and row['source']!=source:
            continue
        actual = any(d['label']==label for d in row['truth'][task])
        found = any(d.get('class_name',d.get('label'))==label for d in predictions[row['id']])
        tp += actual and found
        fp += not actual and found
        fn += actual and not found
        tn += not actual and not found
    return {**rates(tp,fp,fn),'TN':tn,'images':tp+fp+fn+tn,
            'note':'Image-level class presence; no IoU requirement; not temporal alerts'}


def run(base,args):
    import onnxruntime as ort
    manifest = read(base/'manifest.json')
    rows = manifest['images']
    review = read(args.review)
    reviewed_rows = copy.deepcopy(rows)
    for row in reviewed_rows:
        for task, objects in review['replacements'].get(row['id'],{}).items():
            if task in row['truth']:
                row['truth'][task] = objects
    output = base/args.output
    output.mkdir(exist_ok=False)
    sv = onnx_session(args.sv_model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    yolo = ort.InferenceSession(args.yolo_model,sess_options=options,providers=['CPUExecutionProvider'])
    if yolo.get_inputs()[0].shape != [1,3,640,640] or yolo.get_outputs()[0].shape != [1,84,8400]:
        raise ValueError('Expected the working 640 YOLOv8n model contract')
    legacy = None
    if args.legacy_decoder:
        spec = importlib.util.spec_from_file_location('legacy_sv_decoder',args.legacy_decoder)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
    predictions = {'SafetyVision':{},'YOLOv8n':{}}
    if legacy:
        predictions['SafetyVision_before'] = {}
    timings = []
    for index,row in enumerate(rows):
        jpeg = (base/row['path']).read_bytes()
        if digest(jpeg)!=row['jpeg_sha256']:
            raise ValueError('Frozen image changed')
        tensors = {backend:jpeg_to_tensor(jpeg,backend=backend) for backend in ('opencv','pil')}
        started = time.perf_counter()
        raw_sv = sv.run(None,{'images':tensors['opencv'][0]})[0]
        elapsed_sv = (time.perf_counter()-started)*1000
        started = time.perf_counter()
        raw_yolo = yolo.run(None,{'images':tensors['pil'][0]})[0]
        elapsed_yolo = (time.perf_counter()-started)*1000
        np.save(output/f'sv_{index:02d}.npy',raw_sv,allow_pickle=False)
        np.save(output/f'yolo_{index:02d}.npy',raw_yolo,allow_pickle=False)
        # Fixed before seeing results: 0.45 priority violations, 0.40 positive helmet;
        # primary JPEG gateway's original person threshold is 0.25.
        predictions['SafetyVision'][row['id']] = safetyvision_detections(
            raw_sv,tensors['opencv'][1],include_person=True,
            class_thresholds={'no_helmet':.45,'fall_detected':.45})
        predictions['YOLOv8n'][row['id']] = person_detections(raw_yolo,tensors['pil'][1],confidence=.25)
        if legacy:
            before = sv.run(None,{'images':tensors['pil'][0]})[0]
            np.save(output/f'sv_before_{index:02d}.npy',before,allow_pickle=False)
            predictions['SafetyVision_before'][row['id']] = [p for p in legacy.safetyvision_detections(
                before,tensors['pil'][1],include_person=True) if p['class_name'] not in {'no_helmet','fall_detected'} or p['confidence']>=.45]
        timings.append({'id':row['id'],'SV_local_onnx_ms':elapsed_sv,'YOLO_local_onnx_ms':elapsed_yolo})
        print(index+1,row['id'],len(predictions['SafetyVision'][row['id']]),len(predictions['YOLOv8n'][row['id']]),flush=True)
    report = {'kind':manifest['kind'],'backend':'LOCAL ONNX CPU / NOT LinQ TPU',
              'images':len(rows),'selection_seed':manifest['selection_seed'],
              'manifest_sha256':digest((base/'manifest.json').read_bytes()),
              'model_sha256':{'SafetyVision':digest(Path(args.sv_model).read_bytes()),'YOLOv8n':digest(Path(args.yolo_model).read_bytes())},
              'thresholds':{'no_helmet':.45,'fall_detected':.45,'hardhat':.40,'YOLOv8n_person':.25},
              'metrics':{},'timings':timings,
              'presence_metrics':{},
              'review_sha256':digest(Path(args.review).read_bytes()),
              'metrics_publisher':{},
              'notes':['Single-image detections, not temporally confirmed RuleEngine alarms.',
                       'PPE evaluated on 10 PPE images; fall on 10 fall images; person inference on all 20, reviewed metrics on 19.',
                       'Public dataset training overlap with SafetyVision/COCO is unknown; small stratified diagnostic only.',
                       'Publisher annotations and pre-inference visual corrections are scored separately.',
                       'Reviewed person metrics exclude ppe-image1270 because background people lack complete labels.']}
    for name,preds in predictions.items():
        for objects in preds.values():
            for obj in objects:
                obj['model'] = f'LOCAL-ONNX-{name}-NOT-TPU'
        person_rows = [r for r in reviewed_rows if r['id'] not in review['exclude_person_metric']]
        report['metrics'][name] = {'person_reviewed_19':task_scores(person_rows,preds,'person',{'person'})}
        report['presence_metrics'][name] = {'person_reviewed_19':presence_scores(person_rows,preds,'person','person')}
        report['metrics_publisher'][name] = {'person_20':task_scores(rows,preds,'person',{'person'})}
        if name.startswith('SafetyVision'):
            report['metrics'][name]['ppe_10'] = task_scores(reviewed_rows,preds,'ppe',{'hardhat','no_helmet'},'ppe')
            report['metrics'][name]['fall_10'] = task_scores(reviewed_rows,preds,'fall',{'fall_detected'},'fall')
            report['metrics_publisher'][name]['ppe_10'] = task_scores(rows,preds,'ppe',{'hardhat','no_helmet'},'ppe')
            report['presence_metrics'][name]['no_helmet_10'] = presence_scores(reviewed_rows,preds,'ppe','no_helmet','ppe')
            report['presence_metrics'][name]['fall_10'] = presence_scores(reviewed_rows,preds,'fall','fall_detected','fall')
        write(output/f'{name}_predictions.json',{'backend':report['backend'],'predictions':preds})
        montage(rows,base,output/f'{name}.jpg',preds)
    montage(reviewed_rows,base,output/'ground_truth_reviewed.jpg',None)
    write(output/'manifest_reviewed.json',{**manifest,'images':reviewed_rows,'review':review})
    if legacy:
        report['legacy_decoder_sha256'] = digest(Path(args.legacy_decoder).read_bytes())
    write(output/'report.json',report)
    print(json.dumps({'metrics':report['metrics'],'presence':report['presence_metrics']},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--data',type=Path,default=Path('data/internet-eval-20'))
    parser.add_argument('--sv-model',default='safetyvision_v2_640_op11.onnx')
    parser.add_argument('--yolo-model',default='models/yolov8n.onnx')
    parser.add_argument('--legacy-decoder')
    parser.add_argument('--review',default='config/internet_eval20_review.json')
    parser.add_argument('--output',default='results',help='New result subdirectory under --data')
    args = parser.parse_args()
    if args.command=='prepare':
        prepare(args.data)
    else:
        run(args.data,args)


if __name__=='__main__':
    main()

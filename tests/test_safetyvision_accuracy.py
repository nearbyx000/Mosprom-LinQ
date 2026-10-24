import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gateways.safetyvision.decoder import NORMALIZED, safetyvision_detections
from gateways.safetyvision.preprocessing import jpeg_to_tensor
from safesight.evaluation import evaluate
from safesight.logic import Detection, RuleEngine
from safesight.safety import SafetyPolicy
from scripts.safetyvision_eval import cases_digest, load_engine, read, replay
from scripts.safetyvision_bench import CALIBRATION_TAGS, calibration, compare, load_tensor, digest


FIXTURES = Path(__file__).parent/'fixtures'


def test_regression_cases_use_production_rules_and_frozen_prepatch_predictions():
    cases = read(FIXTURES/'safetyvision_rule_cases.json')
    before = read(FIXTURES/'safetyvision_baseline_events.json')
    after = replay(cases)
    expected = {c['id']: c['expected'] for c in cases['cases']}
    assert evaluate(expected, before['predictions']) == before['metrics']
    replayed = replay(cases, load_engine(FIXTURES/'baseline_logic.py'))
    assert cases_digest(replayed['predictions']) == cases_digest(before['predictions'])
    assert replayed['metrics'] == before['metrics']
    assert digest((FIXTURES/'baseline_logic.py').read_bytes()) == before['engine_sha256']
    assert before['cases_sha256'] == cases_digest(cases)
    assert before['metrics']['total'] == dict(TP=6, FP=7, FN=2, precision=6/13, recall=.75, F1=12/21)
    assert after['metrics']['total'] == dict(TP=8, FP=0, FN=0, precision=1, recall=1, F1=1)
    for case in cases['cases']:
        assert len(after['predictions'][case['id']]) == len(case['expected']), case['id']


def test_single_best_class_and_ambiguous_negative_anchor():
    raw = np.zeros((1,17,8400), np.float32)
    raw[0,:4,0] = [100,100,50,50]
    raw[0,7,0], raw[0,11,0] = .8, .83  # Hardhat and NO-Hardhat near tie.
    assert safetyvision_detections(raw, (640,640,1,0,0)) == []
    raw[0,11,0] = .95
    assert [d['class_name'] for d in safetyvision_detections(raw, (640,640,1,0,0))] == ['no_helmet']
    assert safetyvision_detections(raw.transpose(0,2,1), (640,640,1,0,0))[0]['bbox'] == [75,75,125,125]


def test_decoder_thresholds_nms_bad_values_and_rounded_empty_boxes():
    raw = np.zeros((1,17,8400), np.float32)
    raw[0,:4,:2] = np.array([[100,100,50,50],[100,100,50,50]]).T
    raw[0,11,:2] = [.8,.7]
    assert len(safetyvision_detections(raw, (640,640,1,0,0))) == 1
    assert safetyvision_detections(raw, (640,640,1,0,0), class_thresholds={'no_helmet':.85}) == []
    raw[0,11,:2] = [np.nan, 1.1]
    assert safetyvision_detections(raw, (640,640,1,0,0)) == []
    raw[0,:4,0], raw[0,11,0] = [1,1,.01,.01], .9
    assert safetyvision_detections(raw, (640,640,1,0,0)) == []
    for meta in ((640,640,np.nan,0,0), (640,640,1,np.inf,0)):
        with pytest.raises(ValueError):
            safetyvision_detections(raw, meta)


def test_safetyvision_preprocessing_matches_compiler_and_keeps_pil_default():
    image = np.random.default_rng(7).integers(0,256,(713,951,3),dtype=np.uint8)
    ok, jpg = cv2.imencode('.jpg', image)
    assert ok
    tensor, meta = jpeg_to_tensor(jpg.tobytes(), backend='opencv')
    w,h,scale,left,top = meta
    decoded = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    resized = cv2.resize(decoded, (round(w*scale),round(h*scale)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640,640,3),114,np.uint8)
    canvas[top:top+resized.shape[0],left:left+resized.shape[1]] = resized
    expected = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2,0,1)[None].astype(np.float32)/255
    np.testing.assert_array_equal(tensor, expected)
    assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
    np.testing.assert_array_equal(jpeg_to_tensor(jpg.tobytes())[0], jpeg_to_tensor(jpg.tobytes(),backend='pil')[0])
    assert np.abs(tensor-jpeg_to_tensor(jpg.tobytes())[0]).mean() > 0


def test_non_square_coordinate_roundtrip():
    image = np.zeros((333,997,3),np.uint8)
    _, jpeg = cv2.imencode('.jpg',image)
    _, (w,h,r,left,top) = jpeg_to_tensor(jpeg.tobytes(),backend='opencv')
    box = (80,20,900,300)
    raw = np.zeros((1,17,8400),np.float32)
    raw[0,:4,0] = [(box[0]+box[2])/2*r+left,(box[1]+box[3])/2*r+top,
                    (box[2]-box[0])*r,(box[3]-box[1])*r]
    raw[0,4,0] = .9
    assert safetyvision_detections(raw,(w,h,r,left,top))[0]['bbox'] == list(box)


def test_three_of_four_safety_observations_tolerate_one_miss():
    engine = RuleEngine()
    for at, present in enumerate((True, True, False)):
        _, events = engine.process('c', [Detection('fall_detected',.9,(0,0,100,50))] if present else [],
                                   [],640,480, observed_at=at*.7)
        assert not events
    _, events = engine.process('c', [Detection('fall_detected',.9,(0,0,100,50))],[],640,480,observed_at=2.1)
    assert len(events) == 1


def test_two_of_four_safety_observations_cannot_confirm():
    engine = RuleEngine()
    for at, present in enumerate((True, False, True, False, True, False)):
        _, events = engine.process('c', [Detection('fall_detected',.9,(0,0,100,50))] if present else [],
                                   [],640,480,observed_at=at*.7)
        assert not events


def test_primary_traffic_cannot_bridge_long_safety_gap():
    engine = RuleEngine()
    fall = lambda: [Detection('fall_detected',.9,(0,0,100,50))]
    for t in (0,.7):
        engine.process('c',fall(),[],640,480, observed_at=t)
    for t in (1.4,2.1,2.8,3.5):
        engine.process('c',[],[],640,480, observed_at=t, safetyvision_observed=False)
    assert engine.process('c',fall(),[],640,480, observed_at=4.2)[1] == []


def test_invalid_policy_rejected_and_threshold_override(monkeypatch):
    monkeypatch.setenv('SAFESIGHT_SV_THRESHOLDS', '{"no_helmet": 0.7}')
    assert SafetyPolicy.from_env().thresholds['no_helmet'] == .7
    for value in ('{"unknown":0.4}', '{"no_helmet":true}', '{"no_helmet":NaN}', '[]'):
        monkeypatch.setenv('SAFESIGHT_SV_THRESHOLDS', value)
        with pytest.raises(ValueError):
            SafetyPolicy.from_env()


def test_metrics_count_duplicates_wrong_classes_and_missing_predictions():
    obj = {'label':'no_helmet','box':[0,0,10,10]}
    result = evaluate({'a':[obj],'b':[]},{'a':[obj,obj],'b':[obj]},duration_seconds=3600)
    assert result['total']['TP'] == 1 and result['total']['FP'] == 2
    assert result['false_alarms_per_hour'] == 2
    result = evaluate({'a':[obj]},{'a':[dict(obj,label='hardhat')]})
    assert result['total']['FP'] == result['total']['FN'] == 1
    with pytest.raises(ValueError):
        evaluate({'a':[obj]}, {})
    assert evaluate({'a':[]},{'a':[]})['total']['recall'] is None


def test_corrupted_input_tensor_rejected(tmp_path):
    tensor = np.zeros((1,3,640,640),np.float32)
    np.save(tmp_path/'input.npy',tensor)
    entry = {'tensor':'input.npy','tensor_sha256':digest(tensor.tobytes())}
    assert load_tensor(tmp_path,entry).shape == tensor.shape
    entry['tensor_sha256'] = 'wrong'
    with pytest.raises(ValueError):
        load_tensor(tmp_path,entry)


def test_quantization_comparison_rejects_different_images(tmp_path):
    for name, sha in (('a','first'),('b','second')):
        base = tmp_path/name
        base.mkdir()
        (base/'index.json').write_text(json.dumps({'kind':'test','images':[
            {'id':'i','tensor_sha256':sha,'jpeg_sha256':'j','meta':[],'preprocess':'opencv'}]}))
    with pytest.raises(ValueError, match='Different input'):
        compare(SimpleNamespace(reference=tmp_path/'a',candidate=tmp_path/'b',output=tmp_path/'out.json'))


def test_paired_comparison_checks_hashes_and_reports_score_drift(tmp_path):
    obj = {'label':'no_helmet','box':[0,0,10,10],'confidence':.8}
    for name, score in (('a',.9),('b',.8)):
        base = tmp_path/name
        base.mkdir()
        raw = np.zeros((1,17,8400),np.float32)
        raw[0,11,0] = score
        np.save(base/'raw.npy',raw)
        row = {'id':'i','tensor_sha256':'same','jpeg_sha256':'same',
               'meta':[640,480,1,0,80],'preprocess':'opencv','raw':'raw.npy','raw_sha256':digest(raw.tobytes())}
        (base/'index.json').write_text(json.dumps({'kind':'test','images':[row]}))
        (base/'predictions.json').write_text(json.dumps({'predictions':{'i':[obj]}}))
    args = SimpleNamespace(reference=tmp_path/'a',candidate=tmp_path/'b',output=tmp_path/'out.json')
    compare(args)
    report = read(args.output)
    assert report['detection_agreement']['total']['TP'] == 1
    assert report['raw_differences'][0]['score_max_abs_error'] == pytest.approx(.1)
    np.save(tmp_path/'b/raw.npy',np.ones((1,17,8400),np.float32))
    args.output = tmp_path/'corrupted.json'
    with pytest.raises(ValueError,match='Corrupt'):
        compare(args)


def test_calibration_rejects_duplicates_and_capture_group_leakage(tmp_path):
    for i in range(2):
        _, jpeg = cv2.imencode('.jpg',np.full((30,40,3),50+i*50,np.uint8))
        (tmp_path/f'{i}.jpg').write_bytes(jpeg.tobytes())
    row = {'id':'cal','path':'0.jpg','split':'calibration','group':'same-session','tags':sorted(CALIBRATION_TAGS)}
    manifest = tmp_path/'cal.json'
    manifest.write_text(json.dumps({'kind':'test','images':[row,dict(row,id='duplicate')]}))
    args = SimpleNamespace(manifest=manifest,output=tmp_path/'duplicate',min_images=1,eval_manifest=None)
    with pytest.raises(ValueError,match='Calibration needs unique'):
        calibration(args)
    manifest.write_text(json.dumps({'kind':'test','images':[row]}))
    heldout = tmp_path/'eval.json'
    heldout.write_text(json.dumps({'images':[{'id':'test','path':'1.jpg','group':'same-session','split':'test'}]}))
    args.output, args.eval_manifest = tmp_path/'leakage',heldout
    with pytest.raises(ValueError,match='leakage'):
        calibration(args)

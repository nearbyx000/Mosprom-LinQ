import numpy as np

from scripts.research.helmet import helmet_detections


def test_explicit_candidate_labels_remap_without_objectness_or_absence_rule():
    raw = np.zeros((1,6,8400),np.float32)
    assert helmet_detections(raw,(640,640,1,0,0)) == []
    raw[0,:4,0] = [100,100,50,50]
    raw[0,5,0] = .9
    assert helmet_detections(raw,(640,640,1,0,0))[0]['class_name'] == 'no_helmet'
    raw[0,4,0],raw[0,5,0] = .95,.1
    assert helmet_detections(raw,(640,640,1,0,0))[0]['class_name'] == 'hardhat'

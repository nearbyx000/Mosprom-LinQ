import unittest
import numpy as np
from gateways.safetyvision.decoder import CLASSES, safetyvision_detections

class TestSV(unittest.TestCase):
    def test_class_mapping(self):
        self.assertEqual(len(CLASSES),13)
        self.assertEqual(CLASSES[0],'Fall-Detected')
        self.assertEqual(CLASSES[7],'NO-Hardhat')
    def test_fall_and_hardhat(self):
        raw=np.zeros((1,17,8400), dtype=np.float32)
        # anchor 0: fall, box in letterbox coords for original 640x480 with top=80
        raw[0,:4,0]=[300,310,100,180]
        raw[0,4+0,0]=0.93
        raw[0,:4,1]=[100,200,60,70]
        raw[0,4+7,1]=0.88
        raw[0,:4,2]=[300,310,100,180]
        raw[0,4+0,2]=0.85
        dets=safetyvision_detections(raw,(640,480,1,0,80))
        self.assertEqual([d['class_name'] for d in dets],['fall_detected','no_helmet'])
        self.assertEqual(dets[0]['bbox'],[250,140,350,320])
        self.assertEqual(dets[1]['bbox'],[70,85,130,155])
    def test_person_excluded_default(self):
        raw=np.zeros((1,17,8400),dtype=np.float32)
        raw[0,:4,0]=[300,300,100,200]
        raw[0,4+11,0]=.99
        self.assertEqual(safetyvision_detections(raw,(640,640,1,0,0)),[])
        self.assertEqual(safetyvision_detections(raw,(640,640,1,0,0),include_person=True)[0]['class_name'],'person')
    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            safetyvision_detections(np.zeros((1,84,8400)),(640,640,1,0,0))
    def test_negative(self):
        self.assertEqual(safetyvision_detections(np.zeros((1,17,8400)),(640,640,1,0,0)),[])
if __name__=='__main__':unittest.main()

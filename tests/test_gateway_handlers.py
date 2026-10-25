from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gateways.safetyvision.gateway import handle_jpeg as sv_handle
from gateways.yolov8n.gateway import handle_jpeg as primary_handle


class FakeTPU:
    def __init__(self,channels):
        self.channels = channels
        self.calls = []

    def infer(self,tensor):
        self.calls.append(tensor)
        raw = np.zeros((1,self.channels,8400),np.float32)
        raw[0,:4,0] = [100,180,40,60]
        raw[0,11 if self.channels==17 else 4,0] = .9
        return raw,{'inference_time':.002},1.,100,100


def jpeg():
    image = np.zeros((480,640,3),np.uint8)
    image[:,:,0] = 255
    return cv2.imencode('.jpg',image)[1].tobytes()


def test_gateway_contracts_decode_one_real_tpu_boundary_call():
    for handler,channels,label in ((sv_handle,17,'no_helmet'),(primary_handle,84,'person')):
        tpu = FakeTPU(channels)
        response = handler(jpeg(),tpu,'id-42',.4)
        assert len(tpu.calls) == 1
        tensor = tpu.calls[0]
        assert tensor.shape == (1,3,640,640) and tensor.dtype == np.float32
        assert tensor[0,2,80,0]>.95 and tensor[0,0,80,0]<.05  # BGR blue -> RGB blue channel.
        assert response['source_width'] == 640 and response['source_height'] == 480
        assert response['frame_id'] == 'id-42'
        assert response['detections'][0]['class_name'] == label
        assert response['detections'][0]['bbox'] == [80,70,120,130]


def test_safety_gateway_never_decodes_coco_as_safetyvision():
    with pytest.raises(ValueError,match='SafetyVision'):
        sv_handle(jpeg(),FakeTPU(84),'wrong-program',.4)

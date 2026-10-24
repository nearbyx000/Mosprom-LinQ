from dataclasses import replace
import pytest

from safesight.jpeg_gateway import JPEGSettings
from safesight.rabbitmq import Settings


@pytest.mark.parametrize('field,value',[('timeout',float('nan')),('max_result_age',float('inf')),
                                       ('timeout',0),('port',0),('port',65536)])
def test_nonfinite_timeouts_and_invalid_ports_fail_before_rabbitmq(field,value):
    settings = [JPEGSettings(route='safesight.jpeg.test'),
                Settings(program_uri='file:///auto/cache/test/yolov8n.tpu?raw=False')]
    for original in settings:
        with pytest.raises(ValueError):
            replace(original,**{field:value}).validate()

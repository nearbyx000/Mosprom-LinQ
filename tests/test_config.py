import copy
import json

import pytest

from safesight.config import load_config


def config(tmp_path,points):
    path = tmp_path/'camera.json'
    path.write_text(json.dumps({'cameras':[{'id':'c','name':'Camera','source':0,
                         'zones':[{'id':'z','name':'Zone','points':points}]}]}))
    return path


@pytest.mark.parametrize('points',[
    [[0,0],[1,0],[1,1],[0,1]],
    [[0,0],[1,0],[1,1],[0,1],[0,0]],
])
def test_valid_normalized_polygon(tmp_path,points):
    assert len(load_config(config(tmp_path,points))['cameras'][0]['zones'][0]['points']) == 4


@pytest.mark.parametrize('points',[
    [[0,0],[1,1],[0,1],[1,0]], [[0,0],[.5,.5],[1,1]],
    [[0,0],[2,0],[1,1]], [[0,0],[1,0],[1,float('nan')]],
    [[0,0],[True,0],[1,1]], [[0,0],[1,0]],
])
def test_bad_polygon_rejected_before_camera_starts(tmp_path,points):
    with pytest.raises(ValueError):
        load_config(config(tmp_path,points))

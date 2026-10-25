"""Validate operator camera/zone configuration before starting worker threads."""
import json
import math
from pathlib import Path
import re


def identifier(value):
    return isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}',value) is not None


def _cross(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _intersect(a,b,c,d):
    if max(a[0],b[0])<min(c[0],d[0]) or max(c[0],d[0])<min(a[0],b[0]):
        return False
    if max(a[1],b[1])<min(c[1],d[1]) or max(c[1],d[1])<min(a[1],b[1]):
        return False
    return _cross(a,b,c)*_cross(a,b,d)<=0 and _cross(c,d,a)*_cross(c,d,b)<=0


def load_config(path):
    document = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    cameras = document.get('cameras') if isinstance(document,dict) else None
    if not isinstance(cameras,list) or len(cameras)!=1:
        raise ValueError('Configure exactly one camera for this release')
    for camera in cameras:
        if not isinstance(camera,dict) or not identifier(camera.get('id')):
            raise ValueError('Camera id must use 1..64 ASCII letters, digits, underscores or hyphens')
        if not isinstance(camera.get('name'),str) or not camera['name'].strip():
            raise ValueError('Camera name must be nonempty')
        source = camera.get('source')
        if isinstance(source,bool) or not isinstance(source,(str,int)) or (isinstance(source,int) and source<0) or source=='':
            raise ValueError('Camera source must be a device index, video path, URL or synthetic')
        zones = camera.setdefault('zones',[])
        if not isinstance(zones,list) or len(zones)>32:
            raise ValueError('Camera zones must be a list with at most 32 items')
        seen = set()
        for zone in zones:
            if not isinstance(zone,dict) or not identifier(zone.get('id')) or zone['id'] in seen:
                raise ValueError('Zone ids must be valid and unique within the camera')
            seen.add(zone['id'])
            if not isinstance(zone.get('name'),str) or not zone['name'].strip():
                raise ValueError('Zone name must be nonempty')
            points = zone.get('points')
            if not isinstance(points,list) or not 3<=len(points)<=65:
                raise ValueError('A zone needs 3..64 polygon vertices')
            if any(not isinstance(p,list) or len(p)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or
                   not math.isfinite(v) or not 0<=v<=1 for v in p) for p in points):
                raise ValueError('Zone points must be finite normalized x/y coordinates in [0,1]')
            if points[-1]==points[0]:
                points = points[:-1]
                zone['points'] = points
            if not 3<=len(points)<=64 or len(set(map(tuple,points)))!=len(points):
                raise ValueError('Zone vertices must be distinct')
            area = abs(sum(p[0]*points[(i+1)%len(points)][1]-points[(i+1)%len(points)][0]*p[1]
                           for i,p in enumerate(points)))/2
            if area<1e-6:
                raise ValueError('Zone polygon has no usable area')
            for i in range(len(points)):
                for j in range(i+1,len(points)):
                    if j==i+1 or (i==0 and j==len(points)-1):
                        continue
                    if _intersect(points[i],points[(i+1)%len(points)],points[j],points[(j+1)%len(points)]):
                        raise ValueError('Zone polygon edges must not cross')
    return document

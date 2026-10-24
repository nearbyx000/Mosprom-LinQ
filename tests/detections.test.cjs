const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function setup() {
  let now = 0;
  const strokes = [], sockets = [];
  const ctx = {setTransform(){}, clearRect(){}, strokeRect(...v){strokes.push(v);}, fillText(){}};
  const canvas = {width:640, height:480, getContext:()=>ctx};
  const video = {width:640, height:480, clientWidth:800, clientHeight:600};
  const state = {dataset:{}}, updated = {}, age = {};
  class Socket {
    constructor(url){this.url=url;sockets.push(this);}
    close(){}
  }
  const context = vm.createContext({WebSocket:Socket, location:{protocol:'https:',host:'test'},
    window:{devicePixelRatio:1}, performance:{now:()=>now},
    setInterval:()=>1,clearInterval(){},setTimeout:()=>1,clearTimeout(){}, console});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../safesight/static/detections.js'),'utf8'),context);
  const Class = vm.runInContext('DetectionOverlay',context);
  const overlay = new Class(canvas,video,state,updated,age);
  overlay.start('cam-01',true);
  const packet = {type:'detections', camera_id:'cam-01',source:'test-json',frame_id:1,generation:1,
    coordinate_format:'xyxy',frame_width:640,frame_height:480,result_age_ms:100,
    updated_at:'2026-10-24T10:00:00Z',detections:[{label:'person',confidence:.9,bbox:[64,48,320,400]}]};
  return {context,overlay,packet,strokes,video,state,updated,age,sockets,setNow:v=>now=v};
}

test('boxes match letterboxed video coordinates',()=>{
  const env=setup();
  const result=vm.runInContext('projectDetection([128,72,640,360],1280,720,800,600)',env.context);
  assert.deepEqual(Array.from(result),[80,120,320,180]);
});

test('valid JSON draws boxes and timestamp; empty result clears them',()=>{
  const e=setup();
  assert.equal(e.overlay.accept(e.packet),true);
  assert.equal(e.strokes.length,1);
  assert.equal(e.updated.dateTime,e.packet.updated_at);
  assert.match(e.state.textContent,/ТЕСТ JSON/);
  e.overlay.accept({...e.packet,frame_id:2,detections:[]});
  assert.equal(e.overlay.packet.detections.length,0);
});

test('malformed, late, duplicate, wrong-camera and wrong-source packets rejected',()=>{
  const e=setup();
  assert.equal(e.overlay.accept({...e.packet,camera_id:'other'}),false);
  assert.equal(e.overlay.accept({...e.packet,source:'linq'}),false);
  assert.equal(e.overlay.accept({...e.packet,result_age_ms:2000}),false);
  assert.equal(e.overlay.accept({...e.packet,detections:[{label:'x',confidence:.9,bbox:[NaN,0,20,20]}]}),false);
  assert.equal(e.overlay.accept(e.packet),true);
  assert.equal(e.overlay.accept(e.packet),false);
});

test('expired boxes are hidden; disconnect leaves video dimensions unchanged',()=>{
  const e=setup();
  e.overlay.accept(e.packet);
  e.setNow(1600);
  e.overlay.draw();
  assert.equal(e.strokes.length,1);
  assert.match(e.age.textContent,/устарели/);
  e.sockets[0].onclose();
  assert.equal(e.overlay.packet,null);
  assert.deepEqual(e.video,{width:640,height:480,clientWidth:800,clientHeight:600});
});

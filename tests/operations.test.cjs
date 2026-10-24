const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const ctx = vm.createContext({URLSearchParams});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../safesight/static/operations.js'),'utf8'),ctx);
const evaluate = camera => ctx.operatorWarnings(camera);

test('disabled analysis is a system warning, never proof of no violations',()=>{
  const result = evaluate({preview_available:true,inference_status:'disabled',event_status:'ready',warnings_fresh:false,active_warnings:[{violation_type:'restricted_zone'}]});
  assert.equal(result.fresh,false);
  assert.equal(result.incidents.length,0);
  assert.equal(result.system[0].title,'Анализ выключен');
});

test('fresh confirmed incidents coexist with a separate storage failure',()=>{
  const result = evaluate({preview_available:true,inference_status:'running',event_status:'failed',warnings_fresh:true,active_warnings:[{violation_type:'restricted_zone'}]});
  assert.equal(result.incidents.length,1);
  assert.equal(result.system.length,1);
  assert.equal(result.system[0].title,'Ошибка сохранения событий');
});

test('stale observations and disconnected camera are visible independently',()=>{
  const result = evaluate({preview_available:false,status:'unavailable',inference_status:'stopped',event_status:'ready',warnings_fresh:false});
  assert.equal(result.system.length,2);
  assert.equal(result.incidents.length,0);
});

test('primary success cannot claim fresh PPE analysis when SafetyVision is unavailable',()=>{
  const result = evaluate({preview_available:true,inference_status:'running',event_status:'ready',
    warnings_fresh:true,safetyvision_expected:true,safetyvision_fresh:false,
    active_warnings:[{violation_type:'restricted_zone'}]});
  assert.equal(result.fresh,false);
  assert.equal(result.incidents.length,1);
  assert.equal(result.system[0].title,'Нет свежих результатов SafetyVision');
});

test('event filters export explicit UTC and reject reversed periods',()=>{
  const query = ctx.eventQuery({start:'2026-10-25T09:00:00+03:00',end:'2026-10-25T10:00:00+03:00',camera_id:'cam',limit:50});
  assert.equal(query.get('start'),'2026-10-25T06:00:00.000Z');
  assert.equal(query.get('end'),'2026-10-25T07:00:00.000Z');
  assert.equal(query.get('limit'),'50');
  assert.throws(()=>ctx.eventQuery({start:'2026-10-25',end:'2026-10-24'}));
  assert.throws(()=>ctx.eventQuery({start:'not a date'}));
});

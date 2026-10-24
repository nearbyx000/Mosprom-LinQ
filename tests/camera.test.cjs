const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

function setup(secure = true, acquire) {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: id === 'source' ? 'device' : 'environment', dataset: {}, hidden: true,
      textContent: '', videoWidth: 640, videoHeight: 480,
      addEventListener() {}, removeAttribute() {}, hasAttribute() {return true;},
      querySelector() {return element('paragraph');}, play: async () => {},
    });
    return elements.get(id);
  };
  let stopped = 0;
  const track = {stop() {stopped++;}, addEventListener() {}};
  const media = {getTracks: () => [track], getVideoTracks: () => [track]};
  const timers = [];
  const sockets = [];
  class Socket {
    static OPEN = 1;
    constructor(url) {this.url=url; this.readyState=1; this.sent=[]; sockets.push(this);}
    send(blob) {this.sent.push(blob);}
    close() {this.readyState=3;}
  }
  const context = vm.createContext({
    document: {getElementById: element, createElement: () => ({
      getContext: () => ({drawImage() {}}), toBlob: callback => callback('jpeg-blob'),
    })},
    window: {isSecureContext: secure, addEventListener() {}},
    navigator: {mediaDevices: {getUserMedia: acquire || (async () => media)}},
    location: {protocol: 'https:', host: 'camera.example'}, WebSocket: Socket,
    setTimeout: callback => {timers.push(callback); return timers.length;}, clearTimeout() {},
    console,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8'), context);
  return {context, element, timers, sockets, media, stopped: () => stopped};
}

test('device upload waits for acknowledgements and stop closes camera and websocket', async () => {
  const env = setup();
  await vm.runInContext('connect()', env.context);
  const ws = env.sockets[0];
  assert.equal(ws.url, 'wss://camera.example/phone/publish');
  assert.equal(ws.sent.length, 0);
  ws.onmessage();
  env.timers.shift()();
  assert.deepEqual(ws.sent, ['jpeg-blob']);
  assert.equal(env.timers.length, 0); // next upload requires another acknowledgement
  vm.runInContext('stop()', env.context);
  assert.equal(env.stopped(), 1);
  assert.equal(ws.readyState, 3);
});

test('insecure phone page reports HTTPS requirement without requesting the camera', async () => {
  let requested = false;
  const env = setup(false, async () => {requested=true;});
  await vm.runInContext('connect()', env.context);
  assert.equal(requested, false);
  assert.match(env.element('status').textContent, /HTTPS/);
});

test('a camera granted after stop is immediately released', async () => {
  let grant;
  const pending = new Promise(resolve => {grant=resolve;});
  const env = setup(true, () => pending);
  const connecting = vm.runInContext('connect()', env.context);
  vm.runInContext('stop()', env.context);
  grant(env.media);
  await connecting;
  assert.equal(env.stopped(), 1);
  assert.equal(env.sockets.length, 0);
});

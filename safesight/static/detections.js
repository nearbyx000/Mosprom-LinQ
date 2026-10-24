/* Independent WebSocket and transparent canvas; never owns the video connection. */
function validDetections(data, cameraId) {
  if (!data || data.type !== "detections" || data.camera_id !== cameraId ||
      data.coordinate_format !== "xyxy" || !["linq", "demo", "local", "test-json"].includes(data.source) ||
      !Number.isSafeInteger(data.frame_id) || data.frame_id < 0 ||
      !Number.isSafeInteger(data.generation) || data.generation < 0 ||
      !Number.isFinite(data.frame_width) || data.frame_width <= 0 || data.frame_width > 16384 ||
      !Number.isFinite(data.frame_height) || data.frame_height <= 0 || data.frame_height > 16384 ||
      !Number.isFinite(data.result_age_ms) || data.result_age_ms < 0 ||
      !Number.isFinite(Date.parse(data.updated_at)) || !Array.isArray(data.detections) || data.detections.length > 200) return false;
  return data.detections.every(d => typeof d.label === "string" && d.label.length <= 100 &&
    Number.isFinite(d.confidence) && d.confidence >= 0 && d.confidence <= 1 &&
    Array.isArray(d.bbox) && d.bbox.length === 4 && d.bbox.every(Number.isFinite) &&
    d.bbox[2] > d.bbox[0] && d.bbox[3] > d.bbox[1]);
}

function projectDetection(box, frameWidth, frameHeight, viewWidth, viewHeight) {
  const scale = Math.min(viewWidth / frameWidth, viewHeight / frameHeight);
  const dx = (viewWidth - frameWidth * scale) / 2;
  const dy = (viewHeight - frameHeight * scale) / 2;
  const x1 = Math.max(0, Math.min(frameWidth, box[0]));
  const y1 = Math.max(0, Math.min(frameHeight, box[1]));
  const x2 = Math.max(0, Math.min(frameWidth, box[2]));
  const y2 = Math.max(0, Math.min(frameHeight, box[3]));
  return [dx + x1 * scale, dy + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale];
}

class DetectionOverlay {
  constructor(canvas, video, state, updated, age) {
    Object.assign(this, {canvas, video, state, updated, age});
    this.generation = 0;
    this.observer = typeof ResizeObserver === "function" ? new ResizeObserver(() => this.draw()) : null;
  }

  stop() {
    this.generation++;
    clearTimeout(this.retry);
    clearInterval(this.tick);
    this.observer?.disconnect();
    if (this.socket) { this.socket.onclose = null; this.socket.close(); }
    this.packet = null;
    this.lastId = -1;
    this.lastGeneration = null;
    this.canvas.getContext("2d").clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  start(cameraId, test = false) {
    this.stop();
    this.cameraId = cameraId;
    this.test = test;
    const generation = this.generation;
    this.updated.textContent = "—";
    this.age.textContent = "Нет результатов";
    this.observer?.observe(this.video);
    this.tick = setInterval(() => this.draw(), 100);
    let backoff = 500;
    const connect = () => {
      if (generation !== this.generation) return;
      this.state.textContent = "Подключение детекций…";
      const ws = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws/detections/${encodeURIComponent(cameraId)}${test ? "?test=1" : ""}`);
      this.socket = ws;
      ws.onopen = () => {
        if (generation !== this.generation) return;
        backoff = 500;
        this.state.textContent = test ? "ТЕСТ JSON · НЕ TPU" : "WebSocket подключён · ожидание детекций";
        this.state.dataset.state = test ? "test" : "waiting";
      };
      ws.onmessage = event => {
        if (generation !== this.generation) return;
        try {
          if (typeof event.data !== "string" || event.data.length > 256 * 1024) return;
          const data = JSON.parse(event.data);
          if (data.type === "status" && data.camera_id === cameraId) {
            this.state.textContent = `Ожидание детекций · ${data.source}: ${data.inference_status}`;
          } else this.accept(data);
        } catch (_) { /* A malformed detection must never stop live video. */ }
      };
      ws.onerror = () => ws.close();
      ws.onclose = () => {
        if (generation !== this.generation) return;
        this.packet = null;
        this.lastId = -1;
        this.lastGeneration = null;
        this.draw();
        this.state.textContent = "Связь с детекциями потеряна · переподключение";
        this.state.dataset.state = "waiting";
        this.retry = setTimeout(connect, backoff);
        backoff = Math.min(10000, backoff * 2);
      };
    };
    connect();
  }

  accept(data) {
    if (!validDetections(data, this.cameraId) || (this.test ? data.source !== "test-json" : data.source === "test-json")) return false;
    if (this.lastGeneration === data.generation && data.frame_id <= this.lastId) return false;
    if (data.result_age_ms >= 1500) return false;
    this.lastGeneration = data.generation;
    this.lastId = data.frame_id;
    this.packet = data;
    this.received = performance.now();
    this.updated.textContent = new Date(data.updated_at).toLocaleTimeString("ru-RU");
    this.updated.dateTime = data.updated_at;
    this.state.textContent = data.source === "test-json" ? "ТЕСТ JSON · НЕ TPU" : `${data.source === "linq" ? "LinQ TPU" : data.source + " · НЕ TPU"} · кадр №${data.frame_id}`;
    this.state.dataset.state = data.source === "test-json" ? "test" : "live";
    this.draw();
    return true;
  }

  draw() {
    const width = this.video.clientWidth, height = this.video.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(width * ratio), h = Math.round(height * ratio);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!this.packet) { this.age.textContent = "Нет актуальных рамок"; return; }
    const age = this.packet.result_age_ms + performance.now() - this.received;
    this.age.textContent = age >= 1500 ? "Рамки устарели" : `Возраст: ${Math.round(age)} мс`;
    if (age >= 1500 || this.video.hidden) return;
    if (Math.abs((this.video.width / this.video.height) / (this.packet.frame_width / this.packet.frame_height) - 1) > .01) {
      this.age.textContent = "Размер кадра изменился · ожидание новых координат";
      return;
    }
    ctx.strokeStyle = this.packet.source === "test-json" ? "#ffc56a" : "#7ee8ba";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 2;
    ctx.font = "14px system-ui";
    for (const detection of this.packet.detections) {
      const [x, y, w, h] = projectDetection(detection.bbox, this.packet.frame_width, this.packet.frame_height, width, height);
      if (w <= 0 || h <= 0) continue;
      ctx.strokeRect(x, y, w, h);
      ctx.fillText(`${detection.label} ${(detection.confidence * 100).toFixed(0)}%`, x + 3, Math.max(16, y - 5));
    }
  }
}

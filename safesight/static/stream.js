/* MJPEG reader with a single pending frame and measured browser rendering. */
class PreviewStream {
  constructor(canvas, onState) {
    this.canvas = canvas;
    this.onState = onState;
    this.generation = 0;
    this.pending = null;
    this.rendering = false;
    this.ticks = [];
    this.renderTimes = [];
    this.queueTimes = [];
    this.dropped = 0;
  }

  stop() {
    this.generation += 1;
    this.controller?.abort();
    clearTimeout(this.watchdog);
    this.pending = null;
  }

  async start(url) {
    this.stop();
    const generation = this.generation;
    this.controller = new AbortController();
    this.onState("Подключение…");
    const resetWatchdog = () => {
      clearTimeout(this.watchdog);
      this.watchdog = setTimeout(() => {
        if (generation === this.generation) {
          this.controller.abort();
          this.onState("Нет кадров. Проверьте камеру и подключитесь заново.");
        }
      }, 10000);
    };
    resetWatchdog();
    try {
      const response = await fetch(url, { signal: this.controller.signal, cache: "no-store" });
      if (!response.ok) throw new Error(`Камера недоступна (HTTP ${response.status})`);
      const reader = response.body.getReader();
      let buffer = new Uint8Array(0);
      let frameLength = null;
      while (generation === this.generation) {
        const { value, done } = await reader.read();
        if (done) throw new Error("Видеопоток завершён. Подключитесь заново.");
        const joined = new Uint8Array(buffer.length + value.length);
        joined.set(buffer);
        joined.set(value, buffer.length);
        buffer = joined;
        if (buffer.length > 8 * 1024 * 1024) throw new Error("Слишком большой кадр MJPEG");
        while (true) {
          if (frameLength === null) {
            let end = -1;
            for (let i = 0; i + 3 < buffer.length; i += 1) {
              if (buffer[i] === 13 && buffer[i + 1] === 10 && buffer[i + 2] === 13 && buffer[i + 3] === 10) {
                end = i;
                break;
              }
            }
            if (end < 0) break;
            const headers = new TextDecoder().decode(buffer.subarray(0, end));
            const match = headers.match(/Content-Length:\s*(\d+)/i);
            if (!match) throw new Error("Некорректный заголовок MJPEG");
            frameLength = Number(match[1]);
            if (frameLength <= 0 || frameLength > 8 * 1024 * 1024) throw new Error("Некорректный размер JPEG");
            buffer = buffer.subarray(end + 4);
          }
          if (buffer.length < frameLength) break;
          if (this.pending) this.dropped += 1;
          this.pending = { bytes: buffer.slice(0, frameLength), received: performance.now(), generation };
          buffer = buffer.subarray(frameLength);
          frameLength = null;
          resetWatchdog();
          this.renderPending();
        }
      }
    } catch (error) {
      if (generation === this.generation && error.name !== "AbortError") this.onState(error.message);
    } finally {
      if (generation === this.generation) {
        clearTimeout(this.watchdog);
        this.controller.abort();
      }
    }
  }

  async renderPending() {
    if (this.rendering) return;
    this.rendering = true;
    try {
      while (this.pending) {
        const frame = this.pending;
        this.pending = null;
        const started = performance.now();
        const bitmap = await createImageBitmap(new Blob([frame.bytes], { type: "image/jpeg" }));
        if (frame.generation !== this.generation) { bitmap.close(); continue; }
        if (this.canvas.width !== bitmap.width || this.canvas.height !== bitmap.height) {
          this.canvas.width = bitmap.width;
          this.canvas.height = bitmap.height;
        }
        this.canvas.getContext("2d").drawImage(bitmap, 0, 0);
        bitmap.close();
        const now = performance.now();
        this.ticks.push(now);
        this.ticks = this.ticks.filter(t => t >= now - 10000);
        this.renderTimes.push(now - started);
        this.queueTimes.push(started - frame.received);
        if (this.renderTimes.length > 300) this.renderTimes.shift();
        if (this.queueTimes.length > 300) this.queueTimes.shift();
        this.onState("Прямой эфир");
      }
    } catch (error) {
      this.onState(`Ошибка отображения: ${error.message}`);
    } finally {
      this.rendering = false;
    }
  }

  metrics() {
    const now = performance.now();
    const ticks = this.ticks.filter(t => t >= now - 10000);
    const percentile = (values, p) => {
      if (!values.length) return null;
      const sorted = [...values].sort((a, b) => a - b);
      return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))].toFixed(2);
    };
    return { fps: ticks.length > 1 ? ((ticks.length - 1) * 1000 / (now - ticks[0])).toFixed(1) : "0.0",
      decodeDrawP50: percentile(this.renderTimes, 0.5), decodeDrawP95: percentile(this.renderTimes, 0.95),
      queueP95: percentile(this.queueTimes, 0.95), dropped: this.dropped };
  }
}

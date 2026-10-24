const $ = id => document.getElementById(id);
let cameraId = null;
let lastDetection = null;
let lastEventId = null;
let eventsBusy = false;
const preview = new PreviewStream($("live"), text => { $("streamstate").textContent = text; });
const names = { capture_read: "Чтение камеры", jpeg_render: "Подготовка JPEG", capture_to_preview: "Кадр → JPEG готов",
  capture_to_stream_yield: "Кадр → отправка MJPEG", preprocessing: "Предобработка", serialization: "Сериализация",
  rpc_round_trip: "RabbitMQ RTT (сеть + очередь + TPU)", deserialization: "Десериализация", postprocessing: "YOLO decode + NMS",
  capture_to_detection: "Кадр → детекция", event_processing: "Обработка событий", capture_to_event: "Кадр → запись события", backend_call: "Вызов локального backend" };
function td(value) { const cell = document.createElement("td"); cell.textContent = value; return cell; }

async function loadStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const status = await response.json();
    const camera = status.cameras[0];
    if (!camera) return;
    $("mode").textContent = status.mode === "preview" ? "Просмотр камеры · инференс выключен" : status.mode === "linq" ? `LinQ TPU / RabbitMQ · ответ получен: ${camera.tpu_response_received ? "да" : "ещё нет"}` : `${status.backend} · НЕ LinQ TPU`;
    $("mode").className = status.mode === "linq" && camera.tpu_response_received ? "ok" : "warn";
    $("camname").textContent = camera.name;
    $("health").textContent = camera.status;
    $("inferhealth").textContent = camera.inference_status;
    $("eventhealth").textContent = camera.event_status;
    const metrics = camera.metrics;
    $("capturefps").textContent = metrics.fps.capture ?? "—";
    $("previewfps").textContent = metrics.fps.preview ?? "—";
    $("inferencefps").textContent = metrics.fps.detections ?? "—";
    $("inflight").textContent = camera.rabbitmq.inflight ?? 0;
    $("age").textContent = camera.last_detection_age_ms === null ? "—" : `${Math.round(camera.last_detection_age_ms)} мс`;
    $("drops").textContent = metrics.counts.inference_frames_skipped ?? 0;
    const body = $("timings"); body.replaceChildren();
    for (const [key, value] of Object.entries(metrics.latency_ms)) {
      const row = document.createElement("tr"); row.append(td(names[key] || key), td(value.p50), td(value.p95)); body.append(row);
    }
    $("notes").textContent = "RTT объединяет сеть, очередь и TPU. Отдельное время TPU не измерено без телеметрии воркера. Метрика JPEG заканчивается на сервере; FPS браузера измеряется отдельно.";
    if (cameraId !== camera.id) { cameraId = camera.id; preview.start(`/api/stream/${encodeURIComponent(cameraId)}`); }
    if (camera.last_detection_frame !== null && lastDetection !== camera.last_detection_frame) {
      lastDetection = camera.last_detection_frame;
      $("detected").hidden = false;
      $("detected").src = `/api/detection/${encodeURIComponent(cameraId)}?frame=${lastDetection}`;
      $("detectcaption").textContent = `Исходный кадр №${lastDetection}. Этот снимок обновляется с частотой детекций.`;
    }
    if ($("fc").options.length === 1) {
      for (const c of status.cameras) { $("fc").add(new Option(c.id, c.id)); for (const zone of c.zones) $("fz").add(new Option(zone.name, zone.id)); }
    }
  } catch (error) { $("mode").textContent = `Ошибка API: ${error.message}`; }
}

async function loadEvents() {
  if (eventsBusy) return;
  eventsBusy = true;
  const params = new URLSearchParams();
  for (const [id, key] of [["fc", "camera_id"], ["fz", "zone_id"], ["ft", "violation_type"], ["start", "start"], ["end", "end"]]) {
    const value = $(id).value.trim(); if (value) params.set(key, value);
  }
  try {
    const response = await fetch(`/api/events?${params}`, { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${response.status}`);
    $("rows").replaceChildren(); $("count").textContent = `Событий: ${data.length}`;
    // Notifications are based on persisted events; existing history is silent.
    const latest = data.reduce((maximum, event) => Math.max(maximum, event.id), 0);
    if (lastEventId !== null && latest > lastEventId) $("notification").textContent = "Новое подтверждённое нарушение. Кадр сохранён в журнале.";
    lastEventId = Math.max(lastEventId || 0, latest);
    for (const event of data) {
      const row = document.createElement("tr");
      row.append(td(new Date(event.timestamp).toLocaleString("ru-RU")), td(event.violation_type), td(`${event.camera_id} / ${event.zone_id}`));
      const cell = document.createElement("td"), link = document.createElement("a"), image = document.createElement("img");
      link.href = `/evidence/${encodeURIComponent(event.evidence_file)}`; link.target = "_blank"; link.rel = "noopener noreferrer";
      image.className = "thumb"; image.src = link.href; image.alt = "Подтверждающий кадр"; image.loading = "lazy";
      link.append(image); cell.append(link); row.append(cell); $("rows").append(row);
    }
  } catch (error) { $("count").textContent = `Ошибка журнала: ${error.message}`; }
  finally { eventsBusy = false; }
}

$("reconnect").addEventListener("click", () => { if (cameraId) preview.start(`/api/stream/${encodeURIComponent(cameraId)}`); });
$("refresh").addEventListener("click", loadEvents);
window.addEventListener("pagehide", () => preview.stop());
setInterval(loadStatus, 2000);
setInterval(loadEvents, 3000);
setInterval(() => {
  const metrics = preview.metrics(); $("browserfps").textContent = metrics.fps;
  $("browserlat").textContent = `${metrics.decodeDrawP50 ?? "—"} / ${metrics.decodeDrawP95 ?? "—"} мс`;
  $("browserdrops").textContent = `${metrics.dropped} · очередь p95: ${metrics.queueP95 ?? "—"} мс`;
}, 1000);
loadStatus(); loadEvents();

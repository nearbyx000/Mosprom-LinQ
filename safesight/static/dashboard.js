const $ = id => document.getElementById(id);
let cameraId = null;
let lastDetection = null;
let lastEventId = null;
let eventsBusy = false;
let statusBusy = false;
let currentView = "cv";
const preview = new PreviewStream($("live"), text => { $("streamstate").textContent = text; });
const overlays = new DetectionOverlay($("boxes"), $("live"), $("detection-state"), $("detection-updated"), $("detection-age"));
const names = { capture_read: "Чтение камеры", jpeg_render: "Подготовка JPEG", capture_to_preview: "Кадр → JPEG готов",
  capture_to_stream_yield: "Кадр → отправка MJPEG", preprocessing: "Предобработка", serialization: "Сериализация",
  rpc_round_trip: "RabbitMQ RTT (сеть + очередь + TPU)", deserialization: "Десериализация", postprocessing: "YOLO decode + NMS",
  capture_to_detection: "Кадр → детекция", event_processing: "Обработка событий", capture_to_event: "Кадр → запись события", backend_call: "Вызов локального backend" };
function td(value) { const cell = document.createElement("td"); cell.textContent = value; return cell; }

function switchView(view) {
  currentView = view;
  document.querySelectorAll("[data-panel]").forEach(panel => {
    panel.hidden = panel.dataset.panel !== ((view === "clean" || view === "cv") ? "stream" : view);
  });
  document.querySelectorAll("[data-view]").forEach(button => {
    const selected = button.dataset.view === view;
    button.classList.toggle("selected", selected);
    if (selected) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  });
  const clean = view === "clean";
  overlays.enabled = !clean;
  overlays.draw();
  $("cv-controls").hidden = clean;
  $("stream-title").textContent = clean ? "Stream clean" : "Stream CV";
  $("stream-description").textContent = clean ? "Чистое видео с камеры без зон и рамок." : "Видео, опасные зоны и последние результаты распознавания.";
  $("view-badge").textContent = clean ? "CLEAN FEED" : "CV OVERLAY";
  $("summary-view").textContent = ({clean:"Clean",cv:"CV",warns:"Warns",backlog:"Log"})[view];
  document.title = `SafeSight — ${({clean:"Stream clean",cv:"Stream CV",warns:"Warns",backlog:"Backlog"})[view]}`;
  if (view === "backlog") loadEvents();
}

function emptyMessage(parent, text) {
  const node = document.createElement("p"); node.className = "empty"; node.textContent = text; parent.append(node);
}

function renderWarnings(cameras, offline = false) {
  const models = cameras.map(camera => ({camera, ...operatorWarnings(camera)}));
  const incidents = models.flatMap(model => model.incidents.map(item => ({...item, camera: model.camera})));
  const system = models.flatMap(model => model.system);
  const fresh = models.length > 0 && models.every(model => model.fresh);
  if (offline) system.push({title:"Нет связи с сервером",detail:"Актуальность видео и анализа не подтверждена. Ожидается переподключение."});
  $("warn-count").textContent = fresh ? incidents.length : "—";
  $("summary-incidents").textContent = fresh ? incidents.length : "—";
  $("summary-system").textContent = system.length;
  $("warn-freshness").textContent = fresh ? "Свежие результаты" : "Нет свежего анализа";
  $("sidebar-warnings").textContent = !fresh ? "Нет свежих результатов: отсутствие нарушений не подтверждено." : incidents.length ? `Требуют внимания: ${incidents.length}. Откройте Warns.` : "В текущих результатах подтверждённых нарушений нет.";
  const list = $("warnings-list"); list.replaceChildren();
  for (const warning of incidents) {
    const card = document.createElement("article"); card.className = "warning-card";
    const level = document.createElement("span"); level.className = "severity";
    level.textContent = "ПОДТВЕРЖДЕНО ПРАВИЛАМИ";
    const title = document.createElement("h3"); title.textContent = violationName(warning.violation_type);
    const detail = document.createElement("p"); detail.className = "small";
    const source = warning.camera.inference_status === "disabled" ? "Анализ выключен" : warning.camera.tpu_response_received ? "LinQ TPU" : "Локальный / DEMO · НЕ TPU";
    const observedAt = warning.observed_at || warning.camera.warnings_updated_at;
    const when = observedAt ? new Date(observedAt * 1000).toLocaleTimeString("ru-RU") : "—";
    const zoneName = warning.zone_id === 'any' ? 'Вся сцена' : warning.camera.zones?.find(z=>z.id===warning.zone_id)?.name || warning.zone_id;
    detail.textContent = `${warning.camera.name} · ${zoneName} · трек #${warning.track_id}. Уверенность ${Math.round(warning.confidence * 100)}%. ${source}. Обновлено ${when}.`;
    const link = document.createElement("a"); link.href = warning.evidence_file ? `/evidence/${encodeURIComponent(warning.evidence_file)}` : `/api/detection/${encodeURIComponent(warning.camera_id)}?frame=${warning.frame_id ?? warning.camera.warnings_frame_id}${warning.generation ? `&generation=${warning.generation}` : ""}`;
    link.target = "_blank"; link.rel = "noopener noreferrer"; link.textContent = warning.evidence_file ? "Доказательный кадр события ↗" : "Кадр наблюдения ↗";
    card.append(level, title, detail, link); list.append(card);
  }
  if (!incidents.length) emptyMessage(list, fresh ? "Подтверждённых нарушений в текущих результатах нет." : "Нет свежих данных анализа. Проверьте состояние детектора ниже.");
  const infrastructure = $("system-warnings"); infrastructure.replaceChildren();
  for (const warning of system) {
    const card = document.createElement("article"); card.className = "system-card";
    const title = document.createElement("strong"); title.textContent = warning.title;
    const detail = document.createElement("p"); detail.className = "small"; detail.textContent = warning.detail;
    card.append(title, detail); infrastructure.append(card);
  }
  if (!system.length) emptyMessage(infrastructure, "Камера, анализ и журнал доступны.");
}

async function loadStatus() {
  if (statusBusy) return;
  statusBusy = true;
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch("/api/status", { cache: "no-store", signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const status = await response.json();
    const camera = status.cameras[0];
    if (!camera) return;
    $("detection-source").options[0].textContent = status.mode === 'linq' ? 'Рабочий детектор / LinQ TPU' : status.mode === 'demo' ? 'Демо-детектор · НЕ TPU' : status.mode === 'local' ? 'Локальный детектор · НЕ TPU' : 'Анализ выключен';
    const hardwareFresh = status.mode === 'linq' && camera.tpu_response_received && camera.warnings_fresh && camera.inference_status === 'running';
    $("mode").textContent = status.mode === "preview" ? "Просмотр камеры · анализ выключен" : status.mode === "linq" ? `LinQ TPU · ${hardwareFresh ? "анализ активен" : "ожидание свежих результатов"}` : `${sourceName(status.mode)}`;
    $("mode").className = hardwareFresh ? "ok" : "warn";
    const banner = $("mode-banner");
    banner.hidden = status.mode === 'linq';
    banner.textContent = status.mode === 'demo' ? 'ДЕМОНСТРАЦИЯ ПРАВИЛ · Цветовые фигуры, без нейросети и TPU. События помечаются как DEMO.' : status.mode === 'local' ? 'ЛОКАЛЬНЫЙ ИНФЕРЕНС · Результаты получены на этом компьютере, LinQ TPU не используется.' : 'РЕЖИМ ПРОСМОТРА · Видео доступно. Нарушения сейчас не анализируются.';
    $("camname").textContent = camera.name;
    $("health").textContent = camera.status;
    const sv = camera.rabbitmq || {};
    $("inferhealth").textContent = camera.inference_status + (sv.sv_enabled ? ` · SV TPU: объединено ${sv.sv_joined || 0}, пропущено ${sv.sv_missed || 0}` : sv.sv_error ? ` · SafetyVision недоступна: ${sv.sv_error}` : "");
    $("eventhealth").textContent = camera.event_status;
    renderWarnings(status.cameras);
    overlays.zones = camera.zones || [];
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
    if (cameraId !== camera.id) { cameraId = camera.id; preview.start(`/api/stream/${encodeURIComponent(cameraId)}`); overlays.start(cameraId, $("detection-source").value === "test"); }
    if (camera.last_detection_frame !== null && lastDetection !== camera.last_detection_frame) {
      lastDetection = camera.last_detection_frame;
      $("detected").hidden = false;
      $("detected").src = `/api/detection/${encodeURIComponent(cameraId)}?frame=${lastDetection}`;
      $("detectcaption").textContent = `Исходный кадр №${lastDetection}. Этот снимок обновляется с частотой детекций.`;
    }
    if ($("fc").options.length === 1) {
      for (const c of status.cameras) { $("fc").add(new Option(c.id, c.id)); for (const zone of c.zones) $("fz").add(new Option(zone.name, zone.id)); }
    }
  } catch (error) { $("mode").textContent = `Ошибка API: ${error.message}`; renderWarnings([], true); }
  finally { clearTimeout(timeout); statusBusy = false; }
}

async function loadEvents(force = false) {
  if (eventsBusy || (!force && document.activeElement?.closest('#rows'))) return;
  eventsBusy = true;
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(),5000);
  try {
    const filters = {limit:$("event-limit").value};
    for (const [id,key] of [["fc","camera_id"],["fz","zone_id"],["ft","violation_type"],["start","start"],["end","end"]]) filters[key] = $(id).value.trim();
    const params = eventQuery(filters);
    $("export-csv").href = `/api/events/export.csv?${params}`;
    const response = await fetch(`/api/events?${params}`, { cache: "no-store",signal:controller.signal });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${response.status}`);
    $("rows").replaceChildren(); $("count").textContent = `Показано событий: ${data.length}`;
    $("events-empty").hidden = data.length > 0;
    // Notifications are based on persisted events; existing history is silent.
    const latest = data.reduce((maximum, event) => Math.max(maximum, event.id), 0);
    if (lastEventId !== null && latest > lastEventId) $("notification").textContent = "Новое подтверждённое нарушение. Кадр сохранён в журнале.";
    lastEventId = Math.max(lastEventId || 0, latest);
    for (const event of data) {
      const row = document.createElement("tr");
      const zone = event.zone_id === 'any' ? 'Вся сцена' : [...$("fz").options].find(option=>option.value===event.zone_id)?.text || event.zone_id;
      row.append(td(new Date(event.timestamp).toLocaleString("ru-RU")), td(violationName(event.violation_type)), td(`${event.camera_id} / ${zone}`));
      const source = document.createElement('small'); source.className = 'event-source'; source.textContent = sourceName(event.source); row.children[1].append(source);
      const cell = document.createElement("td"), link = document.createElement("a"), image = document.createElement("img");
      link.href = `/evidence/${encodeURIComponent(event.evidence_file)}`; link.target = "_blank"; link.rel = "noopener noreferrer";
      image.className = "thumb"; image.src = link.href; image.alt = "Подтверждающий кадр"; image.loading = "lazy";
      link.append(image); cell.append(link); row.append(cell); $("rows").append(row);
      const reviewCell = document.createElement('td');
      const decision = document.createElement('select'); decision.setAttribute('aria-label',`Решение по событию ${event.id}`);
      for (const [value,label] of [['unreviewed','Не проверено'],['confirmed','Подтверждено'],['false_positive','Ложное срабатывание'],['needs_review','Нужна проверка']]) {
        const option = new Option(label,value); option.disabled = value==='unreviewed'; decision.add(option);
      }
      decision.value = event.review_status || 'unreviewed';
      const note = document.createElement('input'); note.placeholder = 'Комментарий'; note.maxLength = 1000;
      note.setAttribute('aria-label',`Комментарий к событию ${event.id}`); note.value = event.review_note || '';
      const save = document.createElement('button'); save.type = 'button'; save.textContent = 'Сохранить'; save.className = 'secondary';
      save.disabled = true;
      const changed = ()=> { save.disabled = decision.value==='unreviewed' || (decision.value===(event.review_status||'unreviewed') && note.value===(event.review_note||'')); };
      decision.addEventListener('change',changed); note.addEventListener('input',changed);
      save.addEventListener('click',async()=>{
        save.disabled = true;
        const control = new AbortController(), timer = setTimeout(()=>control.abort(),5000);
        try {
          const result = await fetch(`/api/events/${event.id}/review`,{method:'PATCH',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({decision:decision.value,note:note.value}),signal:control.signal});
          if (!result.ok) throw new Error(`HTTP ${result.status}`);
          $('notification').textContent = `Проверка события №${event.id} сохранена. Исходная детекция и JPEG не изменены.`;
          await loadEvents(true);
        } catch (error) { $('notification').textContent = `Не удалось сохранить проверку: ${error.message}`; changed(); }
        finally { clearTimeout(timer); }
      });
      reviewCell.append(decision,note,save); row.append(reviewCell);
    }
  } catch (error) { $("count").textContent = `Ошибка журнала: ${error.message}`; }
  finally { clearTimeout(timeout); eventsBusy = false; }
}

async function loadSummary() {
  const controller = new AbortController(), timeout = setTimeout(()=>controller.abort(),5000);
  try {
    const response = await fetch('/api/summary',{cache:'no-store',signal:controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    $('summary-day').textContent = data.total;
    $('summary-pending').textContent = data.outbox_pending;
  } catch (_) {
    $('summary-day').textContent = '—';
    $('summary-pending').textContent = '—';
  } finally { clearTimeout(timeout); }
}

$("reconnect").addEventListener("click", () => { if (cameraId) preview.start(`/api/stream/${encodeURIComponent(cameraId)}`); });
$("refresh").addEventListener("click", loadEvents);
document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => switchView(button.dataset.view)));
for (const id of ["fc", "fz", "ft", "start", "end", "event-limit"]) $(id).addEventListener("change", loadEvents);
$("detection-source").addEventListener("change", () => { if (cameraId) overlays.start(cameraId, $("detection-source").value === "test"); });
window.addEventListener("pagehide", () => { preview.stop(); overlays.stop(); });
setInterval(loadStatus, 2000);
setInterval(loadEvents, 3000);
setInterval(loadSummary,5000);
setInterval(() => {
  const metrics = preview.metrics(); $("browserfps").textContent = metrics.fps;
  $("browserlat").textContent = `${metrics.decodeDrawP50 ?? "—"} / ${metrics.decodeDrawP95 ?? "—"} мс`;
  $("browserdrops").textContent = `${metrics.dropped} · очередь p95: ${metrics.queueP95 ?? "—"} мс`;
}, 1000);
loadStatus(); loadEvents(); loadSummary();

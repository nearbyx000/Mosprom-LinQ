/* Operator state: fresh confirmed incidents and infrastructure are separate. */
function operatorWarnings(camera) {
  const system = [];
  if (!camera.preview_available) system.push({title: "Нет свежего видео", detail: camera.status});
  if (camera.inference_status === "disabled") system.push({title: "Анализ выключен", detail: "Доступен просмотр камеры. Нарушения сейчас не проверяются."});
  else if (!camera.warnings_fresh) system.push({title: "Нет свежих результатов анализа", detail: camera.inference_status});
  const safetyUnavailable = camera.safetyvision_expected && !camera.safetyvision_fresh;
  if (safetyUnavailable) system.push({title: "Нет свежих результатов SafetyVision", detail: "Каски и падения сейчас не подтверждаются. Проверьте второй детектор."});
  if (camera.event_status !== "ready") system.push({title: "Ошибка сохранения событий", detail: camera.event_status});
  return {system, incidents: camera.warnings_fresh ? camera.active_warnings || [] : [], fresh: Boolean(camera.warnings_fresh) && !safetyUnavailable};
}

function violationName(type) {
  return ({restricted_zone: "Человек в опасной зоне", no_helmet: "Работа без каски", fall_detected: "Возможное падение работника"})[type] || type;
}

function sourceName(source) {
  return ({linq:"LinQ TPU",demo:"DEMO · не TPU",local:"Локальный · не TPU",legacy:"Источник не записан"})[source] || "Источник не указан";
}

function eventQuery(filters) {
  const params = new URLSearchParams();
  params.set('limit', String(filters.limit || 100));
  for (const key of ['camera_id','zone_id','violation_type']) {
    if (filters[key]) params.set(key,filters[key]);
  }
  for (const key of ['start','end']) {
    if (!filters[key]) continue;
    const date = new Date(filters[key]);
    if (!Number.isFinite(date.getTime())) throw new Error('Проверьте даты периода');
    params.set(key,date.toISOString());
  }
  if (params.has('start') && params.has('end') && params.get('start')>params.get('end')) {
    throw new Error('Начало периода должно быть раньше окончания');
  }
  return params;
}

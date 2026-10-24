/* Operator state: fresh confirmed incidents and infrastructure are separate. */
function operatorWarnings(camera) {
  const system = [];
  if (!camera.preview_available) system.push({title: "Нет свежего видео", detail: camera.status});
  if (camera.inference_status === "disabled") system.push({title: "Анализ выключен", detail: "Доступен просмотр камеры. Нарушения сейчас не проверяются."});
  else if (!camera.warnings_fresh) system.push({title: "Нет свежих результатов анализа", detail: camera.inference_status});
  if (camera.event_status !== "ready") system.push({title: "Ошибка сохранения событий", detail: camera.event_status});
  return {system, incidents: camera.warnings_fresh ? camera.active_warnings || [] : [], fresh: Boolean(camera.warnings_fresh)};
}

function violationName(type) {
  return ({restricted_zone: "Человек в опасной зоне", no_helmet: "Работа без каски"})[type] || type;
}

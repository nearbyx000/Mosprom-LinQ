const stream = document.getElementById("stream");
const video = document.getElementById("device-video");
const status = document.getElementById("status");
const message = document.getElementById("message");
const source = document.getElementById("source");
const facing = document.getElementById("facing");
let media = null, socket = null, timer = null, generation = 0;
const canvas = document.createElement("canvas");

function setState(state, text) {
  status.dataset.state = state;
  status.textContent = text;
  message.hidden = state !== "error";
  if (state === "error") {
    message.querySelector("p").textContent = text;
    stream.hidden = true;
    video.hidden = true;
  }
}

function stop() {
  generation++;
  clearTimeout(timer);
  if (socket) { socket.onclose = null; socket.close(); socket = null; }
  if (media) { media.getTracks().forEach(track => track.stop()); media = null; }
  video.srcObject = null;
  video.hidden = true;
  stream.removeAttribute("src");
  stream.hidden = true;
}

function updateHint() {
  document.getElementById("facing-control").hidden = source.value !== "device";
  document.getElementById("hint").textContent = source.value === "device"
    ? "Разрешите камеру. Видео передаётся на этот сервер; его можно смотреть с другого устройства в режиме «Трансляция с телефона». Для телефона требуется доверенный HTTPS."
    : source.value === "phone"
      ? "На телефоне откройте этот же сервер, выберите «Камера этого устройства» и нажмите «Подключить»."
      : "Камера сервера: один зритель одновременно.";
}

async function connect() {
  stop();
  updateHint();
  setState("connecting", "Подключение…");
  const current = generation;
  if (source.value !== "device") {
    stream.hidden = false;
    stream.src = `${source.value === "phone" ? "/phone/video" : "/video"}?t=${Date.now()}`;
    return;
  }
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("Для камеры телефона откройте страницу по HTTPS с доверенным сертификатом. HTTP по IP компьютера не подходит.");
    }
    const acquired = await navigator.mediaDevices.getUserMedia({audio: false, video: {
      facingMode: {ideal: facing.value}, width: {ideal: 640}, height: {ideal: 480}
    }});
    if (current !== generation) { acquired.getTracks().forEach(track => track.stop()); return; }
    media = acquired;
    media.getVideoTracks().forEach(track => track.addEventListener("ended", () => {
      if (current === generation) { stop(); setState("error", "Камера остановлена устройством. Нажмите «Подключить»."); }
    }));
    video.srcObject = media;
    video.hidden = false;
    await video.play();
    if (current !== generation) return;
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/phone/publish`);
    socket = ws;
    const sendFrame = () => {
      if (current !== generation || ws.readyState !== WebSocket.OPEN) return;
      if (!video.videoWidth) { timer = setTimeout(sendFrame, 100); return; }
      const ratio = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
      canvas.width = Math.round(video.videoWidth * ratio);
      canvas.height = Math.round(video.videoHeight * ratio);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(blob => {
        if (blob && current === generation && ws.readyState === WebSocket.OPEN) ws.send(blob);
      }, "image/jpeg", .75);
    };
    // Acknowledgements allow only one upload in flight.
    ws.onmessage = () => {
      if (current !== generation) return;
      setState("live", "Камера передаёт видео на сервер");
      timer = setTimeout(sendFrame, 100);
    };
    ws.onerror = () => { if (current === generation) { stop(); setState("error", "Не удалось подключить трансляцию. Проверьте сервер и HTTPS."); } };
    ws.onclose = () => { if (current === generation) { stop(); setState("error", "Трансляция завершена или другой телефон уже подключён."); } };
  } catch (error) {
    if (current !== generation) return;
    stop();
    const explanations = {
      NotAllowedError: "Разрешите доступ к камере в настройках браузера.",
      NotFoundError: "На устройстве не найдена камера.",
      NotReadableError: "Камера занята другим приложением или недоступна."
    };
    setState("error", explanations[error.name] || error.message);
  }
}

stream.addEventListener("load", () => { if (source.value !== "device" && stream.hasAttribute("src")) setState("live", "Камера подключена"); });
stream.addEventListener("error", () => { if (source.value !== "device" && stream.hasAttribute("src")) setState("error", "Нет видеопотока. Проверьте камеру или запустите трансляцию на телефоне, затем нажмите «Подключить»."); });
source.addEventListener("change", () => { stop(); updateHint(); setState("idle", "Нажмите «Подключить»"); });
facing.addEventListener("change", connect);
document.getElementById("reconnect").addEventListener("click", connect);
document.getElementById("stop").addEventListener("click", () => { stop(); setState("idle", "Остановлено"); });
window.addEventListener("pagehide", stop);
updateHint();
setState("idle", "Выберите источник и нажмите «Подключить»");

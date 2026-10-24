const stream = document.getElementById("stream");
const status = document.getElementById("status");
const message = document.getElementById("message");

function setState(state, text) {
  status.dataset.state = state;
  status.textContent = text;
  message.hidden = state !== "error";
  stream.hidden = state === "error";
}

stream.addEventListener("load", () => setState("live", "Камера подключена"));
stream.addEventListener("error", () => setState("error", "Нет подключения"));
document.getElementById("reconnect").addEventListener("click", () => {
  setState("connecting", "Подключение…");
  stream.src = `/video?t=${Date.now()}`;
});

// A failed request may have completed before the deferred script attached listeners.
if (stream.complete) {
  if (stream.naturalWidth > 0) setState("live", "Камера подключена");
  else setState("error", "Нет подключения");
}

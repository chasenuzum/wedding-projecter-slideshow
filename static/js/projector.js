// Projector: receive approved photos over WebSocket, give each new arrival a
// big "reveal", and shuffle the full gallery between arrivals.

const SHUFFLE_MS = 7000;      // time each gallery photo stays up when idle
const NEW_HOLD_MS = 9000;     // extra dwell after a fresh arrival

const splash = document.getElementById("splash");
const slots = [document.getElementById("photo-a"), document.getElementById("photo-b")];
const connDot = document.getElementById("conn");

let gallery = [];
let active = 0;          // which slot is currently visible
let shuffleTimer = null;
let lastShown = null;

function showImage(photo, { reveal = false } = {}) {
  const next = active ^ 1;
  const cur = slots[active];
  const incoming = slots[next];
  incoming.onload = () => {
    splash.style.opacity = "0";
    incoming.classList.add("show");
    if (reveal) {
      incoming.classList.remove("reveal");
      void incoming.offsetWidth; // restart animation
      incoming.classList.add("reveal");
    }
    cur.classList.remove("show");
    active = next;
  };
  incoming.src = photo.url;
  lastShown = photo.id;
}

function scheduleShuffle(delay = SHUFFLE_MS) {
  clearTimeout(shuffleTimer);
  shuffleTimer = setTimeout(shuffleStep, delay);
}

function shuffleStep() {
  if (gallery.length) {
    let pick = gallery[Math.floor(Math.random() * gallery.length)];
    if (gallery.length > 1 && pick.id === lastShown) {
      pick = gallery[(gallery.indexOf(pick) + 1) % gallery.length];
    }
    showImage(pick);
  }
  scheduleShuffle();
}

function onNewPhoto(photo) {
  if (!gallery.find((p) => p.id === photo.id)) gallery.push(photo);
  showImage(photo, { reveal: true });
  scheduleShuffle(NEW_HOLD_MS);
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/projector`);

  ws.onopen = () => connDot.classList.add("online");
  ws.onclose = () => {
    connDot.classList.remove("online");
    setTimeout(connect, 2000); // auto-reconnect
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "gallery") {
      gallery = msg.photos || [];
      if (gallery.length) {
        showImage(gallery[gallery.length - 1]);
        scheduleShuffle();
      }
    } else if (msg.type === "new_photo") {
      onNewPhoto(msg.photo);
    }
  };

  // keepalive ping
  setInterval(() => ws.readyState === 1 && ws.send("ping"), 25000);
}

connect();

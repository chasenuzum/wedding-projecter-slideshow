// Admin review dashboard: live queue of held photos with approve/reject.

const token = window.SLIDESHOW_ADMIN_TOKEN || "";
const queueEl = document.getElementById("queue");
const emptyEl = document.getElementById("empty");
const countEl = document.getElementById("count");
const connDot = document.getElementById("conn");

function updateCount() {
  const n = queueEl.querySelectorAll(".review-card").length;
  countEl.textContent = `${n} waiting`;
  emptyEl.hidden = n > 0;
}

function cardFor(photo) {
  const card = document.createElement("article");
  card.className = "review-card";
  card.id = "card-" + photo.id;
  const reason = photo.reason || photo.verdict || "held for review";
  card.innerHTML = `
    <img src="${photo.url}?token=${encodeURIComponent(token)}" alt="" />
    <div class="reason">${reason}</div>
    <div class="review-actions">
      <button class="reject" data-action="reject">Reject</button>
      <button class="approve" data-action="approve">Approve</button>
    </div>`;
  card.querySelectorAll("button").forEach((btn) =>
    btn.addEventListener("click", () => decide(photo.id, btn.dataset.action, card))
  );
  return card;
}

function addCard(photo) {
  if (document.getElementById("card-" + photo.id)) return;
  queueEl.appendChild(cardFor(photo));
  updateCount();
}

function removeCard(id) {
  const card = document.getElementById("card-" + id);
  if (card) card.remove();
  updateCount();
}

async function decide(id, action, card) {
  card.querySelectorAll("button").forEach((b) => (b.disabled = true));
  try {
    const resp = await fetch(`/admin/decision?token=${encodeURIComponent(token)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, action }),
    });
    if (resp.ok) removeCard(id);
    else card.querySelectorAll("button").forEach((b) => (b.disabled = false));
  } catch (err) {
    card.querySelectorAll("button").forEach((b) => (b.disabled = false));
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/admin?token=${encodeURIComponent(token)}`);

  ws.onopen = () => connDot.classList.add("online");
  ws.onclose = () => {
    connDot.classList.remove("online");
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "queue") (msg.photos || []).forEach(addCard);
    else if (msg.type === "new_review") addCard(msg.photo);
    else if (msg.type === "resolved") removeCard(msg.id);
  };

  setInterval(() => ws.readyState === 1 && ws.send("ping"), 25000);
}

connect();

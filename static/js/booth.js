// Photo booth: live camera, 3-2-1 countdown, capture, auto-upload (source=booth).
// Built for an iPad in kiosk/guided-access mode — big tap target, auto-resets.

const COUNTDOWN_FROM = 3;
const RESULT_HOLD_MS = 4000;

const cam = document.getElementById("cam");
const canvas = document.getElementById("canvas");
const countdownEl = document.getElementById("countdown");
const flashEl = document.getElementById("flash");
const snapBtn = document.getElementById("snap-btn");
const flipBtn = document.getElementById("flip-btn");
const resultEl = document.getElementById("result");
const resultImg = document.getElementById("result-img");
const resultMsg = document.getElementById("result-msg");
const errorEl = document.getElementById("error");
const errorDetail = document.getElementById("error-detail");

let stream = null;
let facingMode = "user"; // front camera by default for selfies
let busy = false;

function showError(msg) {
  errorEl.hidden = false;
  errorDetail.textContent = msg;
  snapBtn.disabled = true;
}

async function startCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    // getUserMedia only exists in a secure context (HTTPS or localhost).
    showError("Camera needs a secure (https://) connection. Use the tunnel URL, not the LAN IP.");
    return;
  }

  // Try ideal HD first, then fall back to looser constraints so a picky camera
  // (OverconstrainedError) still works.
  const attempts = [
    { video: { facingMode, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false },
    { video: { facingMode }, audio: false },
    { video: true, audio: false },
  ];

  let lastErr;
  for (const constraints of attempts) {
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      cam.srcObject = stream;
      await cam.play().catch(() => {});
      errorEl.hidden = true;
      snapBtn.disabled = false;
      // Flip button is best-effort — never let it surface a camera error.
      try {
        const cams = (await navigator.mediaDevices.enumerateDevices()).filter(
          (d) => d.kind === "videoinput"
        );
        flipBtn.hidden = cams.length < 2;
      } catch {
        flipBtn.hidden = true;
      }
      return; // success
    } catch (err) {
      lastErr = err;
    }
  }
  // All attempts failed — surface the real reason (NotReadableError = camera in
  // use by another app, NotAllowedError = permission, etc.).
  showError(`${lastErr?.name || "Error"}: ${lastErr?.message || lastErr}`);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function runCountdown() {
  countdownEl.hidden = false;
  for (let n = COUNTDOWN_FROM; n > 0; n--) {
    countdownEl.textContent = n;
    countdownEl.classList.remove("pop");
    void countdownEl.offsetWidth;
    countdownEl.classList.add("pop");
    await sleep(1000);
  }
  countdownEl.hidden = true;
}

function flash() {
  flashEl.hidden = false;
  flashEl.classList.add("on");
  setTimeout(() => {
    flashEl.classList.remove("on");
    flashEl.hidden = true;
  }, 350);
}

function capture() {
  const w = cam.videoWidth;
  const h = cam.videoHeight;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (facingMode === "user") {
    // Un-mirror: the preview is mirrored for a natural selfie feel, but we
    // store the true (un-flipped) image.
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(cam, 0, 0, w, h);
  return new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.9));
}

async function upload(blob) {
  const fd = new FormData();
  fd.append("files", blob, "booth.jpg");
  fd.append("source", "booth");
  const resp = await fetch("/upload", { method: "POST", body: fd });
  return resp.ok ? resp.json() : Promise.reject(new Error("upload failed"));
}

async function showResult(blob, ok) {
  resultImg.src = URL.createObjectURL(blob);
  resultMsg.textContent = ok
    ? "Sent to the slideshow! 💛"
    : "Couldn’t send — try again.";
  resultEl.hidden = false;
  await sleep(RESULT_HOLD_MS);
  URL.revokeObjectURL(resultImg.src);
  resultEl.hidden = true;
}

async function takePhoto() {
  if (busy || snapBtn.disabled) return;
  busy = true;
  snapBtn.disabled = true;
  try {
    await runCountdown();
    flash();
    const blob = await capture();
    let ok = true;
    try {
      await upload(blob);
    } catch {
      ok = false;
    }
    await showResult(blob, ok);
  } finally {
    busy = false;
    snapBtn.disabled = false;
  }
}

snapBtn.addEventListener("click", takePhoto);
flipBtn.addEventListener("click", () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  cam.classList.toggle("mirror", facingMode === "user");
  startCamera();
});

cam.classList.add("mirror"); // front camera starts mirrored
startCamera();

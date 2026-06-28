// Guest upload.
//
// Android Chrome (and sometimes iOS Safari) FREEZES the page's JS/rendering when
// you return from the native camera/file picker, until the next event (a touch,
// a rotate, refocus). To stay reliable we:
//   1. stage the RAW file on selection (cheap, no async work in the frozen window)
//   2. re-scan the inputs on every "we're back" event (visibility/focus/resize)
//      so pending photos get queued without needing a rotate or extra tap
//   3. compress + upload inside the Send tap (an awake user gesture)
//   4. do the visible UI reset synchronously in that tap so it paints immediately

const MAX_EDGE = 1600;
const JPEG_QUALITY = 0.85;

const input = document.getElementById("file-input");
const cameraInput = document.getElementById("camera-input");
const cameraBtn = document.getElementById("camera-btn");
const dropzone = document.getElementById("dropzone");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const form = document.getElementById("upload-form");

let queued = []; // { file, url }

function setStatus(msg, kind = "") {
  statusEl.textContent = msg;
  statusEl.className = "status" + (kind ? " " + kind : "");
}

function addOne(file) {
  if (file.type && !file.type.startsWith("image/")) return;
  const url = URL.createObjectURL(file);
  queued.push({ file, url });
  const img = document.createElement("img");
  img.src = url;
  preview.appendChild(img);
}

// Pull any files sitting on the inputs into the queue, then clear the inputs so
// the same files aren't ingested twice on the next wake event.
function ingestPending() {
  let added = 0;
  for (const el of [input, cameraInput]) {
    if (el.files && el.files.length) {
      for (const f of el.files) {
        addOne(f);
        added++;
      }
      el.value = "";
    }
  }
  if (added || queued.length) {
    preview.hidden = queued.length === 0;
    submitBtn.disabled = queued.length === 0;
    if (queued.length) setStatus(`${queued.length} photo(s) ready`);
  }
  void document.body.offsetHeight; // nudge a repaint on Android
}

async function compress(file) {
  if (!file.type || !file.type.startsWith("image/")) return file;
  const bitmap = await createImageBitmap(file).catch(() => null);
  if (!bitmap) return file; // HEIC etc. the browser can't decode — server handles it
  let { width, height } = bitmap;
  const scale = Math.min(1, MAX_EDGE / Math.max(width, height));
  width = Math.round(width * scale);
  height = Math.round(height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  return await new Promise((res) =>
    canvas.toBlob((b) => res(b || file), "image/jpeg", JPEG_QUALITY)
  );
}

input.addEventListener("change", ingestPending);
cameraInput.addEventListener("change", ingestPending);
cameraBtn.addEventListener("click", () => cameraInput.click());

// Any of these means "the user came back from the picker" — process pending files.
["focus", "pageshow", "resize", "orientationchange"].forEach((ev) =>
  window.addEventListener(ev, ingestPending)
);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) ingestPending();
});

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  for (const f of e.dataTransfer.files) addOne(f);
  preview.hidden = queued.length === 0;
  submitBtn.disabled = queued.length === 0;
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  ingestPending(); // catch anything still pending on the inputs
  if (!queued.length) return;

  const items = queued;
  queued = [];
  // Visible reset, synchronous inside the tap gesture -> paints immediately.
  preview.innerHTML = "";
  preview.hidden = true;
  input.value = "";
  cameraInput.value = "";
  submitBtn.disabled = true;
  setStatus(`Sending ${items.length} photo(s)…`);

  try {
    const fd = new FormData();
    for (let i = 0; i < items.length; i++) {
      const blob = await compress(items[i].file);
      fd.append("files", blob, `photo-${i}.jpg`);
    }
    items.forEach((q) => URL.revokeObjectURL(q.url));
    const resp = await fetch("/upload", { method: "POST", body: fd });
    const data = await resp.json();
    if (resp.ok && data.accepted && data.accepted.length) {
      setStatus(`Sent ${data.accepted.length} photo(s)! They’ll appear after a quick review. 💛`, "ok");
    } else {
      const why = (data.rejected && data.rejected[0] && data.rejected[0].error) || "upload failed";
      setStatus("Hmm: " + why, "err");
    }
  } catch (err) {
    setStatus("Network error — try again.", "err");
  } finally {
    submitBtn.disabled = false;
  }
});

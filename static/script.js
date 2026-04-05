const video = document.getElementById("video");
const canvas = document.getElementById("captureCanvas");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("statusText");

const summaryText = document.getElementById("summaryText");
const sumStressVal = document.getElementById("sumStressVal");
const sumCalmVal = document.getElementById("sumCalmVal");
const sumDomEmotion = document.getElementById("sumDomEmotion");
const sumStabVal = document.getElementById("sumStabVal");
const sumStressBar = document.getElementById("sumStressBar");
const sumCalmBar = document.getElementById("sumCalmBar");
const sumStabBar = document.getElementById("sumStabBar");
const affectPieCanvas = document.getElementById("affectPie");
const indicesBarCanvas = document.getElementById("indicesBar");
const affectPieCtx = affectPieCanvas ? affectPieCanvas.getContext("2d") : null;
const indicesBarCtx = indicesBarCanvas ? indicesBarCanvas.getContext("2d") : null;

let captureInterval = null;
let running = false;

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
  } catch (err) {
    console.error("Error accessing camera", err);
    statusText.textContent = "Could not access camera. Check browser permissions.";
  }
}

function captureFrame() {
  if (!video.videoWidth || !video.videoHeight) return null;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg", 0.6); // compressed JPEG
  return dataUrl;
}

async function sendFrame() {
  const imgData = captureFrame();
  if (!imgData) return;

  try {
    const res = await fetch("/api/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imgData }),
    });

    if (!res.ok) return;
    // We intentionally ignore the per-frame metrics here to avoid
    // updating the UI in real time. The backend still accumulates
    // them in the buffer so that /api/summary can compute a
    // session-level summary when the user ends the session.
    await res.json();
  } catch (err) {
    console.error("Error sending frame", err);
  }
}

async function fetchSummary() {
  try {
    const res = await fetch("/api/summary");
    if (!res.ok) return;
    const m = await res.json();

    const stress = m.stress_score ?? 0;
    const calm = m.calmness_score ?? 0;
    const stab = m.emotional_stability ?? 0;
    const dom = m.dominant_emotion ?? "neutral";

    // Update dashboard values
    sumStressVal.textContent = stress.toFixed(2);
    sumCalmVal.textContent = calm.toFixed(2);
    sumStabVal.textContent = stab.toFixed(2);
    sumDomEmotion.textContent = dom;

    // Progress bars (values in [0,1])
    sumStressBar.style.width = `${Math.max(0, Math.min(1, stress)) * 100}%`;
    sumCalmBar.style.width = `${Math.max(0, Math.min(1, calm)) * 100}%`;
    sumStabBar.style.width = `${Math.max(0, Math.min(1, stab)) * 100}%`;

    // Charts for a more dashboard-like, "PowerBI" feel
    updateAffectPie(stress, calm);
    updateIndicesBar(stress, calm, stab);

    const lines = [];

    // 1. Overall emotional load
    if (stress >= 0.75) {
      lines.push("Overall emotional load: high, with frequent stress-related expressions.");
    } else if (stress >= 0.45) {
      lines.push("Overall emotional load: moderate, with some episodes of elevated stress.");
    } else {
      lines.push("Overall emotional load: low, stress markers remained limited.");
    }

    // 2. Calm/neutral balance
    if (calm >= 0.7) {
      lines.push("Calm/neutral presence: predominant, the participant appeared mostly regulated.");
    } else if (calm >= 0.4) {
      lines.push("Calm/neutral presence: mixed, with alternating neutral and activated states.");
    } else {
      lines.push("Calm/neutral presence: reduced, more activated expressions than neutral ones.");
    }

    // 3. Primary affect descriptor
    lines.push(`Primary affect: ${dom}.`);

    // 4. Temporal stability
    if (stab >= 0.75) {
      lines.push("Temporal pattern: stable, facial expressions were mostly consistent over time.");
    } else if (stab >= 0.4) {
      lines.push("Temporal pattern: variable, with noticeable but not extreme shifts in expression.");
    } else {
      lines.push("Temporal pattern: highly variable, with frequent shifts in facial expressions.");
    }

    summaryText.textContent = lines.join("\n");
  } catch (err) {
    console.error("Error fetching summary", err);
  }
}

function updateAffectPie(stressRaw, calmRaw) {
  if (!affectPieCtx) return;

  const stress = Math.max(0, Math.min(1, stressRaw));
  const calm = Math.max(0, Math.min(1, calmRaw));
  const other = Math.max(0, 1 - (stress + calm));
  const total = stress + calm + other || 1;

  const slices = [stress / total, calm / total, other / total];
  const colors = ["#0ea5e9", "#22c55e", "#a5b4fc"];

  const cx = affectPieCanvas.width / 2;
  const cy = affectPieCanvas.height / 2;
  const radius = Math.min(cx, cy) - 8;

  affectPieCtx.clearRect(0, 0, affectPieCanvas.width, affectPieCanvas.height);

  let start = -Math.PI / 2;
  for (let i = 0; i < slices.length; i++) {
    const angle = slices[i] * Math.PI * 2;
    const end = start + angle;
    affectPieCtx.beginPath();
    affectPieCtx.moveTo(cx, cy);
    affectPieCtx.arc(cx, cy, radius, start, end);
    affectPieCtx.closePath();
    affectPieCtx.fillStyle = colors[i];
    affectPieCtx.globalAlpha = 0.9;
    affectPieCtx.fill();
    start = end;
  }

  // inner cutout for donut effect
  affectPieCtx.globalCompositeOperation = "destination-out";
  affectPieCtx.beginPath();
  affectPieCtx.arc(cx, cy, radius * 0.55, 0, Math.PI * 2);
  affectPieCtx.fill();
  affectPieCtx.globalCompositeOperation = "source-over";
}

function updateIndicesBar(stressRaw, calmRaw, stabRaw) {
  if (!indicesBarCtx) return;

  const values = [
    Math.max(0, Math.min(1, stressRaw)),
    Math.max(0, Math.min(1, calmRaw)),
    Math.max(0, Math.min(1, stabRaw)),
  ];
  const labels = ["Stress", "Calm", "Stability"];
  const colors = ["#0ea5e9", "#22c55e", "#38bdf8"];

  const w = indicesBarCanvas.width;
  const h = indicesBarCanvas.height;
  indicesBarCtx.clearRect(0, 0, w, h);

  const paddingX = 28;
  const paddingY = 18;
  const chartW = w - paddingX * 2;
  const chartH = h - paddingY * 2;
  const barWidth = chartW / (values.length * 1.8);

  // axis
  indicesBarCtx.strokeStyle = "rgba(148, 163, 184, 0.7)";
  indicesBarCtx.lineWidth = 1;
  indicesBarCtx.beginPath();
  indicesBarCtx.moveTo(paddingX, h - paddingY);
  indicesBarCtx.lineTo(w - paddingX, h - paddingY);
  indicesBarCtx.stroke();

  indicesBarCtx.font = "10px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  indicesBarCtx.fillStyle = "#9ca3af";
  indicesBarCtx.textAlign = "center";

  values.forEach((v, i) => {
    const x = paddingX + (i + 0.5) * (chartW / values.length);
    const barH = chartH * v;
    const y = h - paddingY - barH;

    indicesBarCtx.fillStyle = colors[i];
    indicesBarCtx.globalAlpha = 0.9;
    indicesBarCtx.fillRect(x - barWidth / 2, y, barWidth, barH);

    indicesBarCtx.globalAlpha = 1;
    indicesBarCtx.fillStyle = "#cbd5f5";
    indicesBarCtx.fillText(v.toFixed(2), x, y - 4);

    indicesBarCtx.fillStyle = "#9ca3af";
    indicesBarCtx.fillText(labels[i], x, h - paddingY + 12);
  });
}

startBtn.addEventListener("click", () => {
  if (running) return;
  running = true;
  statusText.textContent = "Streaming frames and analyzing emotions...";
  startBtn.disabled = true;
  stopBtn.disabled = false;

  // Send a frame every 800ms to avoid overloading the backend
  captureInterval = setInterval(sendFrame, 800);
});

stopBtn.addEventListener("click", () => {
  if (!running) return;
  running = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusText.textContent = "Session stopped. Fetching summary...";

  if (captureInterval) {
    clearInterval(captureInterval);
    captureInterval = null;
  }

  fetchSummary().then(() => {
    statusText.textContent = "Session finished. You can start a new one.";
  });
});

initCamera();

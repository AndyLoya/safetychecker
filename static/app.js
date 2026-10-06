const video = document.querySelector("#camera");
const overlay = document.querySelector("#overlay");
const overlayContext = overlay.getContext("2d");
const placeholder = document.querySelector("#camera-placeholder");
const cameraMessage = document.querySelector("#camera-message");
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const connectionDot = document.querySelector("#connection-dot");
const connectionLabel = document.querySelector("#connection-label");
const resultStatus = document.querySelector("#result-status");
const resultMessage = document.querySelector("#result-message");
const resultIcon = document.querySelector("#result-icon");
const errorMessage = document.querySelector("#error-message");
let cameraStream = null;
let frameTimer = null;
let statusTimer = null;
let sendingFrame = false;

function setConnection(connected) {
  connectionDot.classList.toggle("online", connected);
  connectionLabel.textContent = connected ? "Service online" : "Service unavailable";
}

function showCameraError(message) {
  cameraMessage.textContent = message;
  cameraMessage.classList.add("error-text");
}

function updateResult(data) {
  resultStatus.textContent = data.status;
  resultStatus.dataset.phase = data.phase;
  resultIcon.className = "result-icon";
  if (data.status === "DO NOT PASS") {
    resultIcon.classList.add("failed");
    resultIcon.textContent = "!";
    resultMessage.textContent = `Missing ${data.violations.flatMap(item => item.missing_items).join(", ")}. Please correct the PPE before entry.`;
  } else if (data.status === "GO AHEAD") {
    resultIcon.classList.add("passed");
    resultIcon.textContent = "✓";
    resultMessage.textContent = "Helmet and face mask detected. You may proceed.";
  } else if (data.status === "SCANNING - PLEASE WAIT") {
    resultIcon.classList.add("scanning");
    resultIcon.textContent = "…";
    resultMessage.textContent = "Checking for a person, helmet, and face mask.";
  } else {
    resultIcon.classList.add("waiting");
    resultIcon.textContent = "•";
    resultMessage.textContent = cameraStream
      ? "Step into view for a one-time PPE check."
      : "Start the camera and step into the marked area.";
  }

  document.querySelector("#passed-count").textContent = data.passed_count;
  document.querySelector("#failed-count").textContent = data.failed_count;
  const counts = Object.entries(data.missing_item_counts || {});
  const highest = Math.max(0, ...counts.map(([, count]) => count));
  const missed = counts.filter(([, count]) => count === highest && count > 0);
  document.querySelector("#most-missed").textContent = missed.length
    ? `${missed.map(([item]) => item).join(" / ")} (${highest})`
    : "None yet";

  errorMessage.hidden = !data.error;
  errorMessage.textContent = data.error || "";
  drawDetections(data.people || [], data.violations || []);
}

function drawDetections(people, violations) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return;
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
  overlayContext.clearRect(0, 0, width, height);
  for (const person of people) {
    const failure = violations.find(item => {
      const box = item.person_bbox;
      return Math.abs(box.x1 - person.x1) <= 1
        && Math.abs(box.y1 - person.y1) <= 1;
    });
    const color = failure ? "#ff5c62" : "#48d597";
    overlayContext.strokeStyle = color;
    overlayContext.lineWidth = Math.max(2, width / 320);
    overlayContext.strokeRect(
      person.x1,
      person.y1,
      person.x2 - person.x1,
      person.y2 - person.y1,
    );
    const label = failure
      ? `MISSING: ${failure.missing_items.join(", ")}`
      : "HELMET + MASK OK";
    overlayContext.font = `600 ${Math.max(14, width / 48)}px sans-serif`;
    overlayContext.fillStyle = color;
    overlayContext.fillText(label, person.x1, Math.max(20, person.y1 - 8));
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Status request failed (${response.status}).`);
    setConnection(true);
    updateResult(await response.json());
  } catch (error) {
    setConnection(false);
    errorMessage.hidden = false;
    errorMessage.textContent = error.message;
  }
}

async function submitFrame() {
  if (!cameraStream || sendingFrame || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  sendingFrame = true;
  try {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.72));
    if (!blob) throw new Error("Could not capture a camera frame.");
    const response = await fetch("/api/frame", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || `Frame upload failed (${response.status}).`);
    }
  } catch (error) {
    showCameraError(error.message);
  } finally {
    sendingFrame = false;
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showCameraError("Camera access requires a secure HTTPS connection and a supported browser.");
    return;
  }
  startButton.disabled = true;
  cameraMessage.textContent = "Waiting for camera permission…";
  cameraMessage.classList.remove("error-text");
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    });
    video.srcObject = cameraStream;
    await video.play();
    placeholder.hidden = true;
    document.querySelector("#camera-label").textContent = "Camera active";
    cameraMessage.textContent = "Camera is on. Step into the marked area.";
    stopButton.disabled = false;
    frameTimer = window.setInterval(submitFrame, 500);
    statusTimer = window.setInterval(refreshStatus, 600);
    await submitFrame();
  } catch (error) {
    stopCamera();
    showCameraError(error.name === "NotAllowedError"
      ? "Camera access was denied. Allow camera permission in your browser and try again."
      : `Could not start the camera: ${error.message}`);
  } finally {
    startButton.disabled = Boolean(cameraStream);
  }
}

function stopCamera() {
  if (frameTimer) window.clearInterval(frameTimer);
  if (statusTimer) window.clearInterval(statusTimer);
  frameTimer = null;
  statusTimer = null;
  if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
  cameraStream = null;
  video.srcObject = null;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  placeholder.hidden = false;
  startButton.disabled = false;
  stopButton.disabled = true;
  document.querySelector("#camera-label").textContent = "Camera off";
  cameraMessage.textContent = "Camera permission is requested only after you start.";
  cameraMessage.classList.remove("error-text");
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("pagehide", stopCamera);
refreshStatus();

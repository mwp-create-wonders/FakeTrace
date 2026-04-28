const fileInput = document.querySelector("#fileInput");
const pickButton = document.querySelector("#pickButton");
const runButton = document.querySelector("#runButton");
const clearButton = document.querySelector("#clearButton");
const dropzone = document.querySelector("#dropzone");
const results = document.querySelector("#results");
const statusEl = document.querySelector("#status");
const summary = document.querySelector("#summary");

let selectedFiles = [];
let previewUrls = [];

function setButtons() {
  const hasFiles = selectedFiles.length > 0;
  runButton.disabled = !hasFiles;
  clearButton.disabled = !hasFiles;
}

function clearPreviewUrls() {
  previewUrls.forEach((url) => URL.revokeObjectURL(url));
  previewUrls = [];
}

function setMessage(text) {
  summary.hidden = true;
  results.innerHTML = `<div class="message">${text}</div>`;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderSummary(text) {
  summary.hidden = false;
  summary.textContent = text;
}

function renderSelected(files) {
  if (!files.length) {
    setMessage("请选择一张或多张图片开始检测。");
    return;
  }

  clearPreviewUrls();
  previewUrls = files.map((file) => URL.createObjectURL(file));
  renderSummary(`已选择 ${files.length} 张图片，点击“开始检测”进行批量推理。`);

  results.innerHTML = "";
  files.forEach((file, index) => {
    const card = document.createElement("article");
    card.className = "result-card pending-card";

    const img = document.createElement("img");
    img.className = "thumb";
    img.alt = file.name;
    img.src = previewUrls[index];

    card.innerHTML = `
      <div class="result-body">
        <p class="filename" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</p>
        <span class="badge pending">待检测</span>
      </div>
    `;
    card.prepend(img);
    results.appendChild(card);
  });
}

function setFiles(files) {
  selectedFiles = Array.from(files).filter((file) => file.type.startsWith("image/"));
  renderSelected(selectedFiles);
  setButtons();
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "模型状态读取失败。");
    }
    statusEl.className = "status ready";
    statusEl.textContent = `已就绪 | ${data.device} | ${data.image_size}px | batch ${data.batch_size}`;
  } catch (error) {
    statusEl.className = "status error";
    statusEl.textContent = "模型未就绪";
    setMessage(error.message);
  }
}

async function runPrediction() {
  if (!selectedFiles.length) {
    return;
  }

  runButton.disabled = true;
  runButton.textContent = "检测中...";
  renderSummary(`正在检测 ${selectedFiles.length} 张图片，请稍候。`);

  const formData = new FormData();
  selectedFiles.forEach((file) => formData.append("files", file));

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "检测失败。");
    }
    renderResults(data.results);
  } catch (error) {
    setMessage(error.message);
  } finally {
    runButton.textContent = "开始检测";
    setButtons();
  }
}

function renderResults(items) {
  const realCount = items.filter((item) => item.prediction === "real").length;
  const fakeCount = items.length - realCount;
  renderSummary(`检测完成：共 ${items.length} 张，真实 ${realCount} 张，AI 生成 ${fakeCount} 张。`);

  results.innerHTML = "";
  items.forEach((item, index) => {
    const realPercent = Math.round(item.real_probability * 1000) / 10;
    const fakePercent = Math.round(item.fake_probability * 1000) / 10;
    const confidence = item.prediction === "real" ? realPercent : fakePercent;
    const card = document.createElement("article");
    card.className = `result-card ${item.prediction}`;

    const img = document.createElement("img");
    img.className = "thumb";
    img.alt = item.filename;
    img.src = previewUrls[index] || "";

    card.innerHTML = `
      <div class="result-body">
        <div class="result-head">
          <p class="filename" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</p>
          <span class="badge ${item.prediction}">${item.prediction === "real" ? "Real" : "Fake"}</span>
        </div>
        <p class="verdict">${item.prediction === "real" ? "判定为真实图片" : "判定为 AI 生成图片"}</p>
        <div class="prob">
          <div class="prob-row"><span>真实概率</span><strong>${realPercent.toFixed(1)}%</strong></div>
          <div class="bar"><span style="width: ${realPercent}%"></span></div>
          <div class="prob-row"><span>伪造概率</span><strong>${fakePercent.toFixed(1)}%</strong></div>
          <div class="confidence">分类置信度 ${confidence.toFixed(1)}%</div>
        </div>
      </div>
    `;
    card.prepend(img);
    results.appendChild(card);
  });
}

pickButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => setFiles(event.target.files));
runButton.addEventListener("click", runPrediction);
clearButton.addEventListener("click", () => {
  selectedFiles = [];
  fileInput.value = "";
  clearPreviewUrls();
  setMessage("请选择一张或多张图片开始检测。");
  setButtons();
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});

dropzone.addEventListener("drop", (event) => {
  setFiles(event.dataTransfer.files);
});

setMessage("请选择一张或多张图片开始检测。");
setButtons();
loadStatus();

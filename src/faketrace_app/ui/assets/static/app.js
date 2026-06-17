const fileInput = document.querySelector("#fileInput");
const pickButton = document.querySelector("#pickButton");
const runButton = document.querySelector("#runButton");
const clearButton = document.querySelector("#clearButton");
const dropzone = document.querySelector("#dropzone");
const results = document.querySelector("#results");
const statusEl = document.querySelector("#status");
const summary = document.querySelector("#summary");
const dropTitle = document.querySelector("#dropTitle");
const dropHint = document.querySelector("#dropHint");
const tabs = Array.from(document.querySelectorAll(".feature-tab"));
const modelSelector = document.querySelector("#modelSelector");
const modelSelectLabel = document.querySelector("#modelSelectLabel");
const modelSelect = document.querySelector("#modelSelect");

const featureConfig = {
  detector: {
    title: "拖拽图片到这里，或点击选择文件",
    hint: "真伪检测支持一次上传多张图片，返回 Real / Fake 判断与概率分布。",
    endpoint: "/api/predict",
    pickLabel: "选择图片",
    actionLabel: "开始检测",
    busyLabel: "检测中...",
    emptyMessage: "请选择需要做真伪检测的图片。",
    pendingBadge: "待检测",
    mediaLabel: "图片",
    selectedSummary: (count) => `已选择 ${count} 张图片，确认后可以开始真伪检测。`,
    runningSummary: (count) => `正在检测 ${count} 张图片，请稍候。`,
  },
  localizer: {
    title: "拖拽图片到这里，或点击选择文件",
    hint: "篡改定位会返回定位热力图、叠加结果、置信图和整体篡改分数。",
    endpoint: "/api/localize",
    pickLabel: "选择图片",
    actionLabel: "开始定位",
    busyLabel: "定位中...",
    emptyMessage: "请选择需要做篡改定位的图片。",
    pendingBadge: "待定位",
    mediaLabel: "图片",
    selectedSummary: (count) => `已选择 ${count} 张图片，确认后开始篡改区域定位。`,
    runningSummary: (count) => `正在对 ${count} 张图片进行篡改定位，请稍候。`,
  },
  audio: {
    title: "拖拽音频到这里，或点击选择文件",
    hint: "音频检测支持 wav、flac、mp3、m4a、ogg、aac，返回 Real / Fake 判断与概率分布。",
    endpoint: "/api/audio/predict",
    pickLabel: "选择音频",
    actionLabel: "开始检测",
    busyLabel: "音频检测中...",
    emptyMessage: "请选择需要做真伪检测的音频。",
    pendingBadge: "待检测",
    mediaLabel: "音频",
    selectedSummary: (count) => `已选择 ${count} 个音频文件，确认后可以开始真伪检测。`,
    runningSummary: (count) => `正在检测 ${count} 个音频文件，请稍候。`,
  },
};

const audioExtensions = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"];
const modelOptions = {
  detector: {
    label: "检测模型：",
    disabled: true,
    options: [{ value: "marc", label: "MARC" }],
  },
  localizer: {
    label: "定位模型：",
    disabled: false,
    options: [
      { value: "trufor", label: "TruFor" },
      { value: "catnet", label: "CAT-Net" },
      { value: "fassa", label: "Fassa" },
      { value: "effunetpp", label: "EffUnetPP" },
    ],
  },
  audio: {
    label: "音频模型：",
    disabled: true,
    options: [{ value: "atadd_ast", label: "ATADD AST" }],
  },
};

function initialModeFromLocation() {
  const hashMode = window.location.hash.replace("#", "");
  return featureConfig[hashMode] ? hashMode : "detector";
}

let mode = initialModeFromLocation();
let selectedFiles = [];
let previewUrls = [];

function currentFeature() {
  return featureConfig[mode];
}

function clearPreviewUrls() {
  previewUrls.forEach((url) => URL.revokeObjectURL(url));
  previewUrls = [];
}

function setButtons() {
  const feature = currentFeature();
  const hasFiles = selectedFiles.length > 0;
  pickButton.textContent = feature.pickLabel;
  runButton.disabled = !hasFiles;
  clearButton.disabled = !hasFiles;
  runButton.textContent = feature.actionLabel;
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
  const feature = currentFeature();
  if (!files.length) {
    setMessage(feature.emptyMessage);
    return;
  }

  clearPreviewUrls();
  previewUrls = mode === "audio" ? [] : files.map((file) => URL.createObjectURL(file));
  renderSummary(feature.selectedSummary(files.length));

  results.innerHTML = "";
  files.forEach((file, index) => {
    const card = document.createElement("article");
    card.className = "result-card pending-card";

    let mediaNode;
    if (mode === "audio") {
      mediaNode = document.createElement("div");
      mediaNode.className = "audio-thumb";
      mediaNode.textContent = "音频";
    } else {
      mediaNode = document.createElement("img");
      mediaNode.className = "thumb";
      mediaNode.alt = file.name;
      mediaNode.src = previewUrls[index];
    }

    card.innerHTML = `
      <div class="result-body">
        <p class="filename" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</p>
        <span class="badge pending">${feature.pendingBadge}</span>
      </div>
    `;
    card.prepend(mediaNode);
    results.appendChild(card);
  });
}

function resetSelection() {
  selectedFiles = [];
  fileInput.value = "";
  clearPreviewUrls();
  setButtons();
  setMessage(currentFeature().emptyMessage);
}

function setFiles(files) {
  selectedFiles = Array.from(files).filter((file) => {
    if (mode === "audio") {
      const name = file.name.toLowerCase();
      return file.type.startsWith("audio/") || audioExtensions.some((ext) => name.endsWith(ext));
    }
    return file.type.startsWith("image/");
  });
  renderSelected(selectedFiles);
  setButtons();
}

function applyModeUi() {
  const feature = currentFeature();
  const models = modelOptions[mode];
  dropTitle.textContent = feature.title;
  dropHint.textContent = feature.hint;
  fileInput.accept = mode === "audio" ? "audio/*,.wav,.flac,.mp3,.ogg,.m4a,.aac" : "image/*";
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === mode));
  modelSelector.hidden = false;
  modelSelectLabel.textContent = models.label;
  modelSelect.disabled = models.disabled;
  modelSelect.innerHTML = models.options
    .map((item) => `<option value="${item.value}">${item.label}</option>`)
    .join("");
  resetSelection();
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function readinessText(name, status) {
  return status?.ready ? `${name}: ${status.device}` : `${name}: 未就绪`;
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "状态接口返回异常。");
    }

    const detectorText = readinessText("MARC", data.detector);
    const audioText = readinessText("ATADD", data.audio);
    const truforText = readinessText("TruFor", data.trufor);

    statusEl.className = "status ready";
    statusEl.textContent = `${detectorText} | ${audioText} | ${truforText}`;
  } catch (error) {
    statusEl.className = "status error";
    statusEl.textContent = "模型状态检查失败";
    setMessage(error.message);
  }
}

async function runCurrentFeature() {
  if (!selectedFiles.length) {
    return;
  }

  const feature = currentFeature();
  runButton.disabled = true;
  runButton.textContent = feature.busyLabel;
  renderSummary(feature.runningSummary(selectedFiles.length));

  const formData = new FormData();
  selectedFiles.forEach((file) => formData.append("files", file));

  let endpoint = feature.endpoint;
  if (mode === "localizer") {
    endpoint += `?model=${modelSelect.value}`;
  }

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "处理请求失败。");
    }

    if (mode === "detector") {
      renderDetectorResults(data.results);
    } else if (mode === "audio") {
      renderAudioResults(data.results, data.meta);
    } else {
      renderLocalizerResults(data.results, data.meta?.model);
    }
  } catch (error) {
    setMessage(error.message);
  } finally {
    runButton.textContent = feature.actionLabel;
    setButtons();
  }
}

function renderDetectorResults(items) {
  const realCount = items.filter((item) => item.prediction === "real").length;
  const fakeCount = items.length - realCount;
  renderSummary(`检测完成，共 ${items.length} 张图片，其中真实 ${realCount} 张，疑似伪造 ${fakeCount} 张。`);

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
        <p class="verdict">${item.prediction === "real" ? "更接近真实图片分布。" : "更接近伪造图片分布。"}</p>
        <div class="prob">
          <div class="prob-row"><span>真实概率</span><strong>${realPercent.toFixed(1)}%</strong></div>
          <div class="bar"><span style="width: ${realPercent}%"></span></div>
          <div class="prob-row"><span>伪造概率</span><strong>${fakePercent.toFixed(1)}%</strong></div>
          <div class="confidence">当前判断置信度：${confidence.toFixed(1)}%</div>
        </div>
      </div>
    `;
    card.prepend(img);
    results.appendChild(card);
  });
}

function renderAudioResults(items, meta = {}) {
  const realCount = items.filter((item) => item.prediction === "real").length;
  const fakeCount = items.length - realCount;
  const modelName = meta?.model || "ATADD";
  renderSummary(`音频检测完成（${modelName}），共 ${items.length} 个音频，其中真实 ${realCount} 个，疑似伪造 ${fakeCount} 个。`);

  results.innerHTML = "";
  items.forEach((item) => {
    const realPercent = Math.round(item.real_probability * 1000) / 10;
    const fakePercent = Math.round(item.fake_probability * 1000) / 10;
    const confidence = item.prediction === "real" ? realPercent : fakePercent;
    const card = document.createElement("article");
    card.className = `result-card audio-card ${item.prediction}`;

    const icon = document.createElement("div");
    icon.className = "audio-thumb";
    icon.textContent = item.prediction === "real" ? "真实" : "伪造";

    card.innerHTML = `
      <div class="result-body">
        <div class="result-head">
          <p class="filename" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</p>
          <span class="badge ${item.prediction}">${item.prediction === "real" ? "Real" : "Fake"}</span>
        </div>
        <p class="verdict">${item.prediction === "real" ? "更接近真实音频分布。" : "更接近伪造音频分布。"}</p>
        <div class="prob">
          <div class="prob-row"><span>真实概率</span><strong>${realPercent.toFixed(1)}%</strong></div>
          <div class="bar"><span style="width: ${realPercent}%"></span></div>
          <div class="prob-row"><span>伪造概率</span><strong>${fakePercent.toFixed(1)}%</strong></div>
          <div class="confidence">当前判断置信度：${confidence.toFixed(1)}%</div>
        </div>
      </div>
    `;
    card.prepend(icon);
    results.appendChild(card);
  });
}

function renderLocalizerResults(items, modelName = "TruFor") {
  const suspiciousAverage = items.length
    ? items.reduce((sum, item) => sum + item.suspicious_ratio, 0) / items.length
    : 0;
  renderSummary(`定位完成（${modelName}），共 ${items.length} 张图片，平均可疑区域占比 ${formatPercent(suspiciousAverage)}。`);

  results.innerHTML = "";
  items.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "result-card localizer-card";

    const source = previewUrls[index] || "";
    const scoreText = item.score === null ? "未提供" : formatPercent(item.score);
    const confidenceBlock = item.confidence_map_url
      ? `<div class="image-panel"><img class="thumb detail-thumb" alt="confidence map" src="${item.confidence_map_url}"><span>置信图</span></div>`
      : "";

    card.innerHTML = `
      <div class="result-body">
        <div class="result-head">
          <p class="filename" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</p>
          <span class="badge localizer">${modelName}</span>
        </div>
        <p class="verdict">整体篡改分数：${scoreText}</p>
        <div class="prob">
          <div class="prob-row"><span>可疑区域占比</span><strong>${formatPercent(item.suspicious_ratio)}</strong></div>
        </div>
        <div class="image-grid">
          <div class="image-panel"><img class="thumb detail-thumb" alt="source image" src="${source}"><span>原图</span></div>
          <div class="image-panel"><img class="thumb detail-thumb" alt="overlay image" src="${item.overlay_url}"><span>叠加图</span></div>
          <div class="image-panel"><img class="thumb detail-thumb" alt="localization map" src="${item.localization_map_url}"><span>定位热力图</span></div>
          ${confidenceBlock}
        </div>
      </div>
    `;
    results.appendChild(card);
  });
}

pickButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => setFiles(event.target.files));
runButton.addEventListener("click", runCurrentFeature);
clearButton.addEventListener("click", resetSelection);

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    mode = tab.dataset.mode;
    window.history.replaceState(null, "", `#${mode}`);
    applyModeUi();
  });
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

applyModeUi();
loadStatus();

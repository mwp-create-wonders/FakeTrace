const fileInput = document.querySelector("#fileInput");
const runButton = document.querySelector("#runButton");
const clearButton = document.querySelector("#clearButton");
const dropzone = document.querySelector("#dropzone");
const results = document.querySelector("#results");
const statusEl = document.querySelector("#status");
const summary = document.querySelector("#summary");
const dropLabel = document.querySelector("#dropLabel");
const dropTitle = document.querySelector("#dropTitle");
const dropHint = document.querySelector("#dropHint");
const modelSelector = document.querySelector("#modelSelector");
const modelSelectLabel = document.querySelector("#modelSelectLabel");
const modelSelect = document.querySelector("#modelSelect");
const modeTitle = document.querySelector("#modeTitle");
const modeDescription = document.querySelector("#modeDescription");
const tabs = Array.from(document.querySelectorAll(".feature-tab"));

const featureConfig = {
  detector: {
    label: "图像检测",
    title: "拖拽图像到这里，或点击选择文件",
    hint: "支持批量图片真伪识别。",
    endpoint: "/api/predict",
    actionLabel: "开始检测",
    busyLabel: "检测中...",
    accept: "image/*,.png,.jpg,.jpeg,.bmp,.webp",
    emptyMessage: "请选择需要进行图像检测的文件。",
    pendingBadge: "待检测",
    modeDescription: "批量图片真伪识别与概率输出",
    requiresFiles: true,
  },
  localizer: {
    label: "篡改定位",
    title: "拖拽图像到这里，或点击选择文件",
    hint: "支持篡改热力图与叠加图输出。",
    endpoint: "/api/localize",
    actionLabel: "开始定位",
    busyLabel: "定位中...",
    accept: "image/*,.png,.jpg,.jpeg,.bmp,.webp",
    emptyMessage: "请选择需要进行篡改定位的图像。",
    pendingBadge: "待定位",
    modeDescription: "篡改区域热力图与可视化输出",
    requiresFiles: true,
  },
  audio: {
    label: "音频检测",
    title: "拖拽音频到这里，或点击选择文件",
    hint: "支持 wav、flac、mp3、m4a、ogg、aac。",
    endpoint: "/api/audio/predict",
    actionLabel: "开始检测",
    busyLabel: "检测中...",
    accept: "audio/*,.wav,.flac,.mp3,.ogg,.m4a,.aac",
    emptyMessage: "请选择需要进行音频检测的文件。",
    pendingBadge: "待检测",
    modeDescription: "音频真伪识别与概率输出",
    requiresFiles: true,
  },
  video: {
    label: "视频检测",
    title: "拖拽视频到这里，或点击选择文件",
    hint: "当前接入 TRI 视频检测模型。",
    endpoint: "/api/video/predict",
    actionLabel: "开始检测",
    busyLabel: "检测中...",
    accept: "video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v",
    emptyMessage: "请选择需要进行视频检测的文件。",
    pendingBadge: "待检测",
    modeDescription: "视频真伪识别与概率输出",
    requiresFiles: true,
  },
  security: {
    label: "安全扫描",
    title: "执行项目依赖与静态安全扫描",
    hint: "无需上传文件，直接对当前项目执行扫描。",
    endpoint: "/api/security/scan",
    actionLabel: "开始扫描",
    busyLabel: "扫描中...",
    accept: "",
    emptyMessage: "点击开始扫描即可执行项目安全测试。",
    pendingBadge: "待扫描",
    modeDescription: "依赖漏洞与静态代码扫描",
    requiresFiles: false,
  },
};

const modelOptions = {
  detector: [
    ["marc", "MARC"],
    ["forensic_moe", "Forensic-MoE"],
    ["forgelens", "ForgeLens"],
    ["lota", "LOTA"],
    ["mf2da", "MF2DA"],
    ["univfd", "UnivFD"],
  ],
  localizer: [
    ["trufor", "TruFor"],
    ["catnet", "CAT-Net"],
    ["fassa", "Fassa"],
    ["effunetpp", "EffUnetPP"],
  ],
  audio: [["default", "ATADD AST"]],
  video: [["tri", "TRI"]],
  security: [],
};

const imageExtensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"];
const audioExtensions = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"];
const videoExtensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];

let mode = window.location.hash.replace("#", "") || "detector";
if (!(mode in featureConfig)) {
  mode = "detector";
}
let selectedFiles = [];
let previewUrls = [];

function currentFeature() {
  return featureConfig[mode];
}

function currentModelLabel() {
  const option = modelSelect.options[modelSelect.selectedIndex];
  return option ? option.textContent : "";
}

function clearPreviewUrls() {
  previewUrls.forEach((url) => URL.revokeObjectURL(url));
  previewUrls = [];
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setMessage(text) {
  summary.hidden = true;
  results.innerHTML = `<div class="message">${escapeHtml(text)}</div>`;
}

function renderSummary(text) {
  summary.hidden = false;
  summary.textContent = text;
}

function setButtons() {
  const feature = currentFeature();
  const hasFiles = selectedFiles.length > 0;
  runButton.disabled = feature.requiresFiles ? !hasFiles : false;
  clearButton.disabled = !hasFiles;
  runButton.textContent = feature.actionLabel;
}

function extensionAllowed(name, extensions) {
  const lower = (name || "").toLowerCase();
  return extensions.some((ext) => lower.endsWith(ext));
}

function isAcceptedFile(file) {
  if (mode === "audio") {
    return file.type.startsWith("audio/") || extensionAllowed(file.name, audioExtensions);
  }
  if (mode === "video") {
    return file.type.startsWith("video/") || extensionAllowed(file.name, videoExtensions);
  }
  return file.type.startsWith("image/") || extensionAllowed(file.name, imageExtensions);
}

function createPreviewNode(file, url, index) {
  if (mode === "audio") {
    const node = document.createElement("div");
    node.className = "thumb audio-thumb";
    node.textContent = "AUDIO";
    return node;
  }

  if (mode === "video") {
    const node = document.createElement("video");
    node.className = "thumb";
    node.src = url;
    node.controls = true;
    node.preload = "metadata";
    node.muted = true;
    node.playsInline = true;
    node.setAttribute("aria-label", file.name || `video-${index}`);
    return node;
  }

  const node = document.createElement("img");
  node.className = "thumb";
  node.alt = file.name;
  node.src = url;
  return node;
}

function renderSelected(files) {
  if (!files.length) {
    setMessage(currentFeature().emptyMessage);
    return;
  }

  clearPreviewUrls();
  previewUrls = mode === "audio" ? [] : files.map((file) => URL.createObjectURL(file));
  renderSummary(`已选择 ${files.length} 个文件，模型 ${currentModelLabel() || "默认"} 已就绪。`);
  results.innerHTML = "";

  files.forEach((file, index) => {
    const card = document.createElement("article");
    card.className = "result-card";
    const previewNode = createPreviewNode(file, previewUrls[index], index);
    card.innerHTML = `
      <div class="result-body">
        <div class="result-head">
          <p class="filename">${escapeHtml(file.name)}</p>
          <span class="badge pending">${currentFeature().pendingBadge}</span>
        </div>
      </div>
    `;
    card.prepend(previewNode);
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
  selectedFiles = Array.from(files).filter(isAcceptedFile);
  renderSelected(selectedFiles);
  setButtons();
}

function configureModelSelector() {
  const options = modelOptions[mode];
  if (!options.length) {
    modelSelector.hidden = true;
    return;
  }

  modelSelector.hidden = false;
  modelSelectLabel.textContent = `${currentFeature().label}模型`;
  modelSelect.innerHTML = options
    .map(([value, label]) => `<option value="${value}">${label}</option>`)
    .join("");
  modelSelect.disabled = mode === "audio" || mode === "video";
}

function applyModeUi() {
  const feature = currentFeature();
  dropLabel.textContent = feature.label;
  dropTitle.textContent = feature.title;
  dropHint.textContent = feature.hint;
  modeTitle.textContent = feature.label;
  modeDescription.textContent = feature.modeDescription;
  fileInput.accept = feature.accept;
  fileInput.disabled = !feature.requiresFiles;
  dropzone.classList.toggle("disabled", !feature.requiresFiles);
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === mode));
  configureModelSelector();
  resetSelection();
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function summarizeReadyModels(data) {
  const readyImageModels = Object.values(data.image_models || {})
    .filter((item) => item?.ready)
    .map((item) => item.model || "图像模型");
  const parts = [];

  if (readyImageModels.length) {
    parts.push(`图像: ${readyImageModels.join(" / ")}`);
  }
  if (data.trufor?.ready) {
    parts.push("定位: TruFor");
  }
  if (data.audio?.ready) {
    parts.push(`音频: ${data.audio.model || "ATADD"}`);
  }
  if (data.video?.ready) {
    parts.push(`视频: ${data.video.model || "TRI"}`);
  }
  return parts.length ? parts.join(" | ") : "暂无可用模型";
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "状态接口返回异常。");
    }
    statusEl.className = "status ready";
    statusEl.textContent = summarizeReadyModels(data);
  } catch (error) {
    statusEl.className = "status error";
    statusEl.textContent = "模型状态检查失败";
  }
}

function buildEndpoint() {
  const feature = currentFeature();
  if (mode === "detector" || mode === "localizer" || mode === "video") {
    return `${feature.endpoint}?model=${encodeURIComponent(modelSelect.value)}`;
  }
  return feature.endpoint;
}

async function runCurrentFeature() {
  if (currentFeature().requiresFiles && !selectedFiles.length) {
    return;
  }

  const feature = currentFeature();
  runButton.disabled = true;
  runButton.textContent = feature.busyLabel;
  renderSummary(`${feature.label}执行中，请稍候。`);

  try {
    let response;
    if (mode === "security") {
      response = await fetch(feature.endpoint, { method: "POST" });
    } else {
      const formData = new FormData();
      selectedFiles.forEach((file) => formData.append("files", file));
      response = await fetch(buildEndpoint(), { method: "POST", body: formData });
    }

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "请求处理失败。");
    }

    if (mode === "localizer") {
      renderLocalizerResults(data.results, data.meta?.model || currentModelLabel());
    } else if (mode === "security") {
      renderSecurityResults(data);
    } else {
      renderBinaryResults(data.results, data.meta?.model || currentModelLabel(), feature.label);
    }
  } catch (error) {
    setMessage(error.message);
  } finally {
    setButtons();
  }
}

function renderBinaryResults(items, modelName, mediaName) {
  const realCount = items.filter((item) => item.prediction === "real").length;
  const fakeCount = items.filter((item) => item.prediction === "fake").length;
  renderSummary(`${modelName} ${mediaName}完成，共 ${items.length} 个结果，真实 ${realCount}，伪造 ${fakeCount}。`);

  results.innerHTML = "";
  items.forEach((item, index) => {
    const realPercent = Number(item.real_probability || 0) * 100;
    const fakePercent = Number(item.fake_probability || 0) * 100;
    const card = document.createElement("article");
    card.className = "result-card";
    const previewNode = mode === "audio"
      ? createPreviewNode({ name: item.filename }, "", index)
      : createPreviewNode(selectedFiles[index] || { name: item.filename }, previewUrls[index], index);

    card.innerHTML = `
      <div class="result-body">
        <div class="result-head">
          <p class="filename">${escapeHtml(item.filename)}</p>
          <span class="badge ${item.prediction === "real" ? "real" : "fake"}">${item.prediction}</span>
        </div>
        <p class="verdict">真实概率 ${realPercent.toFixed(1)}%，伪造概率 ${fakePercent.toFixed(1)}%。</p>
      </div>
    `;
    card.prepend(previewNode);
    results.appendChild(card);
  });
}

function renderLocalizerResults(items, modelName) {
  const average = items.length
    ? items.reduce((sum, item) => sum + Number(item.suspicious_ratio || 0), 0) / items.length
    : 0;
  renderSummary(`${modelName} 定位完成，共 ${items.length} 张图像，平均可疑区域占比 ${formatPercent(average)}。`);

  results.innerHTML = "";
  items.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "result-card localizer-card";
    card.innerHTML = `
      <div class="result-body">
        <div class="result-head">
          <p class="filename">${escapeHtml(item.filename)}</p>
          <span class="badge pending">${escapeHtml(modelName)}</span>
        </div>
        <p class="verdict">可疑区域占比 ${formatPercent(item.suspicious_ratio || 0)}</p>
        <div class="image-grid">
          <img class="thumb detail-thumb" alt="source" src="${previewUrls[index] || ""}">
          <img class="thumb detail-thumb" alt="overlay" src="${item.overlay_url}">
          <img class="thumb detail-thumb" alt="localization" src="${item.localization_map_url}">
        </div>
      </div>
    `;
    results.appendChild(card);
  });
}

function renderSecurityResults(payload) {
  renderSummary(`安全扫描完成，共 ${payload.summary?.scan_count ?? 0} 项，发现 ${payload.summary?.risk_count ?? 0} 个风险。`);
  results.innerHTML = "";

  payload.results.forEach((item) => {
    const issues = item.issues.length
      ? item.issues.map((issue) => `<li>${escapeHtml(issue.id || issue.test_id || issue.package || "issue")} - ${escapeHtml(issue.description || "未提供描述")}</li>`).join("")
      : "<li>未发现风险</li>";

    const card = document.createElement("article");
    card.className = "result-card";
    card.innerHTML = `
      <div class="result-body">
        <div class="result-head">
          <p class="filename">${escapeHtml(item.title)}</p>
          <span class="badge ${item.status === "clean" ? "real" : item.status === "issues_found" ? "fake" : "pending"}">${escapeHtml(item.status)}</span>
        </div>
        <p class="verdict">${escapeHtml(item.summary)}</p>
        <ul class="issue-list">${issues}</ul>
      </div>
    `;
    results.appendChild(card);
  });
}

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
    if (!currentFeature().requiresFiles) {
      return;
    }
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

dropzone.addEventListener("click", () => {
  if (currentFeature().requiresFiles) {
    fileInput.click();
  }
});

dropzone.addEventListener("drop", (event) => {
  if (!currentFeature().requiresFiles) {
    return;
  }
  setFiles(event.dataTransfer.files);
});

applyModeUi();
loadStatus();

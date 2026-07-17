const fileInput = document.querySelector("#fileInput");
const pickButton = document.querySelector("#pickButton");
const runButton = document.querySelector("#runButton");
const clearButton = document.querySelector("#clearButton");
const results = document.querySelector("#results");
const statusEl = document.querySelector("#status");
const summary = document.querySelector("#summary");
const tabs = Array.from(document.querySelectorAll(".feature-tab"));
const modelSelector = document.querySelector("#modelSelector");
const modelSelectLabel = document.querySelector("#modelSelectLabel");
const modelSelect = document.querySelector("#modelSelect");

const imageExtensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"];
const audioExtensions = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"];
const videoExtensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];

const featureConfig = {
  detector: {
    endpoint: "/api/predict",
    pickLabel: "选择图像",
    actionLabel: "开始检测",
    busyLabel: "图像检测中...",
    emptyMessage: "请选择需要进行图像真伪检测的文件。",
    pendingBadge: "待检测",
    mediaLabel: "图像",
    accept: "image/*,.png,.jpg,.jpeg,.bmp,.webp",
    selectedSummary: (count) => `已选择 ${count} 张图像，确认后即可开始图像取证检测。`,
    runningSummary: (count, modelLabel) => `正在使用 ${modelLabel} 检测 ${count} 张图像，请稍候。`,
  },
  localizer: {
    endpoint: "/api/localize",
    pickLabel: "选择图像",
    actionLabel: "开始定位",
    busyLabel: "定位中...",
    emptyMessage: "请选择需要进行篡改定位的图像。",
    pendingBadge: "待定位",
    mediaLabel: "图像",
    accept: "image/*,.png,.jpg,.jpeg,.bmp,.webp",
    selectedSummary: (count) => `已选择 ${count} 张图像，确认后即可开始篡改区域定位。`,
    runningSummary: (count, modelLabel) => `正在使用 ${modelLabel} 对 ${count} 张图像进行定位，请稍候。`,
  },
  audio: {
    endpoint: "/api/audio/predict",
    pickLabel: "选择音频",
    actionLabel: "开始检测",
    busyLabel: "音频检测中...",
    emptyMessage: "请选择需要进行音频真伪检测的文件。",
    pendingBadge: "待检测",
    mediaLabel: "音频",
    accept: "audio/*,.wav,.flac,.mp3,.ogg,.m4a,.aac",
    selectedSummary: (count) => `已选择 ${count} 个音频文件，确认后即可开始音频取证检测。`,
    runningSummary: (count, modelLabel) => `正在使用 ${modelLabel} 检测 ${count} 个音频文件，请稍候。`,
  },
  video: {
    endpoint: "/api/video/predict",
    pickLabel: "选择视频",
    actionLabel: "开始检测",
    busyLabel: "视频检测中...",
    emptyMessage: "请选择需要进行视频真伪检测的文件。",
    pendingBadge: "待检测",
    mediaLabel: "视频",
    accept: "video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v",
    selectedSummary: (count) => `已选择 ${count} 个视频文件，确认后即可开始视频取证检测。`,
    runningSummary: (count, modelLabel) => `正在使用 ${modelLabel} 检测 ${count} 个视频文件，请稍候。`,
  },
};

const modelOptions = {
  detector: {
    label: "检测模型：",
    disabled: false,
    options: [
      { value: "marc", label: "MARC" },
      { value: "forensic_moe", label: "Forensic-MoE" },
      { value: "forgelens", label: "ForgeLens" },
      { value: "lota", label: "LOTA" },
      { value: "mf2da", label: "MF2DA" },
      { value: "univfd", label: "UnivFD" },
    ],
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
  video: {
    label: "视频模型：",
    disabled: true,
    options: [{ value: "tri", label: "TRI" }],
  },
};

function initialModeFromLocation() {
  const hashMode = window.location.hash.replace("#", "");
  return featureConfig[hashMode] ? hashMode : "detector";
}

let mode = initialModeFromLocation();
let selectedFiles = [];
let previewUrls = [];
let currentRunUploadTime = null;

function currentFeature() {
  return featureConfig[mode];
}

function currentModelConfig() {
  return modelOptions[mode];
}

function currentModelLabel() {
  const option = modelSelect.options[modelSelect.selectedIndex];
  return option ? option.textContent : "当前模型";
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
  results.innerHTML = `<div class="message">${escapeHtml(text)}</div>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderSummary(text) {
  summary.hidden = false;
  summary.textContent = text;
}

function createPreviewNode(file, url, index) {
  if (mode === "audio") {
    const node = document.createElement("div");
    node.className = "audio-thumb";
    node.textContent = "音频";
    return node;
  }

  if (mode === "video") {
    const node = document.createElement("video");
    node.className = "thumb video-thumb";
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
    const previewNode = createPreviewNode(file, previewUrls[index], index);
    card.innerHTML = `
      <div class="result-body">
        <p class="filename" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</p>
        <span class="badge pending">${feature.pendingBadge}</span>
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

function isAcceptedFile(file) {
  const name = file.name.toLowerCase();
  if (mode === "audio") {
    return file.type.startsWith("audio/") || audioExtensions.some((ext) => name.endsWith(ext));
  }
  if (mode === "video") {
    return file.type.startsWith("video/") || videoExtensions.some((ext) => name.endsWith(ext));
  }
  return file.type.startsWith("image/") || imageExtensions.some((ext) => name.endsWith(ext));
}

function setFiles(fileList) {
  selectedFiles = Array.from(fileList).filter((file) => isAcceptedFile(file));
  renderSelected(selectedFiles);
  setButtons();
}

function applyModeUi() {
  const feature = currentFeature();
  const models = currentModelConfig();
  fileInput.accept = feature.accept;
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
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function summarizeReadyModels(data) {
  const readyImageModels = [];
  if (data.image_models) {
    Object.values(data.image_models).forEach((item) => {
      if (item?.ready) {
        readyImageModels.push(item.model || "图像模型");
      }
    });
  }

  const sections = [];
  if (readyImageModels.length) {
    sections.push(`图像: ${readyImageModels.join(" / ")}`);
  }
  if (data.trufor?.ready) {
    sections.push("定位: TruFor");
  }
  if (data.audio?.ready) {
    sections.push(`音频: ${data.audio.model || "ATADD"}`);
  }
  if (data.video?.ready) {
    sections.push(`视频: ${data.video.model || "TRI"}`);
  }
  return sections.length ? sections.join(" | ") : "暂无可用模型";
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
    setMessage(error.message);
  }
}

function buildEndpoint() {
  const feature = currentFeature();
  const modelValue = modelSelect.value;
  if (mode === "detector" || mode === "localizer" || mode === "video") {
    return `${feature.endpoint}?model=${encodeURIComponent(modelValue)}`;
  }
  return feature.endpoint;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("读取图像失败。"));
    reader.readAsDataURL(file);
  });
}

function collectExtraFields(item) {
  const known = new Set([
    "filename",
    "suspicious_ratio",
    "localization_map_url",
    "overlay_url",
    "confidence_map_url",
    "score",
    "saved_files",
    "path",
  ]);
  return Object.fromEntries(
    Object.entries(item).filter(([key, value]) => !known.has(key) && value != null && typeof value !== "object")
  );
}

function renderReportActions(report) {
  const panel = document.createElement("article");
  panel.className = "result-card report-card";
  panel.innerHTML = `
    <div class="result-body">
      <div class="result-head">
        <p class="filename">取证报告已生成</p>
        <span class="badge localizer">PDF</span>
      </div>
      <p class="verdict">您可以在线浏览报告，也可以下载到本地。</p>
      <div class="report-actions">
        <a class="report-link" href="${report.pdf_url}" target="_blank" rel="noopener">在线浏览 PDF</a>
        <a class="report-link" href="${report.download_url}" download>下载 PDF</a>
      </div>
    </div>
  `;
  results.prepend(panel);
}

async function maybeOfferLocalizationReport(items, modelName, testId) {
  const wantsReport = window.confirm("伪造定位已完成。是否需要取证报告？");
  if (!wantsReport) {
    return;
  }

  const includeAiAnalysis = window.confirm("是否需要大模型为您解析定位结果？");
  renderSummary(includeAiAnalysis ? "正在生成取证报告，并调用大模型解析定位结果..." : "正在生成取证报告...");

  const originalImageUrls = await Promise.all(selectedFiles.map((file) => fileToDataUrl(file)));
  const reportItems = items.map((item, index) => ({
    filename: item.filename || selectedFiles[index]?.name || `image_${index + 1}.png`,
    original_image_url: originalImageUrls[index],
    localization_map_url: item.localization_map_url,
    overlay_url: item.overlay_url,
    suspicious_ratio: Number(item.suspicious_ratio || 0),
    score: item.score == null ? null : Number(item.score),
    confidence_map_url: item.confidence_map_url || null,
    extra_fields: collectExtraFields(item),
  }));

  const response = await fetch("/api/localize/report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: modelSelect.value,
      model_name: modelName,
      test_id: testId,
      upload_time: currentRunUploadTime,
      include_ai_analysis: includeAiAnalysis,
      items: reportItems,
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "报告生成失败。");
  }

  renderSummary("取证报告生成完成。");
  renderReportActions(data);
}

async function runCurrentFeature() {
  if (!selectedFiles.length) {
    return;
  }

  const feature = currentFeature();
  runButton.disabled = true;
  runButton.textContent = feature.busyLabel;
  renderSummary(feature.runningSummary(selectedFiles.length, currentModelLabel()));

  const formData = new FormData();
  selectedFiles.forEach((file) => formData.append("files", file));
  currentRunUploadTime = new Date().toISOString();

  try {
    const response = await fetch(buildEndpoint(), {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "请求处理失败。");
    }

    if (mode === "detector") {
      renderBinaryResults(data.results, data.meta?.model || currentModelLabel(), "图像");
    } else if (mode === "audio") {
      renderBinaryResults(data.results, data.meta?.model || currentModelLabel(), "音频");
    } else if (mode === "video") {
      renderBinaryResults(data.results, data.meta?.model || currentModelLabel(), "视频");
    } else {
      const modelName = data.meta?.model || currentModelLabel();
      renderLocalizerResults(data.results, modelName);
      window.setTimeout(() => {
        maybeOfferLocalizationReport(data.results, modelName, data.localization_test_id).catch((error) => {
          setMessage(error.message);
        });
      }, 80);
    }
  } catch (error) {
    setMessage(error.message);
  } finally {
    runButton.textContent = feature.actionLabel;
    setButtons();
  }
}

function renderBinaryResults(items, modelName, mediaName) {
  const realCount = items.filter((item) => item.prediction === "real").length;
  const fakeCount = items.filter((item) => item.prediction === "fake").length;
  const errorCount = items.filter((item) => item.prediction === "error").length;

  let summaryText = `${modelName} ${mediaName}检测完成，共 ${items.length} 个结果，真实 ${realCount}，疑似伪造 ${fakeCount}`;
  if (errorCount > 0) {
    summaryText += `，处理失败 ${errorCount}`;
  }
  renderSummary(summaryText);

  results.innerHTML = "";
  items.forEach((item, index) => {
    const realPercent = Number(item.real_probability || 0) * 100;
    const fakePercent = Number(item.fake_probability || 0) * 100;
    const confidence = item.prediction === "real" ? realPercent : fakePercent;
    const card = document.createElement("article");
    card.className = `result-card ${item.prediction === "error" ? "pending-card" : item.prediction}`;

    const previewNode = mode === "audio"
      ? (() => {
          const node = document.createElement("div");
          node.className = "audio-thumb";
          node.textContent = item.prediction === "real" ? "真实" : item.prediction === "fake" ? "伪造" : "异常";
          return node;
        })()
      : createPreviewNode(selectedFiles[index] || { name: item.filename }, previewUrls[index], index);

    const badgeText = item.prediction === "real"
      ? "Real"
      : item.prediction === "fake"
        ? "Fake"
        : "Error";
    const verdictText = item.prediction === "real"
      ? `更接近真实${mediaName}分布。`
      : item.prediction === "fake"
        ? `更接近伪造${mediaName}分布。`
        : "模型未能完成该文件的有效处理。";

    card.innerHTML = `
      <div class="result-body">
        <div class="result-head">
          <p class="filename" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</p>
          <span class="badge ${item.prediction === "error" ? "pending" : item.prediction}">${badgeText}</span>
        </div>
        <p class="verdict">${verdictText}</p>
        <div class="prob">
          <div class="prob-row"><span>真实概率</span><strong>${realPercent.toFixed(1)}%</strong></div>
          <div class="bar"><span style="width: ${realPercent}%"></span></div>
          <div class="prob-row"><span>伪造概率</span><strong>${fakePercent.toFixed(1)}%</strong></div>
          <div class="confidence">当前判断置信度：${confidence.toFixed(1)}%</div>
        </div>
      </div>
    `;

    if (previewNode) {
      card.prepend(previewNode);
    }
    results.appendChild(card);
  });
}

function renderLocalizerResults(items, modelName = "TruFor") {
  const suspiciousAverage = items.length
    ? items.reduce((sum, item) => sum + Number(item.suspicious_ratio || 0), 0) / items.length
    : 0;
  renderSummary(`${modelName} 定位完成，共 ${items.length} 张图像，平均可疑区域占比 ${formatPercent(suspiciousAverage)}。`);

  results.innerHTML = "";
  items.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "result-card localizer-card";

    const source = previewUrls[index] || "";
    const scoreText = item.score == null ? "未提供" : formatPercent(item.score);
    const confidenceBlock = item.confidence_map_url
      ? `<div class="image-panel"><img class="thumb detail-thumb" alt="confidence map" src="${item.confidence_map_url}"><span>置信图</span></div>`
      : "";

    card.innerHTML = `
      <div class="result-body">
        <div class="result-head">
          <p class="filename" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</p>
          <span class="badge localizer">${modelName}</span>
        </div>
        <p class="verdict">整体篡改得分：${scoreText}</p>
        <div class="prob">
          <div class="prob-row"><span>可疑区域占比</span><strong>${formatPercent(item.suspicious_ratio || 0)}</strong></div>
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

applyModeUi();
loadStatus();

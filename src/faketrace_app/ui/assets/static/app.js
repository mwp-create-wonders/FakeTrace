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
    options: [{ value: "ast_audioset_ft", label: "ATADD AST" }],
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
  if (mode === "detector" || mode === "localizer" || mode === "audio" || mode === "video") {
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

async function maybeOfferDetectorReport(items, modelName, testId) {
  const wantsReport = window.confirm("伪造检测已完成。是否需要取证报告？");
  if (!wantsReport) {
    return;
  }

  const includeAiAnalysis = window.confirm("是否需要大模型为您解析检测结果？");
  renderSummary(includeAiAnalysis ? "正在生成取证报告，并调用大模型解析检测结果..." : "正在生成取证报告...");

  const originalImageUrls = await Promise.all(selectedFiles.map((file) => fileToDataUrl(file)));
  const reportItems = items.map((item, index) => ({
    filename: item.filename || selectedFiles[index]?.name || `image_${index + 1}.png`,
    original_image_url: originalImageUrls[index],
    prediction: item.prediction,
    fake_probability: Number(item.fake_probability || 0),
    real_probability: Number(item.real_probability || 0),
  }));

  const response = await fetch("/api/predict/report", {
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

async function maybeOfferAudioReport(items, modelName, testId) {
  const wantsReport = window.confirm("语音伪造检测已完成。是否需要取证报告？");
  if (!wantsReport) {
    return;
  }

  const includeAiAnalysis = window.confirm("是否需要大模型为您解析检测结果？");
  renderSummary(includeAiAnalysis ? "正在生成取证报告，并调用大模型解析检测结果..." : "正在生成取证报告...");

  const audioUrls = await Promise.all(selectedFiles.map((file) => fileToDataUrl(file)));
  const reportItems = items.map((item, index) => ({
    filename: item.filename || selectedFiles[index]?.name || `audio_${index + 1}.wav`,
    audio_url: audioUrls[index],
    prediction: item.prediction,
    fake_probability: Number(item.fake_probability || 0),
    real_probability: Number(item.real_probability || 0),
  }));

  const response = await fetch("/api/audio/report", {
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

async function maybeOfferVideoReport(items, modelName, testId) {
  const wantsReport = window.confirm("视频伪造检测已完成。是否需要取证报告？");
  if (!wantsReport) {
    return;
  }

  const includeAiAnalysis = window.confirm("是否需要大模型为您解析检测结果？");
  renderSummary(includeAiAnalysis ? "正在生成取证报告，并调用大模型解析检测结果..." : "正在生成取证报告...");

  const videoUrls = await Promise.all(selectedFiles.map((file) => fileToDataUrl(file)));
  const reportItems = items.map((item, index) => ({
    filename: item.filename || selectedFiles[index]?.name || `video_${index + 1}.mp4`,
    video_url: videoUrls[index],
    duration: Number(item.duration || 0),
    width: Number(item.width || 0),
    height: Number(item.height || 0),
    fps: Number(item.fps || 0),
    total_frames: Number(item.total_frames || 0),
    prediction: item.prediction,
    fake_probability: Number(item.fake_probability || 0),
    real_probability: Number(item.real_probability || 0),
    threshold: Number(item.threshold || 0.5),
    frame_info: item.frame_info || null,
    velocity_l2: item.velocity_l2 || null,
    acceleration_l2: item.acceleration_l2 || null,
    lota_scores: item.lota_scores || null,
    suspicious_frame_b64: item.suspicious_frame_b64 || null,
    suspicious_frame_time: item.suspicious_frame_time != null ? Number(item.suspicious_frame_time) : null,
  }));

  const response = await fetch("/api/video/report", {
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
      const modelName = data.meta?.model || currentModelLabel();
      renderBinaryResults(data.results, modelName, "图像");
      window.setTimeout(() => {
        maybeOfferDetectorReport(data.results, modelName, data.detector_test_id).catch((error) => {
          setMessage(error.message);
        });
      }, 80);
    } else if (mode === "audio") {
      const modelName = data.meta?.model || currentModelLabel();
      renderBinaryResults(data.results, modelName, "音频");
      window.setTimeout(() => {
        maybeOfferAudioReport(data.results, modelName, data.audio_test_id).catch((error) => {
          setMessage(error.message);
        });
      }, 80);
    } else if (mode === "video") {
      const modelName = data.meta?.model || currentModelLabel();
      renderVideoResults(data.results, modelName);
      window.setTimeout(() => {
        maybeOfferVideoReport(data.results, modelName, data.video_test_id).catch((error) => {
          setMessage(error.message);
        });
      }, 80);
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

function _renderTrendChart(velocity, acceleration, lota, frameInfo) {
  if (!velocity || !acceleration || !lota || velocity.length < 2) {
    return "";
  }

  const width = 460;
  const height = 240;
  const padding = 45;
  const rightPad = 45;
  const chartWidth = width - padding - rightPad;
  const chartHeight = height - padding * 2;

  // X 轴：优先用 frameInfo 的秒数
  let xLabels = [];
  let useTime = false;
  if (frameInfo && frameInfo.length === velocity.length) {
    xLabels = frameInfo.map((fi) => Number(fi.target_time || 0).toFixed(2));
    useTime = true;
  } else {
    xLabels = velocity.map((_, i) => String(i + 1));
  }

  const vMax = Math.max(...velocity) * 1.1;
  const aMax = Math.max(...acceleration) * 1.1;
  const maxVal = Math.max(vMax, aMax, 1.0);

  const points = velocity.map((_, i) => ({
    x: padding + (i / (velocity.length - 1)) * chartWidth,
    v: padding + chartHeight - (velocity[i] / maxVal) * chartHeight,
    a: padding + chartHeight - (acceleration[i] / maxVal) * chartHeight,
    l: padding + chartHeight - (lota[i] / 1.0) * chartHeight,
  }));

  const vPath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.v}`).join(" ");
  const aPath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.a}`).join(" ");
  const lPath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.l}`).join(" ");

  const markers = points.map((p, i) => `
    <circle cx="${p.x}" cy="${p.v}" r="5" fill="#E74C3C" />
    <circle cx="${p.x}" cy="${p.a}" r="5" fill="#3498DB" />
    <circle cx="${p.x}" cy="${p.l}" r="5" fill="#2ECC71" />
  `).join("");

  // X 轴刻度标签
  const xTicks = points.map((p, i) =>
    `<text x="${p.x}" y="${padding + chartHeight + 18}" text-anchor="middle" font-size="11" fill="#ccc">${xLabels[i]}</text>`
  ).join("");

  // 左 Y 轴刻度（0, maxVal/2, maxVal）
  const yTicks = [0, 0.5, 1.0].map((ratio) => {
    const val = (maxVal * ratio).toFixed(1);
    const y = padding + chartHeight - ratio * chartHeight;
    return `<text x="${padding - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#ccc">${val}</text>`;
  }).join("");

  // 右 Y 轴刻度（LOTA 0, 0.5, 1.0）
  const rightYTicks = [0, 0.5, 1.0].map((ratio) => {
    const y = padding + chartHeight - ratio * chartHeight;
    return `<text x="${padding + chartWidth + 8}" y="${y + 4}" text-anchor="start" font-size="11" fill="#2ECC71">${ratio.toFixed(1)}</text>`;
  }).join("");

  const xLabelText = useTime ? "时间 (秒)" : "帧序号";

  return `
    <svg viewBox="0 0 ${width} ${height}" class="trend-chart">
      <defs>
        <linearGradient id="vGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#E74C3C" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="#E74C3C" stop-opacity="0"/>
        </linearGradient>
        <linearGradient id="aGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#3498DB" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="#3498DB" stop-opacity="0"/>
        </linearGradient>
        <linearGradient id="lGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#2ECC71" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="#2ECC71" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${padding + chartHeight}" stroke="#bbb" stroke-width="1.5"/>
      <line x1="${padding}" y1="${padding + chartHeight}" x2="${padding + chartWidth}" y2="${padding + chartHeight}" stroke="#bbb" stroke-width="1.5"/>
      <line x1="${padding + chartWidth}" y1="${padding}" x2="${padding + chartWidth}" y2="${padding + chartHeight}" stroke="#bbb" stroke-width="1.5"/>
      ${yTicks}
      ${rightYTicks}
      ${xTicks}
      <path d="${vPath} L ${points[points.length - 1].x} ${padding + chartHeight} L ${points[0].x} ${padding + chartHeight} Z" fill="url(#vGrad)" />
      <path d="${vPath}" stroke="#E74C3C" stroke-width="2.5" fill="none" />
      <path d="${aPath} L ${points[points.length - 1].x} ${padding + chartHeight} L ${points[0].x} ${padding + chartHeight} Z" fill="url(#aGrad)" />
      <path d="${aPath}" stroke="#3498DB" stroke-width="2.5" fill="none" />
      <path d="${lPath} L ${points[points.length - 1].x} ${padding + chartHeight} L ${points[0].x} ${padding + chartHeight} Z" fill="url(#lGrad)" />
      <path d="${lPath}" stroke="#2ECC71" stroke-width="2.5" fill="none" />
      ${markers}
      <text x="${width / 2}" y="${height - 5}" text-anchor="middle" font-size="14" fill="#ccc">${xLabelText}</text>
      <text x="${padding - 30}" y="${padding + chartHeight / 2}" text-anchor="middle" font-size="14" fill="#ccc" transform="rotate(-90, ${padding - 30}, ${padding + chartHeight / 2})">速度 / 加速度</text>
      <text x="${padding + chartWidth + 28}" y="${padding + chartHeight / 2}" text-anchor="middle" font-size="14" fill="#2ECC71" transform="rotate(90, ${padding + chartWidth + 28}, ${padding + chartHeight / 2})">LOTA 分数</text>
      <rect x="${width - 120}" y="${padding - 8}" width="12" height="12" fill="#E74C3C"/>
      <text x="${width - 104}" y="${padding + 2}" font-size="12" fill="#E74C3C">速度 L2</text>
      <rect x="${width - 120}" y="${padding + 12}" width="12" height="12" fill="#3498DB"/>
      <text x="${width - 104}" y="${padding + 22}" font-size="12" fill="#3498DB">加速度 L2</text>
      <rect x="${width - 120}" y="${padding + 32}" width="12" height="12" fill="#2ECC71"/>
      <text x="${width - 104}" y="${padding + 42}" font-size="12" fill="#2ECC71">LOTA 分数</text>
    </svg>
  `;
}

function renderVideoResults(items, modelName = "TRI") {
  const realCount = items.filter((item) => item.prediction === "real").length;
  const fakeCount = items.filter((item) => item.prediction === "fake").length;
  const errorCount = items.filter((item) => item.prediction === "error").length;

  let summaryText = `${modelName} 视频检测完成，共 ${items.length} 个结果，真实 ${realCount}，疑似伪造 ${fakeCount}`;
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

    const previewNode = createPreviewNode(selectedFiles[index] || { name: item.filename }, previewUrls[index], index);

    const badgeText = item.prediction === "real"
      ? "Real"
      : item.prediction === "fake"
        ? "Fake"
        : "Error";
    const verdictText = item.prediction === "real"
      ? "更接近真实视频分布。"
      : item.prediction === "fake"
        ? "更接近伪造视频分布。"
        : "模型未能完成该文件的有效处理。";

    const durationText = item.duration > 0 ? `${item.duration.toFixed(1)}秒` : "未知";
    const resolutionText = item.width && item.height ? `${item.width}×${item.height}` : "未知";
    const fpsText = item.fps > 0 ? `${item.fps.toFixed(1)}fps` : "未知";

    const trendChart = _renderTrendChart(
      item.velocity_l2,
      item.acceleration_l2,
      item.lota_scores,
      item.frame_info
    );

    const suspiciousFrame = item.suspicious_frame_b64
      ? (() => {
          const t = item.suspicious_frame_time;
          const seconds = t != null ? `(${Number(t).toFixed(2)}s)` : "";
          return `<div class="suspicious-frame"><img src="data:image/jpeg;base64,${item.suspicious_frame_b64}" alt="可疑帧"><span>可疑帧${seconds}</span></div>`;
        })()
      : "";

    card.innerHTML = `
      <div class="result-body">
        <div class="result-head">
          <p class="filename" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</p>
          <span class="badge ${item.prediction === "error" ? "pending" : item.prediction}">${badgeText}</span>
        </div>
        <p class="verdict">${verdictText}</p>
        <div class="video-meta">
          <span>时长: ${durationText}</span>
          <span>分辨率: ${resolutionText}</span>
          <span>帧率: ${fpsText}</span>
        </div>
        <div class="prob">
          <div class="prob-row"><span>真实概率</span><strong>${realPercent.toFixed(1)}%</strong></div>
          <div class="bar"><span style="width: ${realPercent}%"></span></div>
          <div class="prob-row"><span>伪造概率</span><strong>${fakePercent.toFixed(1)}%</strong></div>
          <div class="confidence">判断阈值：${(Number(item.threshold || 0.5) * 100).toFixed(2)}%</div>
        </div>
        <div class="video-analysis">
          <div class="chart-section">${trendChart}</div>
          ${suspiciousFrame}
        </div>
      </div>
    `;

    if (previewNode) {
      card.prepend(previewNode);
    }
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

"use strict";

const state = {
  session: null,
  metadata: null,
  evaluation: null,
  imageReady: false,
  objectUrl: null,
};

const $ = (id) => document.getElementById(id);
const elements = {
  status: $("status"),
  evaluationStatus: $("evaluationStatus"),
  fileInput: $("fileInput"),
  dropZone: $("dropZone"),
  previewWrap: $("previewWrap"),
  previewImage: $("previewImage"),
  inputMessage: $("inputMessage"),
  predictButton: $("predictButton"),
  clearButton: $("clearButton"),
  results: $("results"),
  emptyState: $("emptyState"),
  predictedClass: $("predictedClass"),
  confidenceScore: $("confidenceScore"),
  inferenceTime: $("inferenceTime"),
  probabilityList: $("probabilityList"),
  predictionSummary: $("predictionSummary"),
  responsibleUse: $("responsibleUse"),
  datasetWarning: $("datasetWarning"),
  selectedModelBadge: $("selectedModelBadge"),
  selectedModelName: $("selectedModelName"),
  selectionSummary: $("selectionSummary"),
  selectedMetrics: $("selectedMetrics"),
  leaderboardBody: $("leaderboardBody"),
  classMetricsBody: $("classMetricsBody"),
  robustnessList: $("robustnessList"),
  evaluationGallery: $("evaluationGallery"),
  leaderboardImage: $("leaderboardImage"),
  confusionMatrixImage: $("confusionMatrixImage"),
  gradcamImage: $("gradcamImage"),
};

const MODEL_LABELS = {
  simple_cnn: "Simple CNN",
  alexnet_style: "AlexNet-style CNN",
  mobilenetv2_frozen: "MobileNetV2 frozen",
  mobilenetv2_finetuned: "MobileNetV2 fine-tuned",
};

function setStatus(text, kind = "") {
  elements.status.textContent = text;
  elements.status.className = `status ${kind}`.trim();
}

function setEvaluationStatus(text, kind = "") {
  elements.evaluationStatus.textContent = text;
  elements.evaluationStatus.className = `status ${kind}`.trim();
}

function updatePredictButton() {
  elements.predictButton.disabled = !(state.session && state.imageReady);
}

function percentage(value, digits = 2) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "—";
}

function number(value, digits = 2) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function integer(value) {
  return Number.isFinite(Number(value)) ? Number(value).toLocaleString() : "—";
}

function modelLabel(key) {
  return MODEL_LABELS[key] || String(key || "Unknown model").replaceAll("_", " ");
}

function metricCard(label, value, note) {
  const article = document.createElement("article");
  article.className = "metric-card";
  article.innerHTML = `<span>${label}</span><strong>${value}</strong><small>${note}</small>`;
  return article;
}

async function fetchJson(path, required = true) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    if (required) throw new Error(`${path} returned HTTP ${response.status}`);
    return null;
  }
  return response.json();
}

async function loadProject() {
  try {
    state.metadata = await fetchJson("metadata.json");
    elements.responsibleUse.textContent = state.metadata.responsible_use;
    elements.datasetWarning.textContent = state.metadata.dataset.warning;

    try {
      state.evaluation = await fetchJson("evaluation_metrics.json", false);
      if (state.evaluation) {
        renderEvaluation(state.evaluation);
        setEvaluationStatus("Evaluation results loaded", "ready");
      } else {
        setEvaluationStatus("Evaluation dashboard not generated", "error");
      }
    } catch (evaluationError) {
      console.error(evaluationError);
      setEvaluationStatus("Evaluation results unavailable", "error");
    }

    if (!window.ort) throw new Error("ONNX Runtime Web did not load");
    setStatus("Loading ONNX model…");
    state.session = await ort.InferenceSession.create("model/model.onnx", {
      executionProviders: ["wasm"],
    });
    setStatus("Model ready — inference stays in your browser", "ready");
    updatePredictButton();
  } catch (error) {
    console.error(error);
    setStatus("Model could not be loaded", "error");
    elements.inputMessage.textContent = "Confirm that web/model/model.onnx is present and serve the web folder through HTTP.";
  }
}

function renderEvaluation(data) {
  const selectedKey = data.selected_model.key;
  const selected = data.leaderboard.find((row) => row.model === selectedKey) || data.selected_model.metrics;

  elements.selectedModelBadge.textContent = `Selected · ${modelLabel(selectedKey)}`;
  elements.selectedModelName.textContent = modelLabel(selectedKey);
  elements.selectionSummary.textContent = data.selected_model.summary;

  elements.selectedMetrics.replaceChildren(
    metricCard("Test accuracy", percentage(selected.accuracy), "Overall correct classifications"),
    metricCard("Macro F1", percentage(selected.macro_f1), "Primary selection metric"),
    metricCard("Balanced accuracy", percentage(selected.balanced_accuracy), "Average class recall"),
    metricCard("Top-2 accuracy", percentage(selected.top2_accuracy), "True class within top two"),
    metricCard("Macro ROC-AUC", percentage(selected.roc_auc_ovr_macro), "One-vs-rest separation"),
    metricCard("Calibration error", percentage(selected.expected_calibration_error), "Lower is better"),
    metricCard("Model size", `${number(selected.state_dict_size_mb)} MB`, "PyTorch state dictionary"),
    metricCard("GPU latency", `${number(selected.latency_mean_ms)} ms`, "Mean single-image latency"),
  );

  renderLeaderboard(data.leaderboard, selectedKey);
  renderClassMetrics(data.per_class_metrics || []);
  renderRobustness(data.robustness || []);
  renderGallery(data.visuals || {});
}

function renderLeaderboard(rows, selectedKey) {
  elements.leaderboardBody.replaceChildren();
  rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    if (row.model === selectedKey) tr.className = "selected-row";
    tr.innerHTML = `
      <td><span class="rank">${index + 1}</span><strong>${modelLabel(row.model)}</strong>${row.model === selectedKey ? '<span class="winner-tag">Deployed</span>' : ""}</td>
      <td>${percentage(row.accuracy)}</td>
      <td>${percentage(row.macro_f1)}</td>
      <td>${percentage(row.weighted_f1)}</td>
      <td>${percentage(row.top2_accuracy)}</td>
      <td>${percentage(row.roc_auc_ovr_macro)}</td>
      <td>${percentage(row.expected_calibration_error)}</td>
      <td>${number(row.state_dict_size_mb)} MB</td>
      <td>${number(row.latency_mean_ms)} ms</td>`;
    elements.leaderboardBody.appendChild(tr);
  });
}

function renderClassMetrics(rows) {
  elements.classMetricsBody.replaceChildren();
  if (!rows.length) {
    elements.classMetricsBody.innerHTML = '<tr><td colspan="5" class="table-message">Class-level metrics are unavailable.</td></tr>';
    return;
  }
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><strong>${row.class_name}</strong></td><td>${percentage(row.precision)}</td><td>${percentage(row.recall)}</td><td>${percentage(row.f1_score)}</td><td>${integer(row.support)}</td>`;
    elements.classMetricsBody.appendChild(tr);
  });
}

function renderRobustness(rows) {
  elements.robustnessList.replaceChildren();
  if (!rows.length) {
    elements.robustnessList.innerHTML = '<p class="table-message">Robustness metrics are unavailable.</p>';
    return;
  }
  const clean = rows.find((row) => row.condition === "clean")?.macro_f1 || rows[0].macro_f1;
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "robustness-item";
    const relativeWidth = clean > 0 ? Math.max(0, Math.min(100, (row.macro_f1 / clean) * 100)) : 0;
    const dropText = row.drop_from_clean > 0.00005 ? `−${percentage(row.drop_from_clean)}` : "Baseline";
    item.innerHTML = `
      <div class="robustness-label"><span>${row.label}</span><strong>${percentage(row.macro_f1)}</strong></div>
      <div class="robustness-track"><span style="width:${relativeWidth.toFixed(1)}%"></span></div>
      <small>${dropText}</small>`;
    elements.robustnessList.appendChild(item);
  });
}

function renderGallery(visuals) {
  const required = ["leaderboard", "confusion_matrix", "gradcam"];
  if (!required.every((key) => visuals[key])) return;
  elements.leaderboardImage.src = visuals.leaderboard;
  elements.confusionMatrixImage.src = visuals.confusion_matrix;
  elements.gradcamImage.src = visuals.gradcam;
  elements.evaluationGallery.hidden = false;
}

function showFile(file) {
  const allowed = new Set(["image/png", "image/jpeg", "image/webp", "image/bmp"]);
  if (!file || !allowed.has(file.type)) {
    elements.inputMessage.textContent = "Use PNG, JPEG, WebP, or BMP.";
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    elements.inputMessage.textContent = "The image must be 10 MB or smaller.";
    return;
  }
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  elements.previewImage.onload = () => {
    state.imageReady = true;
    elements.previewWrap.hidden = false;
    elements.dropZone.hidden = true;
    elements.inputMessage.textContent = `${file.name} is ready.`;
    updatePredictButton();
  };
  elements.previewImage.src = state.objectUrl;
}

function clearImage() {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
  state.imageReady = false;
  elements.fileInput.value = "";
  elements.previewImage.removeAttribute("src");
  elements.previewWrap.hidden = true;
  elements.dropZone.hidden = false;
  elements.results.hidden = true;
  elements.emptyState.hidden = false;
  elements.inputMessage.textContent = "Load an image to begin.";
  updatePredictButton();
}

function preprocess(image) {
  const size = state.metadata.preprocessing.browser_size[0];
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, size, size);
  const rgba = context.getImageData(0, 0, size, size).data;
  const mean = state.metadata.preprocessing.mean;
  const std = state.metadata.preprocessing.std;
  const data = new Float32Array(3 * size * size);
  for (let index = 0; index < size * size; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      data[channel * size * size + index] = (rgba[index * 4 + channel] / 255 - mean[channel]) / std[channel];
    }
  }
  return new ort.Tensor("float32", data, [1, 3, size, size]);
}

function softmax(values) {
  const maximum = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

async function predict() {
  if (!state.session || !state.imageReady) return;
  elements.predictButton.disabled = true;
  elements.predictButton.textContent = "Running…";
  try {
    const input = preprocess(elements.previewImage);
    const start = performance.now();
    const output = await state.session.run({ input });
    const elapsed = performance.now() - start;
    const values = softmax(Array.from(output.logits.data));
    const ranked = state.metadata.dataset.class_names
      .map((label, index) => ({ label, score: values[index] }))
      .sort((first, second) => second.score - first.score);
    const best = ranked[0];

    elements.predictedClass.textContent = best.label;
    elements.confidenceScore.textContent = percentage(best.score, 1);
    elements.inferenceTime.textContent = `${elapsed.toFixed(1)} ms`;
    elements.probabilityList.replaceChildren();
    ranked.forEach(({ label, score }) => {
      const row = document.createElement("div");
      row.className = "probability-row";
      row.innerHTML = `<span>${label}</span><span class="probability-track"><span class="probability-fill" style="width:${score * 100}%"></span></span><span class="probability-value">${percentage(score, 1)}</span>`;
      elements.probabilityList.appendChild(row);
    });
    elements.predictionSummary.textContent = `The selected model assigns this image to “${best.label}”. Confidence is a model output, not guaranteed truth.`;
    elements.emptyState.hidden = true;
    elements.results.hidden = false;
  } catch (error) {
    console.error(error);
    elements.inputMessage.textContent = `Inference failed: ${error.message}`;
  } finally {
    elements.predictButton.textContent = "Run browser inference";
    updatePredictButton();
  }
}

elements.fileInput.addEventListener("change", (event) => showFile(event.target.files[0]));
elements.clearButton.addEventListener("click", clearImage);
elements.predictButton.addEventListener("click", predict);
["dragenter", "dragover"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  elements.dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("dragging");
}));
elements.dropZone.addEventListener("drop", (event) => showFile(event.dataTransfer.files[0]));
window.addEventListener("DOMContentLoaded", loadProject);

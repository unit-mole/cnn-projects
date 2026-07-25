"use strict";

const ASSET_VERSION = "20260725-input-shape-fix";

const state = {
  model: null,
  metadata: null,
  imageElement: null,
  objectUrl: null,
};

const elements = {};

window.addEventListener("DOMContentLoaded", async () => {
  cacheElements();
  bindEvents();
  renderSamples();
  await initializeModel();
});

function cacheElements() {
  [
    "model-status", "artifact-warning", "image-upload", "image-preview",
    "preview-placeholder", "predict-button", "clear-button", "sample-grid",
    "empty-result", "result-content", "predicted-class", "confidence",
    "probability-list", "prediction-summary", "latency", "model-details"
  ].forEach((id) => { elements[id] = document.getElementById(id); });
}

function bindEvents() {
  elements["image-upload"].addEventListener("change", (event) => {
    const [file] = event.target.files;
    if (file) loadFile(file);
  });
  elements["predict-button"].addEventListener("click", predict);
  elements["clear-button"].addEventListener("click", clearSelection);
}

async function initializeModel() {
  try {
    if (typeof tf === "undefined") throw new Error("TensorFlow.js did not load.");
    state.metadata = await fetchJson(`./metadata.json?v=${ASSET_VERSION}`);
    validateMetadata(state.metadata);
    renderModelDetails(state.metadata);

    if (state.metadata.artifact_status !== "trained") {
      elements["artifact-warning"].hidden = false;
      elements["artifact-warning"].textContent = state.metadata.artifact_warning || "This is not a trained production artifact.";
    }

    state.model = await loadBrowserModel();
    const warmup = tf.zeros([
      1,
      state.metadata.input_height,
      state.metadata.input_width,
      state.metadata.channels,
    ]);
    const output = state.model.predict(warmup);
    tf.dispose([warmup, output]);

    const statusText = state.metadata.artifact_status === "trained"
      ? "Model ready"
      : "Demo model ready";
    setStatus(statusText, "ready");
    updatePredictButton();
  } catch (error) {
    console.error(error);
    setStatus("Model failed", "error");
    elements["artifact-warning"].hidden = false;
    elements["artifact-warning"].textContent = `The model could not be loaded: ${error.message}`;
  }
}

async function loadBrowserModel() {
  try {
    return await tf.loadLayersModel(`./tfjs_model/model.json?v=${ASSET_VERSION}`);
  } catch (loadError) {
    if (state.metadata.artifact_status === "trained") throw loadError;

    console.warn(
      "The bundled smoke-test model manifest could not be loaded. " +
      "Using the equivalent TensorFlow.js browser fallback.",
      loadError
    );
    return createBrowserSmokeTestModel();
  }
}

function createBrowserSmokeTestModel() {
  const model = tf.sequential({ name: "browser_smoke_test_global_rgb" });
  model.add(tf.layers.globalAveragePooling2d({
    inputShape: [
      state.metadata.input_height,
      state.metadata.input_width,
      state.metadata.channels,
    ],
    dataFormat: "channelsLast",
    name: "global_average_pooling",
  }));
  model.add(tf.layers.dense({
    units: state.metadata.num_classes,
    activation: "softmax",
    useBias: true,
    name: "classifier",
  }));

  const kernelValues = [
    0.8, -0.3, 0.1, 0.6, -0.2, 0.2, -0.4, 0.3, -0.1, 0.5,
    -0.2, 0.7, 0.4, -0.3, 0.8, -0.1, 0.6, 0.2, 0.4, -0.5,
    0.1, 0.2, 0.7, 0.3, 0.1, 0.8, 0.5, -0.2, 0.6, 0.4,
  ];
  const biasValues = [
    0.0, 0.02, -0.01, 0.01, -0.02,
    0.0, 0.015, -0.015, 0.005, -0.005,
  ];

  const kernel = tf.tensor2d(kernelValues, [state.metadata.channels, state.metadata.num_classes]);
  const bias = tf.tensor1d(biasValues);
  model.getLayer("classifier").setWeights([kernel, bias]);
  tf.dispose([kernel, bias]);
  return model;
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}`);
  return response.json();
}

function validateMetadata(metadata) {
  const required = ["class_names", "input_height", "input_width", "channels", "normalization"];
  const missing = required.filter((key) => !(key in metadata));
  if (missing.length) throw new Error(`Metadata missing: ${missing.join(", ")}`);
  if (metadata.class_names.length < 2) throw new Error("At least two class names are required.");
}

function setStatus(message, type) {
  elements["model-status"].textContent = message;
  elements["model-status"].className = `status status-${type}`;
}

function renderModelDetails(metadata) {
  const rows = [
    ["Dataset", metadata.dataset],
    ["Model family", metadata.model_family],
    ["Input", `${metadata.input_width} × ${metadata.input_height} × ${metadata.channels}`],
    ["Normalization", metadata.normalization],
    ["Classes", String(metadata.num_classes)],
    ["Runtime", "TensorFlow.js in browser"],
    ["Primary hosting", "GitHub Pages"],
    ["Artifact status", metadata.artifact_status],
  ];
  elements["model-details"].innerHTML = rows.map(([label, value]) =>
    `<div class="detail"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`
  ).join("");
}

function renderSamples() {
  const samples = [
    ["warm gradient", "./sample_images/sample_warm.png"],
    ["cool geometry", "./sample_images/sample_cool.png"],
    ["green pattern", "./sample_images/sample_green.png"],
  ];
  elements["sample-grid"].innerHTML = samples.map(([name, source]) =>
    `<button class="sample-card" type="button" data-source="${source}" aria-label="Load ${name}">
      <img src="${source}" alt="Synthetic ${name} sample"><span>${name}</span>
    </button>`
  ).join("");
  elements["sample-grid"].querySelectorAll(".sample-card").forEach((button) => {
    button.addEventListener("click", () => loadImageSource(button.dataset.source));
  });
}

function loadFile(file) {
  const allowed = ["image/png", "image/jpeg", "image/webp", "image/bmp"];
  if (!allowed.includes(file.type)) {
    alert("Unsupported image type. Choose PNG, JPEG, WebP, or BMP.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    alert("Image is larger than 10 MB. Choose a smaller file.");
    return;
  }
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  loadImageSource(state.objectUrl);
}

function loadImageSource(source) {
  const image = new Image();
  image.onload = () => {
    state.imageElement = image;
    elements["image-preview"].src = source;
    elements["image-preview"].hidden = false;
    elements["preview-placeholder"].hidden = true;
    clearResult();
    updatePredictButton();
  };
  image.onerror = () => alert("The selected image could not be decoded.");
  image.src = source;
}

function updatePredictButton() {
  elements["predict-button"].disabled = !(state.model && state.metadata && state.imageElement);
}

async function predict() {
  if (!state.model || !state.metadata || !state.imageElement) return;
  elements["predict-button"].disabled = true;
  elements["predict-button"].textContent = "Running…";
  const started = performance.now();

  try {
    const probabilities = tf.tidy(() => {
      let tensor = tf.browser.fromPixels(state.imageElement, state.metadata.channels);
      tensor = tf.image.resizeBilinear(tensor, [state.metadata.input_height, state.metadata.input_width], true);
      tensor = tensor.toFloat();
      if (state.metadata.normalization === "zero_one") tensor = tensor.div(255);
      if (state.metadata.normalization === "minus_one_one") tensor = tensor.div(127.5).sub(1);
      const batch = tensor.expandDims(0);
      const prediction = state.model.predict(batch);
      return prediction.squeeze().clone();
    });
    const values = Array.from(await probabilities.data());
    probabilities.dispose();
    renderPrediction(values, performance.now() - started);
  } catch (error) {
    console.error(error);
    alert(`Prediction failed: ${error.message}`);
  } finally {
    elements["predict-button"].textContent = "Run classification";
    updatePredictButton();
  }
}

function renderPrediction(values, latencyMs) {
  if (values.length !== state.metadata.class_names.length) {
    throw new Error(`Model returned ${values.length} scores for ${state.metadata.class_names.length} classes.`);
  }
  const ranked = values
    .map((confidence, index) => ({ className: state.metadata.class_names[index], confidence }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, Math.min(5, values.length));

  const best = ranked[0];
  elements["empty-result"].hidden = true;
  elements["result-content"].hidden = false;
  elements["predicted-class"].textContent = best.className;
  elements["confidence"].textContent = `${formatPercent(best.confidence)} confidence`;
  elements["latency"].textContent = `${latencyMs.toFixed(0)} ms`;
  elements["probability-list"].innerHTML = ranked.map((row) => `
    <div class="probability-row">
      <span class="probability-label">${escapeHtml(row.className)}</span>
      <span class="probability-track"><span class="probability-fill" style="width:${Math.max(0, Math.min(100, row.confidence * 100))}%"></span></span>
      <span class="probability-value">${formatPercent(row.confidence)}</span>
    </div>
  `).join("");

  const statusNote = state.metadata.artifact_status === "trained"
    ? "The result was generated by the exported trained model."
    : "The bundled smoke-test model only verifies browser wiring; this class is not a meaningful CIFAR-10 result.";
  elements["prediction-summary"].textContent = `${statusNote} The highest score is ${best.className} at ${formatPercent(best.confidence)}. Interpret all outputs cautiously.`;
}

function clearSelection() {
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
  }
  state.imageElement = null;
  elements["image-upload"].value = "";
  elements["image-preview"].removeAttribute("src");
  elements["image-preview"].hidden = true;
  elements["preview-placeholder"].hidden = false;
  clearResult();
  updatePredictButton();
}

function clearResult() {
  elements["empty-result"].hidden = false;
  elements["result-content"].hidden = true;
  elements["latency"].textContent = "";
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  }[character]));
}

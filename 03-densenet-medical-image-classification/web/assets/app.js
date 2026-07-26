"use strict";

const MODEL_URL = "model/model.json";
const METADATA_URL = "assets/model_metadata.json";
const CLASS_NAMES = ["normal_like", "pneumonia_like"];
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

let model = null;
let selectedImage = null;
let metadata = null;
let lastPrediction = null;

const byId = (id) => document.getElementById(id);
const ui = {};

function setModelState(state, message) {
  ui.modelStatus.textContent = message;
  ui.modelStatusDot.className = `status-dot ${state}`.trim();
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function setInputMessage(message, isError = false) {
  ui.inputMessage.textContent = message;
  ui.inputMessage.classList.toggle("error", isError);
}

async function selectBestBackend() {
  const preferred = ["webgl", "cpu"];
  for (const backend of preferred) {
    try {
      const ok = await tf.setBackend(backend);
      if (ok) {
        await tf.ready();
        return tf.getBackend();
      }
    } catch (error) {
      console.warn(`TensorFlow.js backend ${backend} unavailable`, error);
    }
  }
  await tf.ready();
  return tf.getBackend();
}

async function loadModel() {
  const started = performance.now();
  setModelState("loading", "Loading DenseNet121 browser model…");
  const backend = await selectBestBackend();
  ui.backendValue.textContent = backend;

  try {
    const metadataResponse = await fetch(METADATA_URL, { cache: "no-store" });
    if (metadataResponse.ok) metadata = await metadataResponse.json();
  } catch (error) {
    console.warn("Metadata could not be loaded", error);
  }

  try {
    model = await tf.loadLayersModel(MODEL_URL);
    const warmup = tf.zeros([1, 96, 96, 3]);
    const warmupOutput = model.predict(warmup);
    await warmupOutput.data();
    warmup.dispose();
    warmupOutput.dispose();

    const elapsed = performance.now() - started;
    ui.loadTimeValue.textContent = `${(elapsed / 1000).toFixed(1)} s`;
    setModelState("", "Model ready for browser inference");
    ui.predictButton.disabled = !selectedImage;
    setInputMessage(selectedImage ? "Image ready. Run the prediction." : "Choose an image or sample to begin.");
  } catch (error) {
    console.error(error);
    setModelState("error", "Model failed to load");
    ui.loadTimeValue.textContent = "Failed";
    ui.predictButton.disabled = true;
    setInputMessage("The TensorFlow.js model files are missing or could not be loaded. Run the GitHub Actions deployment workflow or the local conversion script, then serve the web folder over HTTP.", true);
  }
}

function drawContainedImage(image, canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);

  const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
  const drawWidth = Math.max(1, image.naturalWidth * scale);
  const drawHeight = Math.max(1, image.naturalHeight * scale);
  const x = (width - drawWidth) / 2;
  const y = (height - drawHeight) / 2;
  ctx.drawImage(image, x, y, drawWidth, drawHeight);
}

function preprocessImage(image) {
  drawContainedImage(image, ui.preprocessCanvas);
  return tf.tidy(() => {
    const pixels = tf.browser.fromPixels(ui.preprocessCanvas, 3).toFloat().div(255);
    const resized = tf.image.resizeBilinear(pixels, [96, 96], false);
    const mean = tf.tensor1d(IMAGENET_MEAN);
    const std = tf.tensor1d(IMAGENET_STD);
    return resized.sub(mean).div(std).expandDims(0);
  });
}

async function runPrediction() {
  if (!model || !selectedImage) return;
  ui.predictButton.disabled = true;
  ui.predictButton.textContent = "Running DenseNet…";
  setInputMessage("Inference is running locally in your browser.");

  const inputTensor = preprocessImage(selectedImage);
  const started = performance.now();
  try {
    const outputTensor = model.predict(inputTensor);
    const probabilities = Array.from(await outputTensor.data());
    const elapsed = performance.now() - started;
    outputTensor.dispose();

    const predictedIndex = probabilities[1] > probabilities[0] ? 1 : 0;
    const predictedClass = CLASS_NAMES[predictedIndex];
    const confidence = probabilities[predictedIndex];
    lastPrediction = {
      project: "03-densenet-medical-image-classification",
      model: "DenseNet121 synthetic proxy",
      predicted_class: predictedClass,
      confidence,
      probabilities: {
        normal_like: probabilities[0],
        pneumonia_like: probabilities[1]
      },
      runtime: {
        backend: tf.getBackend(),
        inference_ms: Number(elapsed.toFixed(2)),
        input_shape: [96, 96, 3]
      },
      disclaimer: metadata?.medical_disclaimer || "Educational portfolio demonstration only; not medical advice.",
      generated_at: new Date().toISOString()
    };

    renderResult(lastPrediction);
    ui.inferenceTimeValue.textContent = `${elapsed.toFixed(0)} ms`;
    setInputMessage("Prediction complete. The uploaded image was processed only in this browser.");
  } catch (error) {
    console.error(error);
    setInputMessage(`Prediction failed: ${error.message}`, true);
  } finally {
    inputTensor.dispose();
    ui.predictButton.disabled = false;
    ui.predictButton.textContent = "Run DenseNet prediction";
  }
}

function renderResult(result) {
  ui.emptyResult.hidden = true;
  ui.resultContent.hidden = false;
  ui.downloadButton.disabled = false;
  ui.predictedClass.textContent = result.predicted_class;
  ui.confidenceValue.textContent = formatPercent(result.confidence);
  ui.normalProbability.textContent = formatPercent(result.probabilities.normal_like);
  ui.pneumoniaProbability.textContent = formatPercent(result.probabilities.pneumonia_like);
  ui.normalBar.style.width = formatPercent(result.probabilities.normal_like);
  ui.pneumoniaBar.style.width = formatPercent(result.probabilities.pneumonia_like);
  ui.confidenceRing.style.background = `conic-gradient(var(--accent-2) ${result.confidence * 360}deg, rgba(255,255,255,.08) 0deg)`;
  ui.resultBackend.textContent = result.runtime.backend;
  ui.resultTime.textContent = `${result.runtime.inference_ms.toFixed(0)} ms`;
  ui.interpretationText.textContent = `The model assigned the highest synthetic-proxy probability to “${result.predicted_class}” with ${formatPercent(result.confidence)} confidence. This is an engineering demonstration based on proxy labels and must not be interpreted as a medical finding.`;
}

function loadImageFromUrl(url, label = "sample") {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      selectedImage = image;
      ui.previewImage.src = url;
      ui.previewImage.hidden = false;
      ui.dropPrompt.hidden = true;
      ui.predictButton.disabled = !model;
      setInputMessage(`${label} loaded${model ? ". Ready for prediction." : ". Waiting for the model."}`);
      resolve(image);
    };
    image.onerror = () => reject(new Error("The selected image could not be loaded."));
    image.src = url;
  });
}

function handleFile(file) {
  if (!file) return;
  const allowed = ["image/png", "image/jpeg", "image/webp"];
  if (!allowed.includes(file.type)) {
    setInputMessage("Please choose a PNG, JPEG, or WebP image.", true);
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    setInputMessage("Please choose an image smaller than 10 MB.", true);
    return;
  }
  const url = URL.createObjectURL(file);
  loadImageFromUrl(url, file.name).catch((error) => setInputMessage(error.message, true));
}

function clearSelection() {
  selectedImage = null;
  lastPrediction = null;
  ui.fileInput.value = "";
  ui.previewImage.removeAttribute("src");
  ui.previewImage.hidden = true;
  ui.dropPrompt.hidden = false;
  ui.predictButton.disabled = true;
  ui.emptyResult.hidden = false;
  ui.resultContent.hidden = true;
  ui.downloadButton.disabled = true;
  document.querySelectorAll(".sample-button").forEach((button) => button.classList.remove("active"));
  setInputMessage(model ? "Choose an image or sample to begin." : "Waiting for the model to load.");
}

function downloadPrediction() {
  if (!lastPrediction) return;
  const blob = new Blob([JSON.stringify(lastPrediction, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "densenet-browser-prediction.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

function bindEvents() {
  ui.fileInput.addEventListener("change", (event) => handleFile(event.target.files?.[0]));
  ui.predictButton.addEventListener("click", runPrediction);
  ui.clearButton.addEventListener("click", clearSelection);
  ui.downloadButton.addEventListener("click", downloadPrediction);

  ["dragenter", "dragover"].forEach((name) => ui.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    ui.dropZone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((name) => ui.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    ui.dropZone.classList.remove("dragging");
  }));
  ui.dropZone.addEventListener("drop", (event) => handleFile(event.dataTransfer.files?.[0]));

  document.querySelectorAll(".sample-button").forEach((button) => {
    button.addEventListener("click", async () => {
      document.querySelectorAll(".sample-button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      try {
        await loadImageFromUrl(button.dataset.src, button.textContent.trim());
      } catch (error) {
        setInputMessage(error.message, true);
      }
    });
  });
}

function cacheUi() {
  [
    "modelStatusDot", "modelStatus", "backendValue", "loadTimeValue", "inferenceTimeValue",
    "fileInput", "dropZone", "dropPrompt", "previewImage", "predictButton", "clearButton",
    "downloadButton", "inputMessage", "emptyResult", "resultContent", "predictedClass",
    "confidenceValue", "confidenceRing", "normalProbability", "pneumoniaProbability",
    "normalBar", "pneumoniaBar", "interpretationText", "resultBackend", "resultTime",
    "preprocessCanvas"
  ].forEach((id) => { ui[id] = byId(id); });
}

document.addEventListener("DOMContentLoaded", async () => {
  cacheUi();
  bindEvents();
  if (typeof tf === "undefined") {
    setModelState("error", "TensorFlow.js failed to load");
    setInputMessage("The TensorFlow.js CDN script could not be loaded. Check your internet connection and refresh the page.", true);
    return;
  }
  await loadModel();
});

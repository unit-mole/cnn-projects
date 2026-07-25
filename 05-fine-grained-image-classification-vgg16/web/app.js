const MODEL_URL = "./tfjs_model/model.json";
const METADATA_URL = "./metadata.json";
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/bmp"]);

const elements = {
  modelStatus: document.querySelector("#model-status"),
  imageInput: document.querySelector("#image-input"),
  dropZone: document.querySelector("#drop-zone"),
  previewCard: document.querySelector("#preview-card"),
  imagePreview: document.querySelector("#image-preview"),
  fileName: document.querySelector("#file-name"),
  fileDetails: document.querySelector("#file-details"),
  predictButton: document.querySelector("#predict-button"),
  errorMessage: document.querySelector("#error-message"),
  emptyResults: document.querySelector("#empty-results"),
  predictionResults: document.querySelector("#prediction-results"),
  predictedClass: document.querySelector("#predicted-class"),
  confidenceRing: document.querySelector("#confidence-ring"),
  confidenceValue: document.querySelector("#confidence-value"),
  probabilityList: document.querySelector("#probability-list"),
  uncertaintyWarning: document.querySelector("#uncertainty-warning"),
  predictionSummary: document.querySelector("#prediction-summary"),
  inferenceTime: document.querySelector("#inference-time"),
};

let model = null;
let metadata = null;
let selectedImageUrl = null;
let selectedFile = null;

function setStatus(message, kind) {
  elements.modelStatus.textContent = message;
  elements.modelStatus.className = `status status-${kind}`;
}

function showError(message) {
  elements.errorMessage.textContent = message;
  elements.errorMessage.classList.remove("is-hidden");
}

function clearError() {
  elements.errorMessage.textContent = "";
  elements.errorMessage.classList.add("is-hidden");
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "sample image";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function revokeSelectedUrl() {
  if (selectedImageUrl?.startsWith("blob:")) URL.revokeObjectURL(selectedImageUrl);
}

async function loadImage(url) {
  const image = new Image();
  image.decoding = "async";
  image.crossOrigin = "anonymous";
  image.src = url;
  await image.decode();
  return image;
}

async function chooseFile(file) {
  clearError();
  if (!file) return;
  if (!ACCEPTED_TYPES.has(file.type)) {
    showError("Unsupported image type. Choose PNG, JPEG, WebP, or BMP.");
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    showError("The selected image is larger than 10 MB.");
    return;
  }

  revokeSelectedUrl();
  selectedFile = file;
  selectedImageUrl = URL.createObjectURL(file);
  try {
    const image = await loadImage(selectedImageUrl);
    elements.imagePreview.src = selectedImageUrl;
    elements.fileName.textContent = file.name;
    elements.fileDetails.textContent = `${image.naturalWidth}×${image.naturalHeight} · ${formatBytes(file.size)}`;
    elements.previewCard.classList.remove("is-hidden");
    elements.predictButton.disabled = !model;
  } catch (error) {
    showError("The browser could not decode this image. It may be corrupt.");
    elements.predictButton.disabled = true;
  }
}

async function chooseSample(url, label) {
  clearError();
  revokeSelectedUrl();
  selectedFile = null;
  selectedImageUrl = url;
  try {
    const image = await loadImage(url);
    elements.imagePreview.src = url;
    elements.fileName.textContent = label;
    elements.fileDetails.textContent = `${image.naturalWidth}×${image.naturalHeight} · packaged sample`;
    elements.previewCard.classList.remove("is-hidden");
    elements.predictButton.disabled = !model;
  } catch (error) {
    showError("The packaged sample image could not be loaded.");
  }
}

function preprocessImage(image) {
  return tf.tidy(() => {
    const pixels = tf.browser.fromPixels(image, 3).toFloat();
    const sourceSized = tf.image.resizeBilinear(pixels, metadata.input.source_size, true);
    const modelSized = tf.image.resizeBilinear(sourceSized, metadata.input.model_size, true);
    const bgr = tf.reverse(modelSized, [2]);
    const means = tf.tensor1d(metadata.preprocessing.mean_subtraction, "float32");
    return bgr.sub(means).expandDims(0);
  });
}

function rankPredictions(probabilities) {
  return Array.from(probabilities)
    .map((probability, index) => ({
      className: metadata.classes[index],
      probability,
      index,
    }))
    .sort((left, right) => right.probability - left.probability)
    .slice(0, Math.min(metadata.top_k ?? 5, metadata.classes.length));
}

function renderResults(topPredictions, elapsedMilliseconds) {
  const winner = topPredictions[0];
  const runnerUp = topPredictions[1];
  const confidencePercent = winner.probability * 100;
  const probabilityGap = runnerUp ? winner.probability - runnerUp.probability : 1;
  const isClose = runnerUp && probabilityGap < metadata.similar_class_warning_threshold;

  elements.predictedClass.textContent = winner.className;
  elements.confidenceValue.textContent = `${confidencePercent.toFixed(1)}%`;
  elements.confidenceRing.style.setProperty("--confidence", `${confidencePercent}%`);
  elements.inferenceTime.textContent = `${elapsedMilliseconds.toFixed(0)} ms inference`;

  elements.probabilityList.replaceChildren();
  for (const prediction of topPredictions) {
    const row = document.createElement("div");
    row.className = "probability-row";
    row.innerHTML = `
      <span class="probability-label"></span>
      <span class="probability-track"><span class="probability-fill"></span></span>
      <span class="probability-value"></span>`;
    row.querySelector(".probability-label").textContent = prediction.className;
    row.querySelector(".probability-fill").style.width = `${prediction.probability * 100}%`;
    row.querySelector(".probability-value").textContent = `${(prediction.probability * 100).toFixed(1)}%`;
    elements.probabilityList.append(row);
  }

  if (isClose) {
    elements.uncertaintyWarning.textContent =
      "The top two predictions are close. This may indicate visual similarity between classes or model uncertainty.";
    elements.uncertaintyWarning.classList.remove("is-hidden");
  } else {
    elements.uncertaintyWarning.classList.add("is-hidden");
  }

  elements.predictionSummary.textContent =
    `The VGG16 model predicts ${winner.className} with ${confidencePercent.toFixed(1)}% confidence. ` +
    (isClose
      ? "Because the classes have similar visual evidence, review both probabilities rather than treating the first label as guaranteed truth."
      : "The first prediction is separated from the second by more than the configured 15-point uncertainty threshold.");

  elements.emptyResults.classList.add("is-hidden");
  elements.predictionResults.classList.remove("is-hidden");
}

async function runPrediction() {
  clearError();
  if (!model || !metadata || !selectedImageUrl) return;
  elements.predictButton.disabled = true;
  elements.predictButton.textContent = "Running inference…";

  let inputTensor;
  let outputTensor;
  try {
    const image = await loadImage(selectedImageUrl);
    inputTensor = preprocessImage(image);
    const started = performance.now();
    outputTensor = model.predict(inputTensor);
    const probabilities = await outputTensor.data();
    const elapsed = performance.now() - started;
    renderResults(rankPredictions(probabilities), elapsed);
  } catch (error) {
    console.error(error);
    showError("Inference failed. Refresh the page and verify that all model shards are deployed.");
  } finally {
    inputTensor?.dispose();
    if (Array.isArray(outputTensor)) outputTensor.forEach((tensor) => tensor.dispose());
    else outputTensor?.dispose();
    elements.predictButton.disabled = !model || !selectedImageUrl;
    elements.predictButton.textContent = "Run browser inference";
  }
}

async function initialize() {
  try {
    if (!globalThis.tf) throw new Error("TensorFlow.js did not load from the CDN.");
    [metadata, model] = await Promise.all([
      fetch(METADATA_URL).then((response) => {
        if (!response.ok) throw new Error(`Metadata request failed: ${response.status}`);
        return response.json();
      }),
      tf.loadLayersModel(MODEL_URL, {
        onProgress: (fraction) => setStatus(`Loading model ${Math.round(fraction * 100)}%`, "loading"),
      }),
    ]);
    setStatus("Model ready", "ready");
    elements.predictButton.disabled = !selectedImageUrl;
    console.info(`TensorFlow.js backend: ${tf.getBackend()}`);
  } catch (error) {
    console.error(error);
    setStatus("Model unavailable", "error");
    showError("The model could not load. Serve this folder through HTTP and confirm every .bin shard is present.");
  }
}

elements.imageInput.addEventListener("change", (event) => chooseFile(event.target.files?.[0]));
elements.predictButton.addEventListener("click", runPrediction);
elements.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.imageInput.click();
  }
});
for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("is-dragging");
  });
}
elements.dropZone.addEventListener("drop", (event) => chooseFile(event.dataTransfer?.files?.[0]));
document.querySelectorAll(".sample-button").forEach((button) => {
  button.addEventListener("click", () => chooseSample(button.dataset.sample, button.textContent.trim()));
});
window.addEventListener("beforeunload", revokeSelectedUrl);

initialize();

"use strict";

const MODEL_SIZE = 64;
const DISPLAY_SIZE = 320;
const MODEL_DESCRIPTOR_URL = "./tfjs_model/model.json";
const WEIGHTS_MANIFEST_URL = "./tfjs_model/weights_manifest.json";

const state = {
  model: null,
  ready: false,
  image: null,
  groundTruth: null,
  probabilities: null,
  modelInputPixels: null,
};

const elements = {};

function byId(id) {
  return document.getElementById(id);
}

function cacheElements() {
  [
    "model-dot", "model-status", "backend-name", "image-input", "mask-input",
    "drop-zone", "threshold", "threshold-value", "predict-button", "download-button",
    "action-message", "original-canvas", "probability-canvas", "mask-canvas",
    "overlay-canvas", "inference-time", "region-share", "mean-probability",
    "dice-score", "iou-score", "sample-grid"
  ].forEach((id) => { elements[id] = byId(id); });
}

function setStatus(kind, message) {
  elements["model-dot"].className = `status-dot ${kind}`;
  elements["model-status"].textContent = message;
}

function setMessage(message, isError = false) {
  elements["action-message"].textContent = message;
  elements["action-message"].style.color = isError ? "#ff9c9c" : "";
}

function convLayer(x, filters, kernelSize, name, activation = "relu") {
  return tf.layers.conv2d({
    filters,
    kernelSize,
    padding: "same",
    activation,
    useBias: true,
    kernelInitializer: "zeros",
    biasInitializer: "zeros",
    name,
  }).apply(x);
}

function buildCompactUnet() {
  const input = tf.input({ shape: [MODEL_SIZE, MODEL_SIZE, 1], name: "input_layer" });

  const c1a = convLayer(input, 32, 3, "conv2d");
  const c1b = convLayer(c1a, 32, 3, "conv2d_1");
  const p1 = tf.layers.maxPooling2d({ poolSize: [2, 2], name: "max_pooling2d" }).apply(c1b);

  const c2a = convLayer(p1, 64, 3, "conv2d_2");
  const c2b = convLayer(c2a, 64, 3, "conv2d_3");
  const p2 = tf.layers.maxPooling2d({ poolSize: [2, 2], name: "max_pooling2d_1" }).apply(c2b);

  const b1 = convLayer(p2, 128, 3, "conv2d_4");
  const b2 = convLayer(b1, 128, 3, "conv2d_5");

  const u1 = tf.layers.upSampling2d({ size: [2, 2], name: "up_sampling2d" }).apply(b2);
  const m1 = tf.layers.concatenate({ axis: -1, name: "concatenate" }).apply([u1, c2b]);
  const d1a = convLayer(m1, 64, 3, "conv2d_6");
  const d1b = convLayer(d1a, 64, 3, "conv2d_7");

  const u2 = tf.layers.upSampling2d({ size: [2, 2], name: "up_sampling2d_1" }).apply(d1b);
  const m2 = tf.layers.concatenate({ axis: -1, name: "concatenate_1" }).apply([u2, c1b]);
  const d2a = convLayer(m2, 32, 3, "conv2d_8");
  const d2b = convLayer(d2a, 32, 3, "conv2d_9");

  const output = convLayer(d2b, 1, 1, "conv2d_10", "sigmoid");
  return tf.model({ inputs: input, outputs: output, name: "compact_medical_unet" });
}

function isLittleEndian() {
  const buffer = new ArrayBuffer(4);
  new Uint32Array(buffer)[0] = 0x01020304;
  return new Uint8Array(buffer)[0] === 0x04;
}

function readFloat32(buffer, byteOffset, valueCount) {
  if (isLittleEndian() && byteOffset % 4 === 0) {
    return new Float32Array(buffer, byteOffset, valueCount);
  }
  const view = new DataView(buffer, byteOffset, valueCount * 4);
  const output = new Float32Array(valueCount);
  for (let i = 0; i < valueCount; i += 1) {
    output[i] = view.getFloat32(i * 4, true);
  }
  return output;
}

async function loadBrowserWeights(model) {
  const [descriptorResponse, manifestResponse] = await Promise.all([
    fetch(MODEL_DESCRIPTOR_URL, { cache: "no-cache" }),
    fetch(WEIGHTS_MANIFEST_URL, { cache: "no-cache" }),
  ]);
  if (!descriptorResponse.ok) throw new Error(`model.json returned HTTP ${descriptorResponse.status}`);
  if (!manifestResponse.ok) throw new Error(`weights_manifest.json returned HTTP ${manifestResponse.status}`);

  const descriptor = await descriptorResponse.json();
  const manifest = await manifestResponse.json();
  const weightsResponse = await fetch(`./tfjs_model/${manifest.weight_file}`);
  if (!weightsResponse.ok) throw new Error(`weights.bin returned HTTP ${weightsResponse.status}`);
  const buffer = await weightsResponse.arrayBuffer();
  if (buffer.byteLength !== manifest.weight_bytes) {
    throw new Error(`Weight bundle size mismatch: expected ${manifest.weight_bytes}, received ${buffer.byteLength}`);
  }

  const grouped = new Map();
  for (const item of manifest.weights) {
    const values = readFloat32(buffer, item.byte_offset, item.value_count);
    const tensor = tf.tensor(values, item.shape, "float32");
    if (!grouped.has(item.layer)) grouped.set(item.layer, {});
    grouped.get(item.layer)[item.role] = tensor;
  }

  try {
    for (const [layerName, pair] of grouped.entries()) {
      if (!pair.kernel || !pair.bias) throw new Error(`Incomplete weights for ${layerName}`);
      model.getLayer(layerName).setWeights([pair.kernel, pair.bias]);
    }
  } finally {
    grouped.forEach((pair) => {
      if (pair.kernel) pair.kernel.dispose();
      if (pair.bias) pair.bias.dispose();
    });
  }

  return descriptor;
}

async function initializeModel() {
  try {
    setStatus("loading", "Initializing TensorFlow.js…");
    await tf.ready();
    try {
      await tf.setBackend("webgl");
      await tf.ready();
    } catch (backendError) {
      console.warn("WebGL backend unavailable; using current backend.", backendError);
    }
    elements["backend-name"].textContent = `${tf.getBackend()} · client-side`;

    setStatus("loading", "Loading U-Net weights…");
    const model = buildCompactUnet();
    const descriptor = await loadBrowserWeights(model);

    const warmup = tf.zeros([1, MODEL_SIZE, MODEL_SIZE, 1]);
    const warmResult = model.predict(warmup);
    await warmResult.data();
    warmup.dispose();
    warmResult.dispose();

    state.model = model;
    state.ready = true;
    elements["predict-button"].disabled = !state.image;
    setStatus("ready", "Model ready");
    setMessage(state.image ? "Ready to generate a mask." : "Choose an image or a safe synthetic sample.");
    console.info("Browser model ready", descriptor);
  } catch (error) {
    console.error(error);
    setStatus("error", "Model could not load");
    setMessage(`Model error: ${error.message}`, true);
  }
}

function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${url}`));
    image.src = url;
  });
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    if (!file) return reject(new Error("No file selected."));
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("The selected image could not be decoded."));
    };
    image.src = url;
  });
}

function drawPlaceholder(canvas, label) {
  const context = canvas.getContext("2d");
  context.fillStyle = "#020a10";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#6f8997";
  context.font = "600 15px system-ui";
  context.textAlign = "center";
  context.fillText(label, canvas.width / 2, canvas.height / 2);
}

function drawImageContained(canvas, image) {
  const context = canvas.getContext("2d");
  context.fillStyle = "#020a10";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
  const width = image.width * scale;
  const height = image.height * scale;
  const x = (canvas.width - width) / 2;
  const y = (canvas.height - height) / 2;
  context.imageSmoothingEnabled = true;
  context.drawImage(image, x, y, width, height);
}

function imageToModelPixels(image) {
  const canvas = document.createElement("canvas");
  canvas.width = MODEL_SIZE;
  canvas.height = MODEL_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const rgba = context.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const grayscale = new Float32Array(MODEL_SIZE * MODEL_SIZE);
  for (let index = 0, pixel = 0; index < rgba.length; index += 4, pixel += 1) {
    grayscale[pixel] = (0.299 * rgba[index] + 0.587 * rgba[index + 1] + 0.114 * rgba[index + 2]) / 255;
  }
  return grayscale;
}

function maskImageToArray(image) {
  const pixels = imageToModelPixels(image);
  const mask = new Uint8Array(pixels.length);
  for (let i = 0; i < pixels.length; i += 1) mask[i] = pixels[i] >= 0.5 ? 1 : 0;
  return mask;
}

function probabilityColor(value) {
  const v = Math.max(0, Math.min(1, value));
  const stops = [
    [0.00, [2, 10, 16]],
    [0.25, [25, 65, 115]],
    [0.50, [39, 180, 176]],
    [0.75, [248, 205, 94]],
    [1.00, [255, 91, 91]],
  ];
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [leftPosition, leftColor] = stops[i];
    const [rightPosition, rightColor] = stops[i + 1];
    if (v <= rightPosition) {
      const t = (v - leftPosition) / (rightPosition - leftPosition);
      return leftColor.map((channel, j) => Math.round(channel + t * (rightColor[j] - channel)));
    }
  }
  return stops.at(-1)[1];
}

function renderSmallArray(targetCanvas, pixelWriter) {
  const small = document.createElement("canvas");
  small.width = MODEL_SIZE;
  small.height = MODEL_SIZE;
  const context = small.getContext("2d");
  const imageData = context.createImageData(MODEL_SIZE, MODEL_SIZE);
  for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i += 1) {
    const [r, g, b, a = 255] = pixelWriter(i);
    const offset = i * 4;
    imageData.data[offset] = r;
    imageData.data[offset + 1] = g;
    imageData.data[offset + 2] = b;
    imageData.data[offset + 3] = a;
  }
  context.putImageData(imageData, 0, 0);
  const targetContext = targetCanvas.getContext("2d");
  targetContext.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  targetContext.imageSmoothingEnabled = false;
  targetContext.drawImage(small, 0, 0, targetCanvas.width, targetCanvas.height);
  return small;
}

function calculateOverlap(predicted, truth) {
  let intersection = 0;
  let predictedCount = 0;
  let truthCount = 0;
  for (let i = 0; i < predicted.length; i += 1) {
    if (predicted[i]) predictedCount += 1;
    if (truth[i]) truthCount += 1;
    if (predicted[i] && truth[i]) intersection += 1;
  }
  const union = predictedCount + truthCount - intersection;
  const dice = (2 * intersection + 1e-7) / (predictedCount + truthCount + 1e-7);
  const iou = (intersection + 1e-7) / (union + 1e-7);
  return { dice, iou };
}

function renderPrediction() {
  if (!state.probabilities || !state.modelInputPixels) return;
  const threshold = Number(elements.threshold.value);
  const binary = new Uint8Array(state.probabilities.length);
  let positive = 0;
  let probabilitySum = 0;

  for (let i = 0; i < state.probabilities.length; i += 1) {
    const probability = state.probabilities[i];
    probabilitySum += probability;
    binary[i] = probability >= threshold ? 1 : 0;
    positive += binary[i];
  }

  renderSmallArray(elements["probability-canvas"], (i) => [...probabilityColor(state.probabilities[i]), 255]);
  const rawMaskCanvas = renderSmallArray(elements["mask-canvas"], (i) => binary[i] ? [255, 255, 255, 255] : [0, 0, 0, 255]);

  renderSmallArray(elements["overlay-canvas"], (i) => {
    const gray = Math.round(state.modelInputPixels[i] * 255);
    if (binary[i]) return [Math.min(255, gray * 0.35 + 220), Math.round(gray * 0.45), Math.round(gray * 0.45), 255];
    return [gray, gray, gray, 255];
  });

  elements["region-share"].textContent = `${(100 * positive / binary.length).toFixed(2)}%`;
  elements["mean-probability"].textContent = (probabilitySum / state.probabilities.length).toFixed(4);

  if (state.groundTruth) {
    const truth = maskImageToArray(state.groundTruth);
    const scores = calculateOverlap(binary, truth);
    elements["dice-score"].textContent = scores.dice.toFixed(4);
    elements["iou-score"].textContent = scores.iou.toFixed(4);
  } else {
    elements["dice-score"].textContent = "—";
    elements["iou-score"].textContent = "—";
  }

  state.binaryMaskCanvas = rawMaskCanvas;
  elements["download-button"].disabled = false;
}

async function runPrediction() {
  if (!state.ready || !state.model || !state.image) return;
  elements["predict-button"].disabled = true;
  setMessage("Running U-Net inference in your browser…");

  const pixels = imageToModelPixels(state.image);
  state.modelInputPixels = pixels;
  const input = tf.tensor4d(pixels, [1, MODEL_SIZE, MODEL_SIZE, 1], "float32");
  const start = performance.now();
  let output;
  try {
    output = state.model.predict(input);
    const probabilities = await output.data();
    const elapsed = performance.now() - start;
    state.probabilities = Float32Array.from(probabilities);
    elements["inference-time"].textContent = `${elapsed.toFixed(1)} ms`;
    renderPrediction();
    setMessage("Segmentation complete. Adjust the threshold to update the mask instantly.");
  } catch (error) {
    console.error(error);
    setMessage(`Prediction error: ${error.message}`, true);
  } finally {
    input.dispose();
    if (output) output.dispose();
    elements["predict-button"].disabled = false;
  }
}

async function setInputImage(image, description = "Image loaded.") {
  state.image = image;
  state.probabilities = null;
  state.modelInputPixels = null;
  drawImageContained(elements["original-canvas"], image);
  drawPlaceholder(elements["probability-canvas"], "Run the model");
  drawPlaceholder(elements["mask-canvas"], "Run the model");
  drawPlaceholder(elements["overlay-canvas"], "Run the model");
  ["inference-time", "region-share", "mean-probability", "dice-score", "iou-score"].forEach((id) => { elements[id].textContent = "—"; });
  elements["download-button"].disabled = true;
  elements["predict-button"].disabled = !state.ready;
  setMessage(state.ready ? `${description} Ready to segment.` : `${description} Waiting for the model.`);
}

async function handleImageFile(file) {
  try {
    const image = await loadImageFromFile(file);
    await setInputImage(image, `Loaded ${file.name}.`);
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function handleMaskFile(file) {
  try {
    state.groundTruth = file ? await loadImageFromFile(file) : null;
    if (state.probabilities) renderPrediction();
    setMessage(file ? `Ground-truth mask loaded: ${file.name}.` : "Ground-truth mask cleared.");
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function loadSample(button) {
  try {
    setMessage("Loading safe synthetic sample…");
    const [image, mask] = await Promise.all([
      loadImageFromUrl(button.dataset.image),
      loadImageFromUrl(button.dataset.mask),
    ]);
    state.groundTruth = mask;
    elements["mask-input"].value = "";
    await setInputImage(image, `${button.textContent.trim()} loaded with its reference mask.`);
  } catch (error) {
    setMessage(error.message, true);
  }
}

function downloadMask() {
  if (!state.binaryMaskCanvas) return;
  const link = document.createElement("a");
  link.download = "unet-predicted-mask.png";
  link.href = state.binaryMaskCanvas.toDataURL("image/png");
  link.click();
}

function bindEvents() {
  elements["image-input"].addEventListener("change", (event) => handleImageFile(event.target.files[0]));
  elements["mask-input"].addEventListener("change", (event) => handleMaskFile(event.target.files[0]));
  elements["predict-button"].addEventListener("click", runPrediction);
  elements["download-button"].addEventListener("click", downloadMask);
  elements.threshold.addEventListener("input", () => {
    elements["threshold-value"].textContent = Number(elements.threshold.value).toFixed(2);
    renderPrediction();
  });
  elements["sample-grid"].addEventListener("click", (event) => {
    const button = event.target.closest(".sample-button");
    if (button) loadSample(button);
  });

  ["dragenter", "dragover"].forEach((name) => {
    elements["drop-zone"].addEventListener(name, (event) => {
      event.preventDefault();
      elements["drop-zone"].classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    elements["drop-zone"].addEventListener(name, (event) => {
      event.preventDefault();
      elements["drop-zone"].classList.remove("dragging");
    });
  });
  elements["drop-zone"].addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) handleImageFile(file);
  });
}

function initializeCanvases() {
  drawPlaceholder(elements["original-canvas"], "Choose an image");
  drawPlaceholder(elements["probability-canvas"], "Probability map");
  drawPlaceholder(elements["mask-canvas"], "Binary mask");
  drawPlaceholder(elements["overlay-canvas"], "Mask overlay");
}

window.addEventListener("DOMContentLoaded", async () => {
  cacheElements();
  initializeCanvases();
  bindEvents();
  if (typeof tf === "undefined") {
    setStatus("error", "TensorFlow.js failed to load");
    setMessage("The TensorFlow.js CDN was blocked. Check network filters or browser extensions.", true);
    return;
  }
  await initializeModel();
});

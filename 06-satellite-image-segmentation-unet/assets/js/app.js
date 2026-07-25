'use strict';

const CONFIG = Object.freeze({
  inputSize: 64,
  previewSize: 512,
  defaultThreshold: 0.5,
  maxUploadBytes: 12 * 1024 * 1024,
  expectedParameters: 471553,
  modelUrl: './tfjs_model/model.json',
  manifestUrl: './tfjs_model/weights_manifest.json',
  weightsPrefix: './tfjs_model/',
});

const WEIGHTED_LAYERS = [
  'conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5',
  'conv2d_6', 'conv2d_7', 'conv2d_8', 'conv2d_9', 'conv2d_10',
];

const state = {
  model: null,
  modelReady: false,
  modelLoadMode: null,
  sourceImage: null,
  sourceName: '',
  groundTruthImage: null,
  probability: null,
  threshold: CONFIG.defaultThreshold,
  predictionWidth: CONFIG.inputSize,
  predictionHeight: CONFIG.inputSize,
};

const el = {};

window.addEventListener('DOMContentLoaded', () => {
  cacheElements();
  bindEvents();
  initializeApplication().catch((error) => handleFatalError(error));
});

function cacheElements() {
  const ids = [
    'modelStatusChip', 'backendValue', 'summaryThreshold', 'imageUpload', 'maskUpload',
    'maskUploadText', 'uploadZone', 'sampleGrid', 'thresholdSlider', 'thresholdOutput',
    'predictButton', 'predictButtonText', 'statusBox', 'statusText', 'selectedImageLabel',
    'downloadButton', 'inputCanvas', 'maskCanvas', 'overlayCanvas', 'probabilityCanvas',
    'modelCanvas', 'groundTruthCanvas', 'inputEmpty', 'maskEmpty', 'overlayEmpty',
    'probabilityEmpty', 'inferenceTime', 'coverageValue', 'confidenceValue', 'overlapValue',
    'overlapCaption', 'detailedMetrics', 'precisionValue', 'recallValue', 'f1Value',
  ];
  for (const id of ids) el[id] = document.getElementById(id);
}

function bindEvents() {
  el.imageUpload.addEventListener('change', (event) => handleImageFile(event.target.files?.[0]));
  el.maskUpload.addEventListener('change', (event) => handleMaskFile(event.target.files?.[0]));
  el.sampleGrid.addEventListener('click', (event) => {
    const button = event.target.closest('[data-sample]');
    if (button) loadSample(button.dataset.sample, button);
  });
  el.thresholdSlider.addEventListener('input', () => {
    state.threshold = Number(el.thresholdSlider.value);
    el.thresholdOutput.value = state.threshold.toFixed(2);
    el.summaryThreshold.textContent = state.threshold.toFixed(2);
    if (state.probability) renderPredictionOutputs();
  });
  el.predictButton.addEventListener('click', runPrediction);
  el.downloadButton.addEventListener('click', downloadMask);

  for (const eventName of ['dragenter', 'dragover']) {
    el.uploadZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      el.uploadZone.classList.add('dragover');
    });
  }
  for (const eventName of ['dragleave', 'drop']) {
    el.uploadZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      el.uploadZone.classList.remove('dragover');
    });
  }
  el.uploadZone.addEventListener('drop', (event) => handleImageFile(event.dataTransfer.files?.[0]));
}

async function initializeApplication() {
  setStatus('Loading TensorFlow.js runtime…');
  await waitForTensorFlow();

  let backend = 'cpu';
  try {
    if (tf.findBackend('webgl')) {
      await tf.setBackend('webgl');
      backend = 'webgl';
    }
    await tf.ready();
  } catch (error) {
    console.warn('WebGL backend unavailable; using CPU.', error);
    await tf.setBackend('cpu');
    await tf.ready();
    backend = 'cpu';
  }
  el.backendValue.textContent = backend.toUpperCase();

  setStatus('Loading U-Net architecture and browser weights…');
  state.model = await loadModelWithFallback();
  const parameterCount = state.model.countParams();
  if (parameterCount !== CONFIG.expectedParameters) {
    throw new Error(`Unexpected model parameter count: ${parameterCount.toLocaleString()}.`);
  }

  const warmup = tf.tidy(() => {
    const zeros = tf.zeros([1, CONFIG.inputSize, CONFIG.inputSize, 3]);
    return state.model.predict(zeros);
  });
  await warmup.data();
  warmup.dispose();

  state.modelReady = true;
  el.modelStatusChip.textContent = 'Model ready';
  el.modelStatusChip.className = 'status-chip ready';
  setStatus(`Model ready · ${parameterCount.toLocaleString()} parameters · ${state.modelLoadMode}`, 'success');
  updatePredictButton();
}

function waitForTensorFlow(timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const timer = window.setInterval(() => {
      if (window.tf) {
        clearInterval(timer);
        resolve();
      } else if (performance.now() - started > timeoutMs) {
        clearInterval(timer);
        reject(new Error('TensorFlow.js did not load. Check the network connection or content blocker.'));
      }
    }, 50);
  });
}

async function loadModelWithFallback() {
  try {
    const model = await tf.loadLayersModel(CONFIG.modelUrl, { strict: true });
    state.modelLoadMode = 'LayersModel export';
    return model;
  } catch (error) {
    console.warn('Direct LayersModel loading failed; using deterministic architecture fallback.', error);
    const model = buildUnetModel();
    const response = await fetch(CONFIG.manifestUrl, { cache: 'force-cache' });
    if (!response.ok) throw new Error(`Could not load weight manifest (${response.status}).`);
    const manifest = await response.json();
    const weightMap = await tf.io.loadWeights(manifest, CONFIG.weightsPrefix);

    for (const layerName of WEIGHTED_LAYERS) {
      const kernelName = `${layerName}/kernel`;
      const biasName = `${layerName}/bias`;
      const kernel = weightMap[kernelName];
      const bias = weightMap[biasName];
      if (!kernel || !bias) throw new Error(`Missing browser weights for ${layerName}.`);
      model.getLayer(layerName).setWeights([kernel, bias]);
    }
    Object.values(weightMap).forEach((tensor) => tensor.dispose());
    state.modelLoadMode = 'reconstructed U-Net + tf.io.loadWeights';
    return model;
  }
}

function buildUnetModel() {
  const input = tf.input({ shape: [64, 64, 3], name: 'input_layer' });
  const c1 = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', name: 'conv2d' }).apply(input);
  const c1b = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', name: 'conv2d_1' }).apply(c1);
  const p1 = tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'valid', name: 'max_pooling2d' }).apply(c1b);

  const c2 = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu', name: 'conv2d_2' }).apply(p1);
  const c2b = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu', name: 'conv2d_3' }).apply(c2);
  const p2 = tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'valid', name: 'max_pooling2d_1' }).apply(c2b);

  const bottleneck = tf.layers.conv2d({ filters: 128, kernelSize: 3, padding: 'same', activation: 'relu', name: 'conv2d_4' }).apply(p2);
  const bottleneckB = tf.layers.conv2d({ filters: 128, kernelSize: 3, padding: 'same', activation: 'relu', name: 'conv2d_5' }).apply(bottleneck);

  const up1 = tf.layers.upSampling2d({ size: [2, 2], name: 'up_sampling2d' }).apply(bottleneckB);
  const merge1 = tf.layers.concatenate({ axis: -1, name: 'concatenate' }).apply([up1, c2b]);
  const d1 = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu', name: 'conv2d_6' }).apply(merge1);
  const d1b = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu', name: 'conv2d_7' }).apply(d1);

  const up2 = tf.layers.upSampling2d({ size: [2, 2], name: 'up_sampling2d_1' }).apply(d1b);
  const merge2 = tf.layers.concatenate({ axis: -1, name: 'concatenate_1' }).apply([up2, c1b]);
  const d2 = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', name: 'conv2d_8' }).apply(merge2);
  const d2b = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', name: 'conv2d_9' }).apply(d2);
  const output = tf.layers.conv2d({ filters: 1, kernelSize: 1, padding: 'valid', activation: 'sigmoid', name: 'conv2d_10' }).apply(d2b);

  return tf.model({ inputs: input, outputs: output, name: 'functional' });
}

async function handleImageFile(file) {
  if (!file) return;
  validateImageFile(file);
  const image = await fileToImage(file);
  setSourceImage(image, file.name, null);
}

async function handleMaskFile(file) {
  if (!file) {
    state.groundTruthImage = null;
    return;
  }
  validateImageFile(file);
  state.groundTruthImage = await fileToImage(file);
  el.maskUploadText.textContent = file.name;
  if (state.probability) renderPredictionOutputs();
}

function validateImageFile(file) {
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
    throw new Error('Use a PNG, JPG, or WebP image.');
  }
  if (file.size > CONFIG.maxUploadBytes) {
    throw new Error('The selected image exceeds the 12 MB browser-demo limit.');
  }
}

function fileToImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => { URL.revokeObjectURL(url); resolve(image); };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error('The selected image could not be decoded.')); };
    image.src = url;
  });
}

async function loadSample(sampleId, button) {
  try {
    document.querySelectorAll('.sample-button').forEach((node) => node.classList.toggle('active', node === button));
    const imageUrl = `./assets/samples/images/synthetic_tile_${sampleId}.png`;
    const maskUrl = `./assets/samples/masks/synthetic_tile_${sampleId}_mask.png`;
    const [image, mask] = await Promise.all([loadImageUrl(imageUrl), loadImageUrl(maskUrl)]);
    setSourceImage(image, `Synthetic tile ${sampleId}`, mask);
    el.maskUploadText.textContent = `Sample ${sampleId} mask loaded`;
  } catch (error) {
    handleRecoverableError(error);
  }
}

function loadImageUrl(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load sample asset: ${url}`));
    image.src = url;
  });
}

function setSourceImage(image, name, groundTruth) {
  state.sourceImage = image;
  state.sourceName = name;
  state.groundTruthImage = groundTruth;
  state.probability = null;
  drawImageSquare(el.inputCanvas, image, CONFIG.previewSize, true);
  el.inputEmpty.classList.add('hidden');
  for (const id of ['maskEmpty', 'overlayEmpty', 'probabilityEmpty']) el[id].classList.remove('hidden');
  clearCanvas(el.maskCanvas);
  clearCanvas(el.overlayCanvas);
  clearCanvas(el.probabilityCanvas);
  el.selectedImageLabel.textContent = name;
  el.downloadButton.disabled = true;
  resetMetrics();
  setStatus('Image ready. Run the U-Net segmentation.', 'success');
  updatePredictButton();
}

function updatePredictButton() {
  const ready = state.modelReady && Boolean(state.sourceImage);
  el.predictButton.disabled = !ready;
  el.predictButtonText.textContent = ready ? 'Generate segmentation' : 'Waiting for model and image';
}

async function runPrediction() {
  if (!state.modelReady || !state.sourceImage) return;
  el.predictButton.disabled = true;
  el.predictButtonText.textContent = 'Running U-Net…';
  setStatus(`Running client-side inference with ${tf.getBackend().toUpperCase()}…`);

  try {
    const started = performance.now();
    drawImageSquare(el.modelCanvas, state.sourceImage, CONFIG.inputSize, true);

    const inputTensor = tf.tidy(() =>
      tf.browser.fromPixels(el.modelCanvas, 3)
        .toFloat()
        .div(255)
        .expandDims(0)
    );
    const outputTensor = state.model.predict(inputTensor);
    const probability = await outputTensor.data();
    inputTensor.dispose();
    outputTensor.dispose();

    state.probability = Float32Array.from(probability);
    renderPredictionOutputs();
    const elapsed = performance.now() - started;
    el.inferenceTime.textContent = `${elapsed.toFixed(0)} ms`;
    el.downloadButton.disabled = false;
    setStatus(`Segmentation complete · ${state.modelLoadMode} · image processed locally`, 'success');
  } catch (error) {
    handleRecoverableError(error);
  } finally {
    el.predictButton.disabled = false;
    el.predictButtonText.textContent = 'Generate segmentation again';
  }
}

function renderPredictionOutputs() {
  if (!state.probability || !state.sourceImage) return;
  const threshold = state.threshold;
  const mask = new Uint8Array(state.probability.length);
  let positive = 0;
  let confidenceSum = 0;
  for (let i = 0; i < state.probability.length; i += 1) {
    if (state.probability[i] >= threshold) {
      mask[i] = 1;
      positive += 1;
      confidenceSum += state.probability[i];
    }
  }

  renderBinaryMask(mask, el.maskCanvas);
  renderProbabilityMap(state.probability, el.probabilityCanvas);
  renderOverlay(state.sourceImage, mask, el.overlayCanvas);
  for (const id of ['maskEmpty', 'overlayEmpty', 'probabilityEmpty']) el[id].classList.add('hidden');

  const coverage = positive / mask.length;
  el.coverageValue.textContent = `${(coverage * 100).toFixed(1)}%`;
  el.confidenceValue.textContent = positive ? `${((confidenceSum / positive) * 100).toFixed(1)}%` : '—';
  updateGroundTruthMetrics(mask);
}

function renderBinaryMask(mask, canvas) {
  const small = document.createElement('canvas');
  small.width = CONFIG.inputSize;
  small.height = CONFIG.inputSize;
  const ctx = small.getContext('2d');
  const imageData = ctx.createImageData(CONFIG.inputSize, CONFIG.inputSize);
  for (let i = 0; i < mask.length; i += 1) {
    const value = mask[i] ? 255 : 0;
    const offset = i * 4;
    imageData.data[offset] = value;
    imageData.data[offset + 1] = value;
    imageData.data[offset + 2] = value;
    imageData.data[offset + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  const target = canvas.getContext('2d');
  target.clearRect(0, 0, canvas.width, canvas.height);
  target.imageSmoothingEnabled = false;
  target.drawImage(small, 0, 0, canvas.width, canvas.height);
}

function renderProbabilityMap(probability, canvas) {
  const small = document.createElement('canvas');
  small.width = CONFIG.inputSize;
  small.height = CONFIG.inputSize;
  const ctx = small.getContext('2d');
  const imageData = ctx.createImageData(CONFIG.inputSize, CONFIG.inputSize);
  for (let i = 0; i < probability.length; i += 1) {
    const [r, g, b] = probabilityColor(probability[i]);
    const offset = i * 4;
    imageData.data[offset] = r;
    imageData.data[offset + 1] = g;
    imageData.data[offset + 2] = b;
    imageData.data[offset + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  const target = canvas.getContext('2d');
  target.clearRect(0, 0, canvas.width, canvas.height);
  target.imageSmoothingEnabled = true;
  target.drawImage(small, 0, 0, canvas.width, canvas.height);
}

function probabilityColor(value) {
  const stops = [
    [0.00, [7, 17, 31]],
    [0.25, [28, 72, 128]],
    [0.50, [49, 184, 183]],
    [0.75, [238, 208, 96]],
    [1.00, [255, 122, 114]],
  ];
  const clamped = Math.max(0, Math.min(1, value));
  for (let i = 1; i < stops.length; i += 1) {
    if (clamped <= stops[i][0]) {
      const [x0, c0] = stops[i - 1];
      const [x1, c1] = stops[i];
      const t = (clamped - x0) / (x1 - x0);
      return c0.map((channel, index) => Math.round(channel + (c1[index] - channel) * t));
    }
  }
  return stops.at(-1)[1];
}

function renderOverlay(image, mask, canvas) {
  drawImageSquare(canvas, image, CONFIG.previewSize, true);
  const smallMask = document.createElement('canvas');
  smallMask.width = CONFIG.inputSize;
  smallMask.height = CONFIG.inputSize;
  const smallCtx = smallMask.getContext('2d');
  const imageData = smallCtx.createImageData(CONFIG.inputSize, CONFIG.inputSize);
  for (let i = 0; i < mask.length; i += 1) {
    const offset = i * 4;
    imageData.data[offset] = 255;
    imageData.data[offset + 1] = 88;
    imageData.data[offset + 2] = 78;
    imageData.data[offset + 3] = mask[i] ? 155 : 0;
  }
  smallCtx.putImageData(imageData, 0, 0);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(smallMask, 0, 0, canvas.width, canvas.height);
}

function updateGroundTruthMetrics(predictedMask) {
  if (!state.groundTruthImage) {
    el.overlapValue.textContent = '—';
    el.overlapCaption.textContent = 'Add a ground-truth mask';
    el.detailedMetrics.classList.add('hidden');
    return;
  }

  drawImageSquare(el.groundTruthCanvas, state.groundTruthImage, CONFIG.inputSize, false);
  const ctx = el.groundTruthCanvas.getContext('2d', { willReadFrequently: true });
  const pixels = ctx.getImageData(0, 0, CONFIG.inputSize, CONFIG.inputSize).data;
  let tp = 0; let fp = 0; let fn = 0;
  for (let i = 0; i < predictedMask.length; i += 1) {
    const offset = i * 4;
    const actual = ((pixels[offset] + pixels[offset + 1] + pixels[offset + 2]) / 3) >= 127 ? 1 : 0;
    const predicted = predictedMask[i];
    if (predicted && actual) tp += 1;
    else if (predicted && !actual) fp += 1;
    else if (!predicted && actual) fn += 1;
  }
  const epsilon = 1e-7;
  const dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon);
  const iou = (tp + epsilon) / (tp + fp + fn + epsilon);
  const precision = (tp + epsilon) / (tp + fp + epsilon);
  const recall = (tp + epsilon) / (tp + fn + epsilon);
  const f1 = (2 * precision * recall) / (precision + recall + epsilon);
  el.overlapValue.textContent = `${dice.toFixed(4)} / ${iou.toFixed(4)}`;
  el.overlapCaption.textContent = 'Dice / IoU for supplied mask';
  el.precisionValue.textContent = precision.toFixed(4);
  el.recallValue.textContent = recall.toFixed(4);
  el.f1Value.textContent = f1.toFixed(4);
  el.detailedMetrics.classList.remove('hidden');
}

function drawImageContained(canvas, image) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#03070d';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const scale = Math.min(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
  const width = image.naturalWidth * scale;
  const height = image.naturalHeight * scale;
  const x = (canvas.width - width) / 2;
  const y = (canvas.height - height) / 2;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(image, x, y, width, height);
}

function drawImageSquare(canvas, image, size, smoothing) {
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.clearRect(0, 0, size, size);
  ctx.imageSmoothingEnabled = smoothing;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(image, 0, 0, size, size);
}

function clearCanvas(canvas) {
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

function resetMetrics() {
  el.inferenceTime.textContent = '—';
  el.coverageValue.textContent = '—';
  el.confidenceValue.textContent = '—';
  el.overlapValue.textContent = '—';
  el.overlapCaption.textContent = 'Add a ground-truth mask';
  el.detailedMetrics.classList.add('hidden');
}

function downloadMask() {
  if (!state.probability) return;
  const name = (state.sourceName || 'satellite-image').replace(/[^a-z0-9-_]+/gi, '-').toLowerCase();
  const link = document.createElement('a');
  link.download = `${name}-unet-mask.png`;
  link.href = el.maskCanvas.toDataURL('image/png');
  link.click();
}

function setStatus(message, kind = 'loading') {
  el.statusText.textContent = message;
  el.statusBox.className = `status-box${kind === 'loading' ? '' : ` ${kind}`}`;
}

function handleRecoverableError(error) {
  console.error(error);
  setStatus(error instanceof Error ? error.message : String(error), 'error');
}

function handleFatalError(error) {
  console.error(error);
  state.modelReady = false;
  el.modelStatusChip.textContent = 'Model error';
  el.modelStatusChip.className = 'status-chip error';
  setStatus(error instanceof Error ? error.message : String(error), 'error');
  updatePredictButton();
}

window.addEventListener('error', (event) => {
  if (event.error) handleRecoverableError(event.error);
});
window.addEventListener('unhandledrejection', (event) => {
  handleRecoverableError(event.reason);
});

'use strict';

const MODEL_URL = './tfjs_model/model.json';
const METADATA_URL = './metadata.json';
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/bmp']);

let model = null;
let metadata = null;
let selectedObjectUrl = null;
let predictionRunning = false;

const elements = {};

document.addEventListener('DOMContentLoaded', async () => {
  Object.assign(elements, {
    status: document.getElementById('modelStatus'),
    imageInput: document.getElementById('imageInput'),
    dropZone: document.getElementById('dropZone'),
    previewImage: document.getElementById('previewImage'),
    previewPlaceholder: document.getElementById('previewPlaceholder'),
    predictButton: document.getElementById('predictButton'),
    inputMessage: document.getElementById('inputMessage'),
    resultEmpty: document.getElementById('resultEmpty'),
    resultContent: document.getElementById('resultContent'),
    predictedClass: document.getElementById('predictedClass'),
    confidenceValue: document.getElementById('confidenceValue'),
    confidenceBar: document.getElementById('confidenceBar'),
    topPredictions: document.getElementById('topPredictions'),
    predictionSummary: document.getElementById('predictionSummary'),
  });

  wireInteractions();
  await initializeModel();
});

function setStatus(message, state = '') {
  elements.status.className = `status ${state}`.trim();
  elements.status.querySelector('span:last-child').textContent = message;
}

async function initializeModel() {
  try {
    if (typeof tf === 'undefined') {
      throw new Error('TensorFlow.js did not load. Check your internet connection or CDN access.');
    }
    setStatus('Loading metadata…');
    const response = await fetch(METADATA_URL, { cache: 'no-cache' });
    if (!response.ok) throw new Error(`Metadata request failed (${response.status}).`);
    metadata = await response.json();

    setStatus('Loading ResNet50 model…');
    model = await tf.loadLayersModel(MODEL_URL);
    validateLoadedModel();

    // Warm-up reduces latency for the first user-triggered prediction.
    tf.tidy(() => model.predict(tf.zeros([1, 96, 96, 3])).dataSync());
    setStatus(`Model ready · ${tf.getBackend()}`, 'ready');
    updatePredictButton();
  } catch (error) {
    console.error(error);
    setStatus('Model unavailable', 'error');
    showMessage(`${error.message} Serve this folder through HTTP and confirm all model shards are present.`);
  }
}

function validateLoadedModel() {
  const inputShape = model.inputs[0].shape;
  const outputShape = model.outputs[0].shape;
  if (inputShape.length !== 4 || inputShape[1] !== 96 || inputShape[2] !== 96 || inputShape[3] !== 3) {
    throw new Error(`Unexpected model input shape: ${JSON.stringify(inputShape)}.`);
  }
  if (outputShape.at(-1) !== metadata.num_classes) {
    throw new Error(`Model output has ${outputShape.at(-1)} classes; metadata has ${metadata.num_classes}.`);
  }
}

function wireInteractions() {
  elements.imageInput.addEventListener('change', (event) => {
    const [file] = event.target.files;
    if (file) selectLocalFile(file);
  });
  elements.predictButton.addEventListener('click', runPrediction);

  for (const eventName of ['dragenter', 'dragover']) {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.add('dragging');
    });
  }
  for (const eventName of ['dragleave', 'drop']) {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.remove('dragging');
    });
  }
  elements.dropZone.addEventListener('drop', (event) => {
    const [file] = event.dataTransfer.files;
    if (file) selectLocalFile(file);
  });

  document.querySelectorAll('.sample-button').forEach((button) => {
    button.addEventListener('click', () => selectSample(button.dataset.sample));
  });
}

function validateFile(file) {
  if (!ALLOWED_TYPES.has(file.type)) {
    throw new Error('Choose a PNG, JPEG, WebP, or BMP image.');
  }
  if (file.size > MAX_FILE_BYTES) {
    throw new Error('The selected image is larger than the recommended 10 MB limit.');
  }
}

function selectLocalFile(file) {
  try {
    validateFile(file);
    if (selectedObjectUrl) URL.revokeObjectURL(selectedObjectUrl);
    selectedObjectUrl = URL.createObjectURL(file);
    setPreviewSource(selectedObjectUrl);
  } catch (error) {
    showMessage(error.message);
  }
}

async function selectSample(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Sample image could not be loaded.');
    const blob = await response.blob();
    if (selectedObjectUrl) URL.revokeObjectURL(selectedObjectUrl);
    selectedObjectUrl = URL.createObjectURL(blob);
    setPreviewSource(selectedObjectUrl);
  } catch (error) {
    showMessage(error.message);
  }
}

function setPreviewSource(src) {
  showMessage('');
  resetResults();
  elements.previewImage.onload = () => {
    elements.previewImage.hidden = false;
    elements.previewPlaceholder.hidden = true;
    updatePredictButton();
  };
  elements.previewImage.onerror = () => {
    showMessage('The selected file could not be decoded as an image.');
    elements.previewImage.hidden = true;
    elements.previewPlaceholder.hidden = false;
    updatePredictButton();
  };
  elements.previewImage.src = src;
}

function updatePredictButton() {
  elements.predictButton.disabled = !model || !elements.previewImage.src || !elements.previewImage.complete || predictionRunning;
}

function preprocessImage(imageElement) {
  return tf.tidy(() => {
    const rgb = tf.browser.fromPixels(imageElement, 3)
      .resizeBilinear([metadata.browser_input.height, metadata.browser_input.width], true)
      .toFloat();
    const [red, green, blue] = tf.split(rgb, 3, 2);
    const bgrMeanSubtracted = tf.concat([
      blue.sub(103.939),
      green.sub(116.779),
      red.sub(123.68),
    ], 2);
    return bgrMeanSubtracted.expandDims(0);
  });
}

async function runPrediction() {
  if (!model || predictionRunning) return;
  predictionRunning = true;
  updatePredictButton();
  elements.predictButton.textContent = 'Running inference…';
  showMessage('');

  let inputTensor;
  let outputTensor;
  try {
    inputTensor = preprocessImage(elements.previewImage);
    outputTensor = model.predict(inputTensor);
    const probabilities = Array.from(await outputTensor.data());
    if (probabilities.length !== metadata.num_classes) {
      throw new Error(`Expected ${metadata.num_classes} probabilities but received ${probabilities.length}.`);
    }
    renderPrediction(probabilities);
  } catch (error) {
    console.error(error);
    showMessage(`Prediction failed: ${error.message}`);
  } finally {
    inputTensor?.dispose();
    outputTensor?.dispose();
    predictionRunning = false;
    elements.predictButton.textContent = 'Run browser prediction';
    updatePredictButton();
  }
}

function topK(probabilities, k = 3) {
  return probabilities
    .map((probability, index) => ({ index, probability, label: metadata.class_names[index] }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, k);
}

function renderPrediction(probabilities) {
  const ranked = topK(probabilities, 3);
  const best = ranked[0];
  elements.resultEmpty.hidden = true;
  elements.resultContent.hidden = false;
  elements.predictedClass.textContent = formatLabel(best.label);
  elements.confidenceValue.textContent = formatPercent(best.probability);
  elements.confidenceBar.style.width = `${Math.max(1, best.probability * 100)}%`;
  elements.topPredictions.replaceChildren(...ranked.map(createProbabilityItem));

  const level = best.probability >= 0.75 ? 'high' : best.probability >= 0.45 ? 'moderate' : 'low';
  elements.predictionSummary.textContent = `The model's highest-scoring class is ${formatLabel(best.label)} with ${formatPercent(best.probability)} confidence. This is a ${level}-confidence model estimate and may be wrong, especially for images unlike CIFAR-100.`;
}

function createProbabilityItem(item) {
  const li = document.createElement('li');
  li.className = 'probability-item';
  const name = document.createElement('span');
  name.className = 'probability-name';
  name.textContent = formatLabel(item.label);
  const track = document.createElement('span');
  track.className = 'probability-track';
  const fill = document.createElement('span');
  fill.style.width = `${Math.max(1, item.probability * 100)}%`;
  track.appendChild(fill);
  const value = document.createElement('span');
  value.className = 'probability-value';
  value.textContent = formatPercent(item.probability);
  li.append(name, track, value);
  return li;
}

function resetResults() {
  elements.resultEmpty.hidden = false;
  elements.resultContent.hidden = true;
  elements.confidenceBar.style.width = '0%';
}

function showMessage(message) {
  elements.inputMessage.textContent = message;
}

function formatLabel(label) {
  return label.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatPercent(value) {
  return new Intl.NumberFormat('en-US', { style: 'percent', maximumFractionDigits: 1 }).format(value);
}

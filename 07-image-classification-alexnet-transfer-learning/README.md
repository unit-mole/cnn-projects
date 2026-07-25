# AlexNet-Style Image Classification and Transfer-Learning Baseline

[![CI](https://github.com/unit-mole/cnn-projects/actions/workflows/07-image-classification-alexnet-transfer-learning.yml/badge.svg)](https://github.com/unit-mole/cnn-projects/actions/workflows/07-image-classification-alexnet-transfer-learning.yml)

A recruiter-friendly CNN portfolio project that trains a browser-aware **AlexNet-style convolutional neural network** for CIFAR-10 image classification, compares it with a **MobileNetV2 transfer-learning baseline**, converts the trained Keras model to **TensorFlow.js**, and serves predictions from a static **GitHub Pages** application with **Cloudflare Pages** as an optional fallback.

> **Artifact status:** The uploaded source notebook was a neural-style-transfer notebook, not an AlexNet classifier. This package therefore contains a complete, honest classifier implementation and a tiny **smoke-test-only** TensorFlow.js model so the browser interface can be validated immediately. Train and export the actual model before presenting prediction quality or production metrics.

## Live demo links

- **Primary — GitHub Pages:** `https://unit-mole.github.io/cnn-projects/07-image-classification-alexnet-transfer-learning/`
- **Fallback — Cloudflare Pages:** `https://YOUR-PROJECT.pages.dev`
- **Training notebook:** `ADD_KAGGLE_OR_COLAB_URL`

## Responsible-use notice

This project is for educational and portfolio demonstration purposes only. The model may classify images incorrectly, especially when inputs are unclear, out-of-distribution, low-quality, cropped, or different from the training data. Do not use the model as the sole basis for medical, legal, safety-critical, security, hiring, insurance, financial, quality-release, or production decisions. Do not upload private, sensitive, confidential, copyrighted, or personally identifiable images to a public demo. Predictions are machine-learning outputs, not guaranteed truth.

## Problem statement

Given an image, can a CNN classify it into the correct CIFAR-10 category directly inside the browser?

The application returns:

- uploaded image preview,
- predicted class,
- confidence score,
- top class probabilities,
- model and preprocessing details,
- artifact-status warning,
- responsible-use guidance.

## Source-file review

The supplied notebook implements neural style transfer with a pretrained VGG19 feature extractor. It uses one CIFAR-10 image as content, creates a synthetic style image, optimizes content/style/total-variation losses, and produces a content/style/stylized triptych. It does **not** contain an AlexNet classifier, label-driven training, transfer-learning comparison, classification metrics, or a TensorFlow.js classification artifact. The original files are preserved under `archive/source-neural-style-transfer/` for traceability. See `PROJECT_AUDIT.md`.

## Dataset

The reusable dataset reference in the attached notebook is **CIFAR-10**. This project uses the complete classification dataset through `tf.keras.datasets.cifar10`.

| Property | Value |
|---|---|
| Image format | RGB arrays loaded by TensorFlow |
| Original image size | 32 × 32 × 3 |
| Training images | 50,000 |
| Test images | 10,000 |
| Classes | 10 |
| Task type | Multi-class image classification |
| Class names | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Model input | resized to 227 × 227 × 3 |
| Normalization | float32 pixels scaled to [0, 1] |

The training loader creates a stratified validation split from the training set. The full dataset is downloaded at runtime and is not committed to GitHub.

## Why AlexNet-style rather than a false pretrained-AlexNet claim?

AlexNet helped popularize deep learning for image classification. It uses convolution layers to learn visual patterns, pooling to reduce spatial dimensions, ReLU nonlinearities, and dense layers for classification. Newer architectures are usually more efficient, but AlexNet remains valuable for understanding the foundations of modern CNNs.

This project uses an **AlexNet-style architecture trained from scratch**. It does not claim ImageNet-pretrained AlexNet weights because those weights are not supplied by the standard TensorFlow/Keras applications interface used here. Transfer learning is demonstrated honestly through a separate MobileNetV2 baseline with ImageNet weights.

## Browser-aware AlexNet-style architecture

```text
Input 227×227×3
→ Conv 11×11, stride 4 + BatchNorm + ReLU + MaxPool
→ Conv 5×5 + BatchNorm + ReLU + MaxPool
→ Conv 3×3 + ReLU
→ Conv 3×3 + ReLU
→ Conv 3×3 + ReLU + MaxPool
→ Global Average Pooling
→ Dense 512 + Dropout
→ Dense 256 + Dropout
→ Softmax classifier
```

The classic 4096-unit fully connected AlexNet head is replaced with global average pooling and smaller dense layers. This keeps AlexNet-inspired feature extraction while reducing browser download size and inference latency.

## Transfer-learning baseline

Transfer learning reuses general visual features learned from a large source dataset and adapts them to a target dataset. The MobileNetV2 baseline freezes an ImageNet-pretrained feature extractor, trains a CIFAR-10 classification head, and can optionally fine-tune selected upper layers. It provides a modern comparison without misrepresenting the AlexNet model as pretrained.

## Data pipeline

1. Load CIFAR-10 or a folder-based custom dataset.
2. Decode images as RGB.
3. Resize to 227 × 227.
4. Convert to float32 and normalize to [0, 1].
5. Create stratified training and validation indices.
6. Apply safe augmentation only to training data: horizontal flip, small rotation, zoom, and contrast adjustment.
7. Cache/prefetch datasets.
8. Train the selected architecture.
9. Evaluate with more than accuracy.
10. Export the chosen Keras model to TensorFlow.js.

The same input size and normalization are stored in `models/model_metadata.json` and `web/metadata.json`, and the JavaScript preprocessing mirrors the Python pipeline.

## Evaluation

The evaluation code saves:

- accuracy,
- macro and weighted precision,
- macro and weighted recall,
- macro and weighted F1-score,
- top-k accuracy,
- per-class classification report,
- confusion matrix,
- one-vs-rest ROC-AUC when valid,
- training curves,
- sample predictions.

Accuracy alone can hide weak minority-class performance. Macro F1 weights each class equally, weighted F1 reflects class frequency, and per-class recall shows which categories are being missed.

## Browser inference workflow

```text
Keras model
→ scripts/convert_to_tfjs.py
→ models/tfjs_model/model.json + binary shards
→ copied into web/tfjs_model/
→ tf.loadLayersModel('./tfjs_model/model.json')
→ browser image preprocessing
→ model.predict()
→ top-k probabilities displayed
```

TensorFlow.js loads model topology from `model.json` and weights from the referenced binary shard files. The web app runs without a Python backend.

## Quick start

### 1. Create an environment

Python 3.11 is recommended for the broadest TensorFlow/TensorFlow.js converter compatibility.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Train AlexNet-style model

```bash
python scripts/train_model.py --model alexnet --epochs 20
```

### 4. Train the transfer-learning baseline

```bash
python scripts/train_model.py --model mobilenetv2 --epochs 12
```

### 5. Evaluate a trained model

```bash
python scripts/evaluate_model.py --model-path models/alexnet_cifar10.keras
```

### 6. Convert the selected model to TensorFlow.js

```bash
python scripts/convert_to_tfjs.py --model-path models/alexnet_cifar10.keras
```

This replaces the smoke-test model in both `models/tfjs_model/` and `web/tfjs_model/` and updates metadata to `trained`.

### 7. Run the browser demo

```bash
python scripts/run_local_web_server.py --port 8000
```

Open `http://localhost:8000`.

## Custom folder dataset

A custom dataset can use this layout:

```text
data/raw/custom_dataset/
├── class_a/
├── class_b/
└── class_c/
```

Run:

```bash
python scripts/train_model.py \
  --dataset folder \
  --data-dir data/raw/custom_dataset \
  --model alexnet \
  --image-size 227 \
  --epochs 20
```

Class names are inferred from folder names and stored in metadata.

## Cloudflare Pages

The static app lives entirely in `web/`. In a Git-connected monorepo setup, configure:

- root directory: `07-image-classification-alexnet-transfer-learning`
- build command: leave blank
- build output directory: `web`

You can also run `npm run deploy` after authenticating Wrangler. See `README_CLOUDFLARE.md`.

## GitHub Pages deployment

The root workflow `.github/workflows/04-image-classification-resnet.yml` automatically assembles and deploys both the existing ResNet demo and this project. The AlexNet-style app is published at `/07-image-classification-alexnet-transfer-learning/`. See `README_GITHUB_PAGES.md` and the root `GITHUB_PAGES_DEPLOYMENT.md`.

## Folder structure

```text
07-image-classification-alexnet-transfer-learning/
├── .streamlit/
├── app/
├── archive/source-neural-style-transfer/
├── data/{raw,processed,sample_images}/
├── images/
├── models/{tfjs_model}/
├── notebooks/
├── outputs/{metrics,predictions,reports,visualizations}/
├── scripts/
├── src/
├── tests/
├── web/{sample_images,tfjs_model}/
├── Dockerfile
├── FILE_MANIFEST.csv
├── IMPROVEMENTS.md
├── MONOREPO_INTEGRATION.md
├── PROJECT_AUDIT.md
├── README.md
├── README_CLOUDFLARE.md
├── README_GITHUB_PAGES.md
├── README_HOSTING.md
├── package.json
├── requirements.txt
├── requirements-dev.txt
├── run_local.bat
├── run_local.sh
├── train_model.py
└── wrangler.toml
```

## Portfolio wording

**One-line description**

> Browser-deployed CIFAR-10 image classifier using a compact AlexNet-style CNN, a MobileNetV2 transfer-learning baseline, TensorFlow.js, Cloudflare Pages, and rigorous classification evaluation.

**Pinned-repository description**

> End-to-end computer-vision project covering image preprocessing, AlexNet-style CNN training, transfer-learning comparison, class-level evaluation, TensorFlow.js conversion, and static browser inference on Cloudflare Pages.

## Skills demonstrated

CNN architecture design, AlexNet concepts, transfer learning, image preprocessing, augmentation, stratified splitting, class mapping, model evaluation, error analysis, Keras serialization, TensorFlow.js conversion, browser inference, static-site deployment, Cloudflare Pages, GitHub Pages, automated tests, GitHub Actions, and responsible AI communication.

## Screenshots to add after training

1. Browser upload and prediction screen.
2. Top-5 probability panel.
3. Training/validation accuracy and loss curves.
4. Confusion matrix.
5. Correct and incorrect prediction gallery.
6. Cloudflare Pages deployment screen.
7. GitHub Actions passing workflow.

## Limitations

- CIFAR-10 images are very small and can become blurry when resized.
- Similar classes such as cat/dog or automobile/truck may be confused.
- Background bias and limited image diversity affect generalization.
- The included browser artifact is a smoke test until replaced by a trained export.
- The project is not suitable for operational quality-release decisions.

## Future improvements

- train longer with tuned augmentation and learning-rate schedules,
- add calibration and confidence reliability diagrams,
- add Grad-CAM to Python analysis,
- quantize the TensorFlow.js model,
- compare AlexNet-style, MobileNetV2, and EfficientNet-Lite,
- add model-card documentation and dataset-version tracking,
- add Cloudflare deployment automation after secrets are configured.

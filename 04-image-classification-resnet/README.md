# ResNet50 Image Classification — CIFAR-100 Browser Inference

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](#continuous-integration)
[![Browser ML](https://img.shields.io/badge/Browser_ML-TensorFlow.js-FF6F00?logo=tensorflow&logoColor=white)](#browser-based-inference)
[![Deployment](https://img.shields.io/badge/Deployment-GitHub_Pages-222222?logo=github&logoColor=white)](#github-pages-deployment)

An end-to-end **100-class image-classification project** that applies an ImageNet-pretrained **ResNet50** backbone to CIFAR-100 and deploys inference as a static TensorFlow.js application. Uploaded images are processed locally in the browser; no Python server is required for the live demo.

> **Responsible-use notice:** This project is for education and portfolio demonstration only. The model can misclassify unclear, low-quality, out-of-distribution, or unfamiliar images. Do not use its output as the sole basis for medical, legal, safety-critical, security, hiring, insurance, financial, or production decisions. Do not upload private, sensitive, confidential, copyrighted, or personally identifiable images to a public demo. Predictions are machine-learning estimates, not guaranteed truth.

## Portfolio summary

**One-line description:** ResNet50 transfer learning for CIFAR-100 image classification, converted to TensorFlow.js for privacy-preserving browser inference on GitHub Pages.

**Pinned-repository description:** Built and evaluated a 100-class ResNet50 image classifier, packaged reproducible Python pipelines, converted the trained model into sharded TensorFlow.js assets, and deployed a responsive no-backend browser demo.

**Live demo:** `https://<your-github-username>.github.io/cnn-projects/`  
**Notebook:** [`notebooks/image_classification_resnet.ipynb`](./notebooks/image_classification_resnet.ipynb)

## Verified project results

These values come from the supplied project metrics artifact and are not regenerated or invented in this repository build.

| Model | Validation accuracy | Test accuracy | Top-5 test accuracy |
|---|---:|---:|---:|
| Logistic-regression baseline | 0.1262 | 0.1257 | — |
| ResNet50 transfer learning | 0.6707 | **0.6709** | **0.9120** |

The ResNet model improves absolute test accuracy by **54.52 percentage points** over the flattened-image logistic-regression baseline.

Precision, recall, F1, confusion-matrix, ROC, and precision-recall artifacts are intentionally **not fabricated**. `scripts/evaluate_model.py` generates them from real model predictions when TensorFlow and CIFAR-100 are available.

## Problem statement

Given an RGB image, classify it into one of the 100 CIFAR-100 fine-label categories and return:

- the predicted class;
- model confidence;
- the top three class probabilities;
- model and preprocessing details;
- a clear limitation and responsible-use note.

## Dataset

The attached notebook uses `tf.keras.datasets.cifar100.load_data(label_mode="fine")`.

| Property | Value |
|---|---|
| Task | Multi-class image classification |
| Dataset | CIFAR-100 fine labels |
| Original train images | 50,000 |
| Original test images | 10,000 |
| Notebook split | 40,000 train / 10,000 validation / 10,000 test |
| Original image shape | 32 × 32 × 3 RGB |
| Number of classes | 100 |
| Class balance | CIFAR-100 contains 500 original training and 100 test images per fine class; the notebook's final 10,000-sample validation slice is not claimed to be perfectly stratified |

The full dataset is downloaded by TensorFlow at runtime and is not committed to GitHub. Only generated, non-sensitive browser test images are included.

## Why ResNet

ResNet is a convolutional architecture that uses **residual or skip connections**. Instead of forcing every group of layers to learn a complete transformation, a residual block learns a change that is added back to its input. This improves gradient flow and makes deep networks easier to optimize. For transfer learning, ResNet50 offers strong reusable visual features learned from ImageNet while allowing a compact task-specific classification head to be trained for CIFAR-100.

## Architecture used by the attached project

```text
32×32 RGB input
    ↓
Resize to 96×96
    ↓
Training-only augmentation
    ↓
ResNet ImageNet preprocessing
    ↓
Frozen ResNet50 backbone (include_top=False)
    ↓
Global Average Pooling
    ↓
Dense(512, ReLU)
    ↓
Batch Normalization
    ↓
Dropout(0.5)
    ↓
Dense(100, Softmax)
```

The browser export accepts a preprocessed **96×96×3** tensor. Resizing and ResNet preprocessing are deliberately implemented in `web/app.js`, which removes training-only augmentation and unsupported generated preprocessing operations from the exported inference graph.

## Preprocessing consistency

The three inference paths use the same logic:

1. Decode as RGB.
2. Resize to 96×96.
3. Convert values to the 0–255 range.
4. Convert RGB to BGR.
5. Subtract ImageNet means `[103.939, 116.779, 123.68]` in BGR order.
6. Add the batch dimension.
7. Run softmax output inference and rank class probabilities.

Training starts with CIFAR-100 arrays normalized to `[0, 1]`; the Keras pipeline multiplies by 255 before `tf.keras.applications.resnet.preprocess_input`. The JavaScript code reproduces the resulting BGR mean-subtraction behavior directly.

## Training strategy

- Reproducibility seed: `42`.
- Frozen ImageNet-pretrained ResNet50 feature extractor.
- Adam optimizer with learning rate `0.001`.
- Categorical cross-entropy and accuracy.
- Batch size `128` and up to `12` epochs in the notebook.
- Early stopping and learning-rate reduction callbacks.
- Safe augmentation: horizontal flip, small rotation, zoom, and contrast variation.

A future fine-tuning stage can unfreeze the final ResNet block with a lower learning rate after the classification head has converged. This repository keeps that behavior configurable rather than silently changing the supplied experiment.

## Browser-based inference

`web/index.html` loads TensorFlow.js and then calls:

```javascript
const model = await tf.loadLayersModel('./tfjs_model/model.json');
```

The `model.json` topology references 24 sharded binary weight files. The page previews the chosen image, creates the preprocessed tensor inside `tf.tidy`, runs `model.predict`, and displays the top three classes. Uploaded image pixels stay in the browser application.

## Evaluation

`scripts/evaluate_model.py` supports:

- accuracy and top-5 accuracy;
- macro and weighted precision, recall, and F1;
- per-class classification report;
- confusion matrix;
- one-vs-rest ROC-AUC where computable;
- sample prediction export;
- low-confidence and high-confidence error tables.

Accuracy alone can hide weak classes. Macro F1 gives every class equal importance, weighted F1 reflects class frequency, and per-class recall shows which categories the model fails to capture.

## Error analysis

The notebook already explores correct predictions, misclassifications, confident correct predictions, confident wrong predictions, and per-class accuracy. Common failure causes for CIFAR-100 include 32×32 source resolution, visually similar categories, small objects, background ambiguity, and distribution mismatch when a user uploads a normal high-resolution photograph.

## Folder structure

```text
04-image-classification-resnet/
├── app/                         # Reserved optional Python UI package
├── archive/                     # Superseded experiments and notes
├── data/
│   ├── README_data.md
│   └── sample_images/
├── images/                      # Portfolio screenshots
├── models/
│   ├── resnet50_cifar100.keras  # Supplied model; use LFS/release asset
│   ├── model_metadata.json
│   ├── class_mapping.json
│   └── tfjs_model/              # Reference and conversion notes
├── notebooks/
│   └── image_classification_resnet.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── scripts/
├── src/
├── tests/
├── web/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── metadata.json
│   ├── sample_images/
│   └── tfjs_model/
├── README.md
├── README_GITHUB_PAGES.md
├── requirements.txt
└── requirements-ci.txt
```

## Local setup

```bash
cd cnn-projects/04-image-classification-resnet
python -m venv .venv
```

Windows:

```powershell
.venv\Scriptsctivate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run training only when needed:

```bash
python scripts/train_model.py
```

Run evaluation:

```bash
python scripts/evaluate_model.py --model models/resnet50_cifar100.keras
```

Create the flattened browser-inference model, then convert it with the official converter:

```bash
python scripts/export_model.py --keras-model models/resnet50_cifar100.keras --output models/resnet50_cifar100_browser.h5
python scripts/convert_to_tfjs.py --keras-model models/resnet50_cifar100_browser.h5 --output web/tfjs_model
```

Serve the browser app through HTTP rather than opening `index.html` directly:

```bash
python scripts/run_local_web_server.py --port 8000
```

Open `http://localhost:8000`.

## GitHub Pages deployment

The root workflow `.github/workflows/04-image-classification-resnet.yml` validates the project and publishes `04-image-classification-resnet/web` with GitHub's Pages artifact workflow. In the repository, open **Settings → Pages → Build and deployment → Source**, select **GitHub Actions**, then push to `main`.

See [`README_GITHUB_PAGES.md`](./README_GITHUB_PAGES.md) for model-size, caching, CORS, and troubleshooting notes.

## Continuous integration

The CI job intentionally does not retrain ResNet. It:

- installs lightweight validation dependencies;
- compiles Python files;
- runs unit tests;
- validates metadata and class mapping;
- checks `index.html`, `app.js`, `style.css`, and `model.json`;
- verifies every weight shard referenced by the manifest exists and is non-empty.

## Artifact handling

The supplied `.keras` model is about 103 MB and should not be committed through normal Git history. Keep it local, track it with Git LFS, or attach it to a GitHub release. The browser model is split into smaller shards under `web/tfjs_model/` so the static deployment can fetch and cache them individually.

## Suggested screenshots

1. Browser demo immediately after model load.
2. Uploaded image preview with predicted class and confidence.
3. Top-three probability bars.
4. Baseline-versus-ResNet metrics chart.
5. Notebook training curves.
6. Confusion matrix and representative errors generated from real evaluation.
7. GitHub Actions validation and Pages deployment success.

## Skills demonstrated

CNN modeling · residual learning · transfer learning · image preprocessing · multiclass classification · top-k inference · evaluation design · error analysis · model serialization · TensorFlow.js conversion · JavaScript inference · static web deployment · GitHub Actions · responsible AI communication

## Quality-data-science relevance

The same workflow patterns transfer naturally to automated visual inspection, product family classification, defect triage, inspection-image review, and image-based quality analytics: define controlled labels, maintain preprocessing parity, evaluate class-level failure modes, package reproducible inference, and communicate model limits.

## Future improvements

- Fine-tune the final ResNet stage at a low learning rate.
- Add calibrated confidence and out-of-distribution warnings.
- Quantize the browser model after measuring accuracy impact.
- Add real confusion-matrix and per-class report artifacts from a reproducible evaluation run.
- Add optional Grad-CAM to the Python evaluation workflow.
- Benchmark a lightweight browser model such as MobileNetV2 while retaining ResNet50 as the primary portfolio experiment.

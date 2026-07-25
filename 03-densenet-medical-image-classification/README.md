---
title: DenseNet Medical Image Classification
emoji: 🩺
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# DenseNet Medical Image Classification

[![CI](https://github.com/unit-mole/cnn-projects/actions/workflows/03-densenet-medical-image-classification.yml/badge.svg)](https://github.com/unit-mole/cnn-projects/actions/workflows/03-densenet-medical-image-classification.yml)

A portfolio-ready DenseNet121 image-classification project with modular Python code, an interactive
Gradio demo, Hugging Face Spaces deployment files, a Kaggle training notebook, model evaluation,
error-analysis outputs, optional Grad-CAM, tests, Docker, and GitHub Actions.

> [!WARNING]
> **Artifact audit:** the attached executable notebook trained the bundled model on a Fashion-MNIST-derived
> synthetic binary proxy, not on chest X-rays. The class labels are therefore `normal_like` and
> `pneumonia_like`. The included metrics are retained as proof of the attached experiment, but they are
> not clinical pneumonia-detection results. Use the Kaggle notebook to train a properly documented
> chest-X-ray model before changing this disclosure.

## Medical disclaimer

This project is for educational and portfolio demonstration purposes only. It is not a medical diagnostic tool. The model must not be used to diagnose, treat, prevent, or manage any medical condition. Medical image interpretation requires clinical validation, domain expertise, and review by qualified healthcare professionals. Do not upload private, sensitive, confidential, or personally identifiable medical images. Predictions are machine-learning outputs, not medical advice.

## Live links

- **Live demo:** Hugging Face Spaces link to be added
- **Training notebook:** Kaggle Notebook link to be added
- **Main repository:** https://github.com/unit-mole/cnn-projects

## Project objective

The project demonstrates the end-to-end engineering pattern for binary medical-image classification:
image validation, consistent preprocessing, DenseNet transfer learning, class-probability output,
metrics beyond accuracy, error analysis, optional visual explanation, artifact export, and web deployment.

The current bundled model is best understood as a **software and architecture prototype**. The separate
Kaggle notebook provides the path for retraining the same project structure on a real, licensed,
folder-based chest-X-ray dataset.

## Actual attached experiment

| Item | Recorded configuration |
|---|---|
| Executed dataset | Fashion-MNIST-derived synthetic proxy |
| Proxy classes | `normal_like`, `pneumonia_like` |
| Proxy positive source classes | Fashion-MNIST classes 2, 4, and 6 |
| Train / validation / test | 52,000 / 8,000 / 10,000 |
| Input to saved model | 28×28×3, scaled to `[0, 1]` |
| Internal resize | 96×96 |
| Backbone | DenseNet121, ImageNet weights, frozen |
| Head | GAP → Dense(256) → BatchNorm → Dropout(0.5) → Softmax(2) |
| Optimizer / loss | Adam / categorical cross-entropy |
| Requested epochs / batch | 12 / 128 |

## Recorded results — synthetic proxy only

| Model | Validation accuracy | Test accuracy | Test ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression baseline | 0.9313 | 0.9282 | 0.9740 |
| DenseNet121 | 0.9599 | 0.9564 | 0.9934 |

DenseNet proxy-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal_like | 0.9694 | 0.9683 | 0.9688 | 7,000 |
| pneumonia_like | 0.9262 | 0.9287 | 0.9274 | 3,000 |
| Macro average | 0.9478 | 0.9485 | 0.9481 | 10,000 |
| Weighted average | 0.9564 | 0.9564 | 0.9564 | 10,000 |

## Why DenseNet?

DenseNet is a convolutional neural network in which layers receive feature maps from earlier layers.
These dense connections encourage feature reuse, improve gradient flow, and make the architecture a
strong transfer-learning backbone. In medical-imaging research, DenseNet can be effective when paired
with appropriate data, careful validation, and responsible domain review.

## Workflow

```text
Image input
   ↓
File/type validation and EXIF-safe loading
   ↓
RGB conversion → resize → model-specific normalization
   ↓
DenseNet121 feature extraction
   ↓
Global average pooling and classification head
   ↓
Class probabilities, confidence, interpretation, optional Grad-CAM
   ↓
Gradio / Hugging Face Spaces demo
```

## Evaluation philosophy

Accuracy alone is not sufficient. The real-data Kaggle workflow includes precision, recall, F1,
macro F1, weighted F1, confusion matrix, ROC-AUC, precision-recall AUC, threshold analysis, false-positive
and false-negative review, low-confidence cases, and optional Grad-CAM. Recall is particularly important
in screening-style experiments because false negatives can be consequential, but no portfolio metric
establishes clinical safety.

## Run locally

```bash
git clone https://github.com/unit-mole/cnn-projects.git
cd cnn-projects/03-densenet-medical-image-classification
python -m venv .venv
```

Windows:

```powershell
.venv\Scriptsctivate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL shown in the terminal.

## Train on a real folder-based dataset

```bash
python scripts/train_model.py --dataset /path/to/chest_xray
python scripts/evaluate_model.py --dataset /path/to/chest_xray
```

The preferred reproducible path is the Kaggle notebook because it includes EDA, class-distribution
review, plots, metrics, error analysis, Grad-CAM, and artifact export in one documented workflow.

## Hugging Face Spaces deployment

1. Create a new Space and select the Gradio SDK.
2. Copy `app.py`, `gradio_app.py`, `requirements.txt`, `README.md`, `models/`, `src/`, and `data/sample_images/`.
3. Preserve the README YAML metadata at the top of this file.
4. Push or upload the files and review the build logs.
5. Test the interface and add the final Space link above.

See [`README_HUGGINGFACE.md`](README_HUGGINGFACE.md) for the full guide.

## Kaggle reproduction

Upload [`notebooks/densenet_medical_image_classification_kaggle.ipynb`](notebooks/densenet_medical_image_classification_kaggle.ipynb),
attach a licensed dataset, enable an available accelerator, run all cells, and download the exported
model and metadata. See [`kaggle/README_KAGGLE.md`](kaggle/README_KAGGLE.md).

## Folder structure

```text
03-densenet-medical-image-classification/
├── app.py
├── gradio_app.py
├── README.md
├── README_HUGGINGFACE.md
├── PROJECT_AUDIT.md
├── IMPROVEMENTS.md
├── data/
├── images/
├── kaggle/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── requirements.txt
├── requirements-ci.txt
├── requirements-dev.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```

## Skills demonstrated

DenseNet121 · CNNs · transfer learning · image preprocessing · dataset auditing · class imbalance ·
classification metrics · threshold analysis · error analysis · Grad-CAM · Gradio · Hugging Face Spaces ·
Kaggle Notebooks · Docker · testing · GitHub Actions · responsible medical-AI communication

## Portfolio positioning

**One-line description:**
> Built an audited DenseNet121 image-classification pipeline with reusable preprocessing, evaluation,
> explainability hooks, Kaggle training, and a Hugging Face / Gradio deployment workflow.

This project connects naturally to quality-data-science work through visual inspection, defect
classification, automated quality checks, image-based anomaly review, and production-oriented model validation.

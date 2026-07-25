---
title: CNN Handwritten Digit Object Detector
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 02 — Object Detection Using CNN

[![CI](https://github.com/unit-mole/cnn-projects/actions/workflows/02-object-detection-using-cnn.yml/badge.svg)](https://github.com/unit-mole/cnn-projects/actions)
[![Hugging Face](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](#live-demo)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](#technology-stack)

A compact end-to-end computer-vision project that predicts **which handwritten
digit is present and where it is located**. The supplied model is a custom
multi-output CNN with a classification head and a bounding-box regression head.

> **Responsible use:** This project is for education and portfolio
> demonstration only. The model can miss objects, predict an incorrect class,
> or produce an inaccurate box. It must not be used as the sole basis for
> surveillance, safety-critical monitoring, autonomous driving, medical,
> security, legal, or production inspection decisions. Do not upload private,
> confidential, copyrighted, or personally identifiable images to a public
> demo.

## What the original project actually does

This is a **single-object handwritten-digit detector**, not a general-purpose
multi-object detector.

- Source data: MNIST grayscale digits.
- Generated input: one randomly resized digit on a 64×64 black canvas.
- Classes: digits `0` through `9`.
- Annotation: one normalized `XYXY` bounding box per image.
- Model outputs:
  1. a 10-class softmax distribution;
  2. one normalized bounding box `[x1, y1, x2, y2]`.
- NMS: not applicable because the model predicts exactly one box.
- External VOC, COCO, or YOLO files: not used; annotations are generated in memory.

## Results from the supplied artifacts

| Metric | Baseline | CNN detector |
|---|---:|---:|
| Test class accuracy | 0.177 | **0.932** |
| Mean test IoU | **0.368** | 0.349 |
| Top-3 accuracy | — | **0.989** |

The CNN substantially improves class recognition, but its mean IoU is slightly
below the fixed center-box baseline. That result is intentionally reported
without overstatement: localization is the main weakness and the best future
improvement target.

## Architecture

```text
64×64×1 input
→ Conv2D(32) + MaxPool
→ Conv2D(64) + MaxPool
→ Conv2D(128)
→ GlobalAveragePooling
→ Dense(128) + Dropout(0.4)
├── class_output: Dense(10, softmax)
└── box_output: Dense(4, sigmoid)
```

Training uses categorical cross-entropy for digit classification and mean
squared error for box regression.

## Why this is object detection

Image classification answers **what** is present. Object detection answers both
**what** is present and **where** it is located. The shared CNN backbone learns
visual features, the classification head predicts the digit, and the regression
head predicts the bounding-box coordinates.

## Repository structure

```text
02-object-detection-using-cnn/
├── app.py
├── gradio_app.py
├── data/
├── notebooks/
├── src/
├── models/
├── outputs/
├── images/
├── tests/
├── scripts/
├── requirements.txt
├── requirements-dev.txt
├── README_HUGGINGFACE.md
├── Dockerfile
├── .dockerignore
└── .gitignore
```

## Run locally

```bash
cd cnn-projects/02-object-detection-using-cnn
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
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

Open the local Gradio URL printed in the terminal.

## Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Hugging Face Spaces deployment

1. Create a new **Gradio** Space.
2. Copy the contents of this project folder to the root of the Space repository.
3. Keep `app.py`, `requirements.txt`, `README.md`, `src/`, and `models/` at the Space root.
4. Confirm that `models/cnn_detector.keras` and `models/model_metadata.json` are present.
5. Push the files. The Space starts from `app.py`; it does not retrain the model.
6. After the build succeeds, place the live Space URL in this README and the root repository README.

The included model is about 1.4 MB, so ordinary Git is sufficient. Git LFS is
not required for this artifact.

## Docker

```bash
docker build -t cnn-digit-detector .
docker run --rm -p 7860:7860 cnn-digit-detector
```

## Error analysis

Good cases generally contain one bright digit with clear contrast on a dark
background. Weak cases include thin strokes, unusual placement, excessive
background content, photographs, multiple digits, or inputs far from the
synthetic MNIST distribution. A high classification confidence does not
guarantee a well-localized box.

## Future improvements

- Replace MSE with a localization-aware loss such as IoU/GIoU/DIoU.
- Add coordinate-aware spatial features instead of relying only on global pooling.
- Train with broader translations, scales, rotations, and contrast conditions.
- Report IoU at multiple thresholds and class-aware average precision.
- Extend the target representation to multiple boxes and objectness scores.
- Compare against a lightweight pretrained detector.

## Portfolio positioning

**One-line description:** Built and deployed a custom CNN that jointly
classifies handwritten digits and predicts their bounding boxes through a
Gradio interface on Hugging Face Spaces.

**Skills demonstrated:** TensorFlow/Keras, CNN modeling, bounding-box
regression, synthetic detection-data generation, image preprocessing, IoU
evaluation, Gradio, Hugging Face Spaces, testing, Docker, CI, and responsible
AI communication.

This project also connects naturally to visual inspection and quality analytics:
the same pattern—classifying an item and localizing the region of interest—is a
foundation for defect localization and image-based inspection systems.

---
title: Satellite Image Segmentation U-Net
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.5.1
python_version: 3.11
app_file: app.py
pinned: false
license: mit
short_description: U-Net binary segmentation demo for synthetic satellite-style urban tiles.
---

# Satellite Image Segmentation using U-Net

[![CI](https://github.com/YOUR_USERNAME/cnn-projects/actions/workflows/06-satellite-image-segmentation-unet.yml/badge.svg)](https://github.com/YOUR_USERNAME/cnn-projects/actions/workflows/06-satellite-image-segmentation-unet.yml)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/satellite-image-segmentation-unet)
[![Kaggle](https://img.shields.io/badge/Kaggle-Training%20Notebook-blue)](https://www.kaggle.com/code/YOUR_USERNAME/satellite-image-segmentation-unet)

A complete CNN portfolio project that trains, evaluates, packages, tests, and deploys a compact **U-Net** for binary semantic segmentation of synthetic satellite-style urban tiles.

> **Portfolio one-liner:** Built and deployed an end-to-end U-Net semantic-segmentation pipeline with reproducible training, Dice/IoU evaluation, visual error analysis, a Gradio app on Hugging Face Spaces, and a Kaggle-ready notebook.

## Responsible-use notice

This project is for educational and portfolio demonstration purposes only. The supplied model was trained on procedurally generated 64×64 tiles—not operational satellite imagery. Satellite segmentation models may produce inaccurate masks because of cloud cover, shadows, resolution differences, seasonal variation, sensor differences, annotation noise, or out-of-distribution imagery.

Do not use this model as the sole basis for emergency response, environmental enforcement, land ownership, military, legal, financial, agricultural, infrastructure, or public-policy decisions. Do not upload private, restricted, confidential, copyrighted, or sensitive geospatial imagery to a public demo. Predicted masks are machine-learning outputs, not official geospatial analysis.

## Live project links

- **Hugging Face Spaces demo:** [Replace with your public Space URL](https://huggingface.co/spaces/YOUR_USERNAME/satellite-image-segmentation-unet)
- **Kaggle training notebook:** [Replace with your public Kaggle URL](https://www.kaggle.com/code/YOUR_USERNAME/satellite-image-segmentation-unet)
- **CNN monorepo:** [Replace with your GitHub URL](https://github.com/YOUR_USERNAME/cnn-projects)

## Problem statement

Given an RGB satellite-style image, predict a pixel-level binary mask that highlights the target urban-structure region. The deployed app returns:

1. input image,
2. predicted binary mask,
3. color overlay,
4. probability map,
5. model metadata and reported metrics,
6. optional Dice, IoU, precision, recall, and F1 when a ground-truth mask is provided,
7. downloadable PNG mask.

## What the supplied project actually does

The original notebook procedurally generates 2,500 satellite-style RGB tiles. Each binary mask contains two to five rectangular building-like regions, and pixels inside the mask are made brighter. It uses a deterministic seed of 42 and produces the following split:

| Split | Samples |
|---|---:|
| Training | 1,750 |
| Validation | 375 |
| Test | 375 |
| **Total** | **2,500** |

| Data property | Value |
|---|---|
| Input | 64×64×3 RGB, float32 |
| Target | 64×64×1 binary mask |
| Classes | Background; synthetic urban structure |
| Positive-pixel rate | Approximately 10.6% in training |
| Normalization | Pixel values in `[0, 1]` |
| Image resizing | Bilinear |
| Mask resizing | Nearest-neighbor |
| Inference threshold | 0.50 |

The repository includes six safe 512×512 nearest-neighbor-upscaled sample pairs for the interactive demo. They contain no real coordinates or sensitive geospatial metadata.

## U-Net architecture

```text
RGB image (64×64×3)
       │
       ▼
Encoder block: 2× Conv(32) ──────────────┐
       │ MaxPool                         │ skip connection
       ▼                                 │
Encoder block: 2× Conv(64) ────────┐     │
       │ MaxPool                   │     │
       ▼                           │     │
Bottleneck: 2× Conv(128)           │     │
       │ Upsample                  │     │
       ├──── concatenate Conv(64) ◄┘     │
       │ Upsample                        │
       ├──── concatenate Conv(32) ◄──────┘
       ▼
1×1 Conv + sigmoid → binary probability mask
```

U-Net is a CNN architecture designed for segmentation. The encoder learns high-level visual features, the decoder reconstructs a pixel-level mask, and skip connections transfer fine spatial details from encoder layers to matching decoder layers. This is useful when boundaries and small regions matter.

**Saved-model summary:** 471,553 parameters; binary cross-entropy loss; Adam optimizer; Dice and IoU training metrics; Keras v3 `.keras` artifact.

## Preprocessing and inference

The same core preprocessing rules are used across training and inference:

- decode and convert input to RGB,
- apply EXIF orientation safely,
- resize imagery to 64×64 with bilinear interpolation,
- normalize 8-bit values to float32 `[0, 1]`,
- retain mask labels with nearest-neighbor interpolation,
- run one batch through the pre-trained U-Net,
- threshold the sigmoid probability at 0.50,
- resize the binary mask back to the uploaded image size with nearest-neighbor interpolation,
- create a red mask overlay and blue-to-red probability map.

The app performs **inference only** at startup. It never trains the model when a user opens the Space.

## Evaluation results

These are the values exported by the supplied notebook:

| Approach / metric | Value |
|---|---:|
| Threshold baseline Dice | 0.999878 |
| Threshold baseline IoU | 0.999756 |
| U-Net test Dice | 0.999849 |
| U-Net test IoU | 0.999698 |
| Pixel accuracy | 0.999992 |

### Honest interpretation

The threshold baseline is marginally better than the U-Net. This is expected because the synthetic generator brightens the exact target rectangles, making the task nearly solvable with a fixed intensity threshold. The very high numbers verify that the implementation works end to end; they do **not** demonstrate generalization to real satellite imagery.

Pixel accuracy is also inflated by background dominance. Dice and IoU are more informative for overlap, but even those are optimistic on this simple synthetic distribution.

## Visual results

| Training / evaluation | Examples |
|---|---|
| ![Dice curve](outputs/figures/dice_training_curve.png) | ![Best predictions](outputs/figures/best_segmentation_examples.png) |
| ![Loss curve](outputs/figures/loss_training_curve.png) | ![Error examples](outputs/figures/segmentation_error_examples.png) |
| ![Threshold sweep](outputs/figures/threshold_sweep.png) | ![Overlays](outputs/figures/overlay_predictions.png) |

Additional figures are under [`outputs/figures`](outputs/figures/).

## Error analysis

The notebook saves the strongest and weakest test examples, overlap distributions, overlays, a threshold sweep, and predicted-versus-true area analysis. Remaining errors are tiny on this benchmark and primarily occur around synthetic rectangle boundaries.

For real imagery, likely error sources would include:

- cloud and shadow confusion,
- low-resolution or blurred boundaries,
- small-object omission,
- false positives from bright roofs, soil, or reflective surfaces,
- seasonal or sensor shifts,
- annotation misalignment,
- train/test geographic leakage.

## Gradio demo

Run locally:

```bash
cd 06-satellite-image-segmentation-unet
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL displayed in the terminal. Select a safe example or upload an RGB image. An optional ground-truth mask enables per-upload overlap metrics.

## Reproduce training

### Local

```bash
pip install -r requirements-training.txt
python train_model.py --samples 2500 --epochs 15 --batch-size 32 --seed 42
python scripts/evaluate_model.py
```

Training writes the model to `models/satellite_unet_segmentation_model.keras` and evaluation artifacts to `outputs/metrics/`.

### Kaggle

Upload [`notebooks/satellite_image_segmentation_unet_kaggle.ipynb`](notebooks/satellite_image_segmentation_unet_kaggle.ipynb), choose a Python notebook image, optionally enable GPU, and run all cells. No external dataset is required for the supplied synthetic benchmark. Follow [`kaggle/README_KAGGLE.md`](kaggle/README_KAGGLE.md) for publishing and artifact-transfer steps.

## Hugging Face Spaces deployment

The project root is already structured as a Gradio Space:

```text
app.py
requirements.txt
README.md            # contains Space YAML metadata
models/
src/
data/sample_images/
```

Create a new Gradio Space, clone it, copy the project contents into the Space root, replace `YOUR_USERNAME` placeholders, commit, and push. Full instructions are in [`README_HUGGINGFACE.md`](README_HUGGINGFACE.md).

## Docker

```bash
docker build -t satellite-unet-demo .
docker run --rm -p 7860:7860 satellite-unet-demo
```

## Tests and CI

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
python scripts/validate_project.py
```

The repository-level workflow is correctly stored at:

```text
cnn-projects/.github/workflows/06-satellite-image-segmentation-unet.yml
```

It validates files, runs unit tests, imports the Gradio and inference modules, and performs one pre-trained-model smoke inference. It does not retrain the U-Net.

## Project structure

```text
06-satellite-image-segmentation-unet/
├── .streamlit/                 # portfolio-layout compatibility; not used for deployment
├── app/
├── archive/
├── data/
│   ├── sample_images/
│   ├── sample_masks/
│   ├── README_data.md
│   └── sample_manifest.csv
├── images/
├── kaggle/
├── models/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── predictions/
│   └── runtime/
├── scripts/
├── src/
├── tests/
├── app.py
├── gradio_app.py
├── train_model.py
├── Dockerfile
├── FILE_MANIFEST.csv
├── IMPROVEMENTS.md
├── LICENSE
├── MONOREPO_INTEGRATION.md
├── PROJECT_AUDIT.md
├── README.md
├── README_HOSTING.md
├── README_HUGGINGFACE.md
├── requirements.txt
├── requirements-training.txt
├── requirements-dev.txt
├── run_local.bat
└── run_local.sh
```

## Portfolio positioning

### Pinned-repository description

> Production-style CNN portfolio with seven computer-vision projects, including a deployable U-Net satellite segmentation demo, reproducible Kaggle training, evaluation artifacts, CI, tests, and responsible-AI documentation.

### Skills demonstrated by this project

- CNN and U-Net architecture
- binary semantic segmentation and pixel-level prediction
- image and mask preprocessing
- synchronized spatial augmentation
- Dice, IoU, precision, recall, F1, and visual evaluation
- Keras model serialization and lazy inference loading
- Gradio application development
- Hugging Face Spaces deployment
- Kaggle notebook reproducibility
- tests, GitHub Actions, Docker, and artifact governance

### Connection to Quality Data Science

Semantic segmentation maps naturally to visual inspection and quality workflows: identifying regions of interest, localizing defects, measuring affected area, automating image-based checks, and tracking spatial patterns. This project demonstrates how quality analytics can extend from tabular metrics into applied computer vision.

## Future improvements

1. Replace procedural tiles with a licensed real benchmark.
2. Use geographic scene-level splits and cross-region evaluation.
3. Add multi-class masks for buildings, roads, water, and vegetation.
4. Add multispectral band support and documented band selection.
5. Compare compact U-Net with U-Net++, DeepLabV3+, and a pretrained encoder.
6. Add boundary IoU, calibration, uncertainty, and robustness tests.
7. Quantize or export the model for faster CPU inference.

## License

MIT. Dataset licensing for any future real-data extension must be documented separately.

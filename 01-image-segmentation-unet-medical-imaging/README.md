# Medical Image Segmentation using U-Net

[![CI](https://github.com/unit-mole/cnn-projects/actions/workflows/01-image-segmentation-unet-medical-imaging.yml/badge.svg)](https://github.com/unit-mole/cnn-projects/actions/workflows/01-image-segmentation-unet-medical-imaging.yml)
[![Vercel](https://img.shields.io/badge/Vercel-Live%20Browser%20Demo-black)](ADD_VERCEL_URL)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22.0-orange)](https://www.tensorflow.org/js)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **binary medical-image-segmentation proof of concept** using a compact U-Net, deterministic synthetic MRI-style data, Dice/IoU evaluation, visual error analysis, modular Python code, and a static **Vercel + TensorFlow.js** browser demo.

> **Portfolio one-liner:** Built and deployed a compact U-Net segmentation pipeline with reproducible synthetic-data generation, Dice/IoU evaluation, Keras-to-browser weight export, and client-side TensorFlow.js inference on Vercel.

## Medical disclaimer

This project is for educational and portfolio demonstration purposes only. It is **not a medical diagnostic tool**. The model must not be used to diagnose, treat, prevent, or manage any medical condition. Medical-image interpretation requires clinical validation, domain expertise, and review by qualified healthcare professionals. Do not upload private, sensitive, confidential, or personally identifiable medical images. Predicted masks are machine-learning outputs, not medical advice.

## Live demo

- **Vercel + TensorFlow.js:** [Replace with the production Vercel URL](ADD_VERCEL_URL)
- **GitHub repository:** [unit-mole/cnn-projects](https://github.com/unit-mole/cnn-projects)
- **Project source:** [`01-image-segmentation-unet-medical-imaging`](https://github.com/unit-mole/cnn-projects/tree/main/01-image-segmentation-unet-medical-imaging)

The Vercel site performs inference directly in the visitor's browser. The uploaded image is not sent to a Python application server.

## What the supplied notebook actually does

The notebook generates **2,500 deterministic synthetic MRI-style grayscale image-mask pairs**. Each 64×64 image contains noise plus a brighter elliptical region; the binary mask identifies that ellipse. It uses seed 42 and a 70/15/15 split:

| Split | Samples |
|---|---:|
| Training | 1,750 |
| Validation | 375 |
| Test | 375 |
| **Total** | **2,500** |

No real Kaggle, BraTS, ISIC, hospital, or patient dataset is loaded by the executable notebook. Therefore, the model is presented as a technical segmentation proof of concept—not as a clinically validated medical model.

## Model architecture

```text
Input: 64 × 64 × 1
        ↓
Conv 32 → Conv 32 → MaxPool
        ↓                    ↘ skip
Conv 64 → Conv 64 → MaxPool
        ↓                    ↘ skip
Conv 128 → Conv 128
        ↓
Upsample → Concatenate skip 64 → Conv 64 → Conv 64
        ↓
Upsample → Concatenate skip 32 → Conv 32 → Conv 32
        ↓
1 × 1 Conv + Sigmoid
        ↓
64 × 64 binary probability mask
```

- Parameters: **470,977**
- Optimizer: Adam, learning rate 0.001
- Loss: binary cross-entropy
- Training: 15 epochs, batch size 32
- Metrics: soft Dice, soft IoU, pixel accuracy

## Recorded results

| Approach | Test Dice | Test IoU |
|---|---:|---:|
| Intensity-threshold baseline | 0.9659 | 0.9360 |
| Compact U-Net | **0.9977** | **0.9954** |

Pixel accuracy at threshold 0.5: **0.99984**.

These very high values reflect the simple deterministic synthetic generator. They must not be interpreted as expected performance on clinical images.

## Vercel browser application

The static website provides:

- image upload and safe bundled examples;
- 64×64 grayscale preprocessing in JavaScript;
- client-side U-Net inference with TensorFlow.js;
- probability heatmap;
- adjustable-threshold binary mask;
- mask overlay;
- optional reference-mask Dice and IoU;
- downloadable PNG mask;
- visible medical disclaimer and limitations.

```text
web/
├── index.html
├── style.css
├── app.js
├── metadata.json
├── sample_images/
├── sample_masks/
└── tfjs_model/
    ├── model.json
    ├── weights_manifest.json
    ├── weights.bin
    └── model_metadata.json
```

The browser app reconstructs the exact compact U-Net topology in TensorFlow.js and loads the 22 Conv2D tensors exported from the supplied Keras 3 archive. No training occurs in the browser or on Vercel.

## Run the Vercel website locally

Windows:

```bat
run_vercel_local.bat
```

Manual alternative:

```bash
node scripts/validate-web.mjs
python -m http.server 8000 --directory web
```

Open `http://127.0.0.1:8000` and wait for **Model ready**.

## Run the Python reference application locally

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

The Python/Gradio app is retained as a local reference implementation. Vercel does not execute it.

## Re-export browser weights

```bash
python -m pip install h5py numpy
python scripts/export_tfjs_assets.py
python scripts/validate_tfjs_export.py
npm run build
```

The exporter does not require TensorFlow. It extracts the trained Conv2D kernels and biases from `models/unet_medical.keras` and produces a deterministic little-endian float32 bundle.

## Evaluation interpretation

- **Dice coefficient** measures overlap between predicted and reference masks.
- **IoU/Jaccard** measures intersection divided by union.
- **Pixel accuracy** can be misleading when background pixels dominate.
- **Overlays and probability maps** provide essential visual checks beyond numerical metrics.

## Limitations

- Trained only on synthetic grayscale images, not real patient scans.
- Target masks are simple elliptical regions and do not represent anatomy, lesions, or tumors.
- Performance on MRI, CT, X-ray, ultrasound, pathology, or dermoscopy is unknown.
- Ordinary PNG/JPG/WebP files are supported; DICOM parsing is not included.
- No clinical validation, regulatory review, fairness assessment, or prospective evaluation has been performed.
- Browser predictions depend on the visitor's device and TensorFlow.js backend.

## Project structure

```text
01-image-segmentation-unet-medical-imaging/
├── web/                       # Vercel static TensorFlow.js site
├── models/                    # Keras model, metadata, metrics
├── src/                       # Modular Python pipelines
├── scripts/                   # Training, evaluation, export, validation
├── tests/                     # Python and browser-bundle tests
├── notebooks/                 # Original/reorganized experiment
├── data/                      # Safe synthetic demo examples
├── outputs/                   # Metrics, plots, and error analysis
├── package.json               # Static build validation
├── vercel.json                # Vercel deployment configuration
├── README_VERCEL.md           # Exact deployment guide
├── app.py                     # Optional local Gradio reference
└── README.md
```

## Skills demonstrated

U-Net · convolutional neural networks · semantic segmentation · image/mask preprocessing · skip connections · Dice/IoU evaluation · TensorFlow/Keras · TensorFlow.js · JavaScript · browser ML · Vercel · model artifact conversion · GitHub Actions · responsible AI communication.

## Connection to quality analytics

Pixel-level segmentation is directly relevant to defect localization, automated visual inspection, surface-area measurement, anomaly-region highlighting, inspection analytics, and image-based quality workflows. This project connects computer-vision engineering with my current work as a Quality Data Scientist.

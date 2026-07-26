# DenseNet121 Medical Image Classification

[![Project 03 CI and Pages](https://github.com/unit-mole/cnn-projects/actions/workflows/03-densenet-medical-image-classification.yml/badge.svg)](https://github.com/unit-mole/cnn-projects/actions/workflows/03-densenet-medical-image-classification.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-2ea44f)](https://unit-mole.github.io/cnn-projects/03-densenet-medical-image-classification/)
[![TensorFlow.js](https://img.shields.io/badge/Inference-TensorFlow.js-ff6f00)](https://www.tensorflow.org/js)

A portfolio-ready DenseNet121 image-classification project with a browser-based TensorFlow.js demo, GitHub Pages deployment, modular Python code, model evaluation, error-analysis outputs, Kaggle training workflow, tests, Docker support, and responsible AI documentation.

## Live project

- **GitHub Pages demo:** https://unit-mole.github.io/cnn-projects/03-densenet-medical-image-classification/
- **Source repository:** https://github.com/unit-mole/cnn-projects
- **Project source:** https://github.com/unit-mole/cnn-projects/tree/main/03-densenet-medical-image-classification
- **Kaggle notebook:** add the public notebook URL after publishing

## Important artifact audit

The attached executed notebook describes pneumonia detection, but its executable cells actually train the bundled model on a **Fashion-MNIST-derived synthetic binary proxy**. The public demo therefore uses the labels:

```text
normal_like
pneumonia_like
```

The bundled scores are proof of the proxy experiment and must not be presented as clinical pneumonia-detection performance. The separate Kaggle notebook provides the correct workflow for retraining on a licensed and properly documented chest-X-ray dataset.

> **Medical disclaimer:** This project is for education and portfolio demonstration only. It is not a medical diagnostic tool. Do not use its output to diagnose, treat, prevent, or manage any medical condition. Do not upload private, sensitive, confidential, or personally identifiable medical images. Predictions are machine-learning outputs, not medical advice.

## GitHub Pages + TensorFlow.js deployment

The public site is a static browser application stored in [`web/`](web/). No Python server is required after deployment.

The workflow performs the following steps:

1. Runs project validation and unit tests.
2. Validates the static HTML, CSS, JavaScript, metadata, and browser-model artifact.
3. Converts [`models/densenet121_medical_browser.h5`](models/densenet121_medical_browser.h5) to TensorFlow.js LayersModel format.
4. Applies two-byte weight quantization to reduce browser download size.
5. Publishes only the Project 03 web folder into the existing `gh-pages` branch at:

```text
03-densenet-medical-image-classification/
```

6. Preserves the existing root website and other deployed project folders.

The final URL is:

```text
https://unit-mole.github.io/cnn-projects/03-densenet-medical-image-classification/
```

Detailed instructions are available in [`README_GITHUB_PAGES.md`](README_GITHUB_PAGES.md).

## Browser inference pipeline

```text
Uploaded image
    ↓
Private browser-only processing
    ↓
RGB conversion and aspect-ratio-safe 28×28 compatibility resize
    ↓
96×96 TensorFlow.js tensor resize
    ↓
ImageNet/DenseNet channel normalization
    ↓
DenseNet121 frozen feature extractor
    ↓
Global Average Pooling
    ↓
Dense(256) → Batch Normalization → Dropout
    ↓
Dense(2, softmax)
    ↓
Proxy class, confidence, probability bars, and downloadable JSON
```

## Model architecture

| Component | Configuration |
|---|---|
| Input used by browser model | `96 × 96 × 3` RGB tensor |
| Backbone | DenseNet121 with ImageNet initialization |
| Backbone state | Frozen in the attached experiment |
| Feature aggregation | GlobalAveragePooling2D |
| Classification head | Dense(256, ReLU) → BatchNorm → Dropout(0.5) |
| Output | Dense(2, Softmax) |
| Browser runtime | TensorFlow.js WebGL with CPU fallback |
| Source model format | Keras HDF5 inference model |
| Published format | TensorFlow.js LayersModel |

DenseNet connects each layer to earlier feature maps. These dense connections improve feature reuse and gradient flow, which makes DenseNet a strong transfer-learning architecture for image-classification experiments.

## Recorded proxy results

| Model | Validation accuracy | Test accuracy | Test ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression baseline | 0.9313 | 0.9282 | 0.9740 |
| DenseNet121 | 0.9599 | 0.9564 | 0.9934 |

Additional DenseNet proxy metrics:

| Metric | Value |
|---|---:|
| Macro F1 | 0.9481 |
| Weighted F1 | 0.9564 |
| `normal_like` recall | 0.9683 |
| `pneumonia_like` recall | 0.9287 |

These values apply only to the synthetic proxy dataset used by the attached notebook.

## Project structure

```text
03-densenet-medical-image-classification/
├── app.py                              # Optional local Gradio app
├── gradio_app.py                       # Gradio interface implementation
├── data/
├── images/
├── kaggle/
├── models/
│   ├── densenet121_medical.keras       # Original audited model
│   ├── densenet121_medical_browser.h5  # Inference-only model for TF.js conversion
│   ├── model_metadata.json
│   └── metrics.json
├── notebooks/
├── outputs/
├── scripts/
│   ├── prepare_browser_model.py
│   ├── convert_browser_model.py
│   ├── validate_web_demo.py
│   └── ...
├── src/
├── tests/
├── web/
│   ├── index.html
│   ├── .nojekyll
│   ├── assets/
│   │   ├── app.js
│   │   ├── styles.css
│   │   ├── model_metadata.json
│   │   └── metrics/
│   ├── samples/
│   └── model/                          # Generated during GitHub Actions
├── README_GITHUB_PAGES.md
├── requirements-pages.txt
├── requirements.txt
├── Dockerfile
└── README.md
```

The root-level workflow is located at:

```text
.github/workflows/03-densenet-medical-image-classification.yml
```

## Run the browser site locally

Install the model-conversion dependencies:

```bash
cd 03-densenet-medical-image-classification
python -m venv .venv-pages
```

Windows:

```bat
.venv-pages\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-pages.txt
python scripts/convert_browser_model.py
python -m http.server 8000 --directory web
```

Open:

```text
http://localhost:8000
```

Opening `web/index.html` directly with a `file://` URL is not supported because browsers block model-file fetches. Use a local HTTP server.

## Run the optional Python/Gradio version

```bash
python -m venv .venv
pip install -r requirements.txt
python app.py
```

## Rebuild the inference-only HDF5 model

The browser HDF5 artifact is already included. To regenerate it from the original `.keras` model:

```bash
python scripts/prepare_browser_model.py \
  --source models/densenet121_medical.keras \
  --output models/densenet121_medical_browser.h5
```

The script verifies that the browser model reproduces the original model’s inference output within a small numerical tolerance before saving.

## Reproduce training on Kaggle

Use:

```text
notebooks/densenet_medical_image_classification_kaggle.ipynb
```

The notebook contains dataset loading, exploratory analysis, splitting, augmentation, DenseNet transfer learning, class-imbalance handling, confusion matrix, classification report, ROC/PR analysis, error analysis, Grad-CAM hooks, artifact export, and medical-use limitations.

See [`kaggle/README_KAGGLE.md`](kaggle/README_KAGGLE.md).

## Skills demonstrated

- CNN architecture and DenseNet121 transfer learning
- Image preprocessing and browser tensor preparation
- Keras-to-TensorFlow.js model conversion
- Static ML deployment on GitHub Pages
- Client-side private inference
- GitHub Actions CI/CD
- Model evaluation beyond accuracy
- Error analysis and explainability planning
- Dataset and artifact auditing
- Responsible communication of medical-AI limitations
- Modular Python, tests, Docker, Gradio, and Kaggle workflows

## Portfolio description

> Built and deployed an audited DenseNet121 image-classification pipeline with TensorFlow.js browser inference, GitHub Pages CI/CD, reusable preprocessing, evaluation evidence, and responsible AI disclosures.

# CNN Projects

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-d00000.svg)](https://keras.io/)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-Browser%20Inference-ffca28.svg)](https://www.tensorflow.org/js)
[![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime-Browser%20Inference-005ced.svg)](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
[![Vercel](https://img.shields.io/badge/Vercel-4%20Live%20Apps-black.svg)](https://vercel.com/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-3%20Live%20Apps-222222.svg)](https://unit-mole.github.io/cnn-projects/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Project--Specific%20CI-2088ff.svg)](https://github.com/unit-mole/cnn-projects/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A structured computer-vision portfolio containing **seven completed Convolutional Neural Network projects** across medical-image segmentation, object detection, medical-image classification, residual learning, fine-grained classification, satellite-image segmentation, transfer learning, model comparison, explainability, and browser-based inference.

Each project is developed as an end-to-end case study with task-specific preprocessing, model training or transfer learning, evaluation, reusable source code, deployment assets, automated validation, responsible-use guidance, and a publicly accessible application.

**Portfolio status:** 7 completed and deployed projects  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)  
**Deployment portfolio:** 4 Vercel applications · 3 GitHub Pages applications

---

## Portfolio Objective

This repository demonstrates how Convolutional Neural Networks and modern computer-vision architectures can be applied across classification, detection, segmentation, transfer learning, explainability, and browser-side deployment.

Each project is designed to move beyond notebook-only experimentation and generally contains:

- a clearly defined computer-vision problem;
- reproducible image preprocessing;
- deterministic data splitting where applicable;
- a task-appropriate CNN architecture;
- transfer learning or from-scratch training;
- task-specific evaluation metrics;
- model comparison where meaningful;
- saved reports, charts, predictions, and deployment assets;
- modular source code and reusable utilities;
- automated testing or GitHub Actions validation;
- an interactive public demonstration;
- responsible-use guidance;
- transparent limitations and future improvements.

The portfolio is intended to demonstrate skills relevant to:

- Data Science;
- Machine Learning;
- Applied Artificial Intelligence;
- Computer Vision;
- Deep Learning;
- Image Analytics;
- Quality Analytics;
- Analytics Engineering;
- AI application development;
- browser-based machine learning deployment.

---

## Completed Projects

| No. | Project | Computer-Vision Problem | Primary Deployment | Status |
|---:|---|---|---|---|
| 1 | [Medical Image Segmentation with U-Net](01-image-segmentation-unet-medical-imaging/) | Pixel-level medical-image segmentation | Vercel | [Live Demo](https://medical-image-segmentation-unet.vercel.app/) |
| 2 | [Object Detection Using CNN](02-object-detection-using-cnn/) | Object localization and classification | Vercel | [Live Demo](https://cnn-object-detection.vercel.app/) |
| 3 | [DenseNet Medical Image Classification](03-densenet-medical-image-classification/) | Transfer-learning medical-image classification | GitHub Pages | [Live Demo](https://unit-mole.github.io/cnn-projects/03-densenet-medical-image-classification/) |
| 4 | [Image Classification with ResNet50 and TensorFlow.js](04-image-classification-resnet/) | Residual-network image classification and browser inference | GitHub Pages | [Live Demo](https://unit-mole.github.io/cnn-projects/04-image-classification-resnet/) |
| 5 | [Fine-Grained Image Classification with VGG16](05-fine-grained-image-classification-vgg16/) | Fine-grained visual-category classification | Vercel | [Live Demo](https://vgg16-fine-grained-image-classifica.vercel.app/) |
| 6 | [Satellite Image Segmentation with U-Net](06-satellite-image-segmentation-unet/) | Semantic segmentation of satellite imagery | Vercel | [Live Demo](https://satellite-image-segmentation-unet.vercel.app/) |
| 7 | [CNN Model Comparison with AlexNet-Style Networks and MobileNetV2](07-image-classification-alexnet-transfer-learning/) | From-scratch CNN comparison, transfer learning, explainability, and ONNX deployment | GitHub Pages | [Live Demo](https://unit-mole.github.io/cnn-projects/) |

---

## Portfolio at a Glance

| Coverage Area | Projects |
|---|---|
| Medical-image segmentation | Project 01 |
| Object detection | Project 02 |
| Medical-image classification | Project 03 |
| Residual learning | Project 04 |
| Fine-grained image classification | Project 05 |
| Satellite-image segmentation | Project 06 |
| From-scratch CNN development | Project 07 |
| Transfer learning | Projects 03, 04, 05, and 07 |
| U-Net architecture | Projects 01 and 06 |
| Dense connectivity | Project 03 |
| Residual networks | Project 04 |
| VGG architecture | Project 05 |
| AlexNet-style architecture | Project 07 |
| MobileNetV2 fine-tuning | Project 07 |
| Explainability | Project 07 |
| TensorFlow.js browser inference | Projects 03 and 04 |
| ONNX Runtime Web | Project 07 |
| Static deployment | Projects 03, 04, and 07 |
| Full web application deployment | Projects 01, 02, 05, and 06 |
| Automated validation | All seven projects |

---

## What the Portfolio Covers

The projects are intentionally varied so that the repository demonstrates multiple CNN families, application domains, training strategies, evaluation methods, and deployment patterns.

### Image Segmentation

- **Medical Image Segmentation with U-Net** demonstrates pixel-level prediction for medical imagery.
- **Satellite Image Segmentation with U-Net** applies semantic segmentation to geospatial and satellite-image data.

These projects demonstrate:

- encoder-decoder CNNs;
- skip connections;
- pixel-level classification;
- segmentation masks;
- image-mask preprocessing;
- Dice-oriented evaluation;
- Intersection over Union;
- qualitative overlay inspection;
- deployment of segmentation results.

### Object Detection

- **Object Detection Using CNN** demonstrates the combined tasks of locating objects and assigning classes.

This project demonstrates:

- bounding-box prediction;
- class prediction;
- image annotation;
- confidence thresholds;
- non-maximum suppression concepts;
- localization and classification evaluation;
- browser-facing detection results.

### Image Classification

- **DenseNet Medical Image Classification** demonstrates dense connectivity and medical-image transfer learning.
- **ResNet50 Image Classification** demonstrates residual learning and TensorFlow.js deployment.
- **Fine-Grained Image Classification with VGG16** focuses on visually similar categories.
- **Project 07** compares from-scratch CNNs with frozen and fine-tuned MobileNetV2.

These projects demonstrate:

- image preprocessing;
- convolutional feature extraction;
- class-probability estimation;
- transfer learning;
- partial fine-tuning;
- controlled model comparison;
- class-level evaluation;
- browser-side inference.

### Explainability and Robustness

Project 07 extends classification beyond aggregate accuracy by including:

- confusion matrices;
- per-class precision, recall, and F1;
- high-confidence error analysis;
- Grad-CAM visualizations;
- corruption and robustness testing;
- calibration metrics;
- model-size and latency comparison.

### Deployment Engineering

The portfolio uses two deployment strategies:

- **Vercel** for four interactive web applications;
- **GitHub Pages** for three static browser-inference applications.

The GitHub Pages applications are built into one combined deployment artifact so that Projects 03, 04, and 07 remain online simultaneously without overwriting one another.

---

## Project Summaries

### 01 — Medical Image Segmentation with U-Net

[![Open Project 01](https://img.shields.io/badge/Open-Project%2001-2ea44f.svg)](01-image-segmentation-unet-medical-imaging/)
[![Live Demo](https://img.shields.io/badge/Vercel-Live%20Demo-black.svg)](https://medical-image-segmentation-unet.vercel.app/)

This project applies a U-Net-style encoder-decoder CNN to medical-image segmentation. It focuses on predicting a pixel-level mask rather than one class for the entire image.

**Key capabilities:**

- medical-image preprocessing;
- image-mask alignment;
- U-Net encoder-decoder architecture;
- skip connections;
- segmentation-mask prediction;
- Dice and overlap-oriented evaluation;
- qualitative image-mask visualization;
- interactive web deployment.

---

### 02 — Object Detection Using CNN

[![Open Project 02](https://img.shields.io/badge/Open-Project%2002-2ea44f.svg)](02-object-detection-using-cnn/)
[![Live Demo](https://img.shields.io/badge/Vercel-Live%20Demo-black.svg)](https://cnn-object-detection.vercel.app/)

This project demonstrates object detection by combining visual feature extraction with object localization and class prediction.

**Key capabilities:**

- image upload and validation;
- object localization;
- bounding-box rendering;
- class-confidence presentation;
- detection-result visualization;
- reusable inference workflow;
- interactive Vercel application.

---

### 03 — DenseNet Medical Image Classification

[![Open Project 03](https://img.shields.io/badge/Open-Project%2003-2ea44f.svg)](03-densenet-medical-image-classification/)
[![Live Demo](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-222222.svg)](https://unit-mole.github.io/cnn-projects/03-densenet-medical-image-classification/)

This project uses DenseNet transfer learning for medical-image classification and publishes a browser-based application through GitHub Pages.

**Key capabilities:**

- DenseNet architecture;
- dense feature reuse;
- transfer learning;
- medical-image preprocessing;
- class-level evaluation;
- confusion-matrix analysis;
- TensorFlow.js browser model;
- static GitHub Pages deployment.

---

### 04 — Image Classification with ResNet50 and TensorFlow.js

[![Open Project 04](https://img.shields.io/badge/Open-Project%2004-2ea44f.svg)](04-image-classification-resnet/)
[![Live Demo](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-222222.svg)](https://unit-mole.github.io/cnn-projects/04-image-classification-resnet/)

This project uses a ResNet50 transfer-learning pipeline and deploys the converted model for direct browser inference with TensorFlow.js.

**Key capabilities:**

- residual connections;
- ResNet50 transfer learning;
- image classification;
- model evaluation;
- TensorFlow.js conversion;
- browser-side preprocessing;
- local browser inference;
- GitHub Pages deployment.

---

### 05 — Fine-Grained Image Classification with VGG16

[![Open Project 05](https://img.shields.io/badge/Open-Project%2005-2ea44f.svg)](05-fine-grained-image-classification-vgg16/)
[![Live Demo](https://img.shields.io/badge/Vercel-Live%20Demo-black.svg)](https://vgg16-fine-grained-image-classifica.vercel.app/)

This project uses VGG16 transfer learning for fine-grained classification, where categories can share similar visual structures and require more detailed feature discrimination.

**Key capabilities:**

- VGG16 architecture;
- transfer learning;
- fine-grained category recognition;
- image augmentation;
- class-level evaluation;
- prediction-confidence presentation;
- interactive Vercel deployment.

---

### 06 — Satellite Image Segmentation with U-Net

[![Open Project 06](https://img.shields.io/badge/Open-Project%2006-2ea44f.svg)](06-satellite-image-segmentation-unet/)
[![Live Demo](https://img.shields.io/badge/Vercel-Live%20Demo-black.svg)](https://satellite-image-segmentation-unet.vercel.app/)

This project applies a U-Net segmentation pipeline to satellite imagery for pixel-level scene understanding.

**Key capabilities:**

- satellite-image preprocessing;
- geospatial image segmentation;
- U-Net architecture;
- image-mask visualization;
- overlap-oriented evaluation;
- prediction overlays;
- interactive Vercel application.

---

### 07 — CNN Model Comparison with AlexNet-Style Networks and MobileNetV2

[![Open Project 07](https://img.shields.io/badge/Open-Project%2007-2ea44f.svg)](07-image-classification-alexnet-transfer-learning/)
[![Live Demo](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-222222.svg)](https://unit-mole.github.io/cnn-projects/)

This project compares a Simple CNN, an AlexNet-style CNN trained from scratch, frozen MobileNetV2, and partially fine-tuned MobileNetV2 on a controlled four-class image-classification task. The strongest model is selected using macro F1, exported to ONNX, explained with Grad-CAM, tested for robustness, and deployed through ONNX Runtime Web.

**Key capabilities:**

- Simple CNN baseline;
- AlexNet-style CNN from scratch;
- MobileNetV2 transfer learning;
- partial fine-tuning;
- controlled four-model comparison;
- accuracy, balanced accuracy, precision, recall, and F1;
- top-2 accuracy and ROC-AUC;
- calibration metrics;
- robustness analysis;
- Grad-CAM explainability;
- ONNX export;
- browser inference;
- GitHub Pages deployment.

**Selected deployment model:** Fine-tuned MobileNetV2  
**Test accuracy:** 91.03%  
**Macro F1:** 90.89%  
**Top-2 accuracy:** 98.74%

---

## CNN Architecture Coverage

| Architecture Family | Demonstrated Through |
|---|---|
| U-Net | Projects 01 and 06 |
| Generic CNN feature extraction | Projects 02 and 07 |
| DenseNet | Project 03 |
| ResNet50 | Project 04 |
| VGG16 | Project 05 |
| AlexNet-style CNN | Project 07 |
| MobileNetV2 | Project 07 |
| Encoder-decoder segmentation | Projects 01 and 06 |
| Residual learning | Project 04 |
| Dense connectivity | Project 03 |
| Depthwise-separable convolution | Project 07 |
| Transfer learning | Projects 03, 04, 05, and 07 |
| From-scratch CNN training | Project 07 |

---

## Evaluation Coverage

The projects use evaluation methods aligned with the actual task instead of relying on one universal metric.

| Task | Evaluation Methods |
|---|---|
| Image segmentation | Dice coefficient, Intersection over Union, pixel overlap, qualitative mask review |
| Object detection | Localization quality, class confidence, predicted boxes, qualitative inspection |
| Image classification | Accuracy, precision, recall, macro F1, weighted F1, confusion matrices |
| Fine-grained classification | Class-level metrics, confidence analysis, error review |
| Model comparison | Controlled splits, macro F1, parameters, size, latency |
| Calibration | Negative log likelihood, Brier score, expected calibration error |
| Robustness | Corruption-specific macro F1 and drop from clean performance |
| Explainability | Grad-CAM activation overlays |
| Browser deployment | Required-file validation, model loading, JavaScript syntax, static-asset checks |

### Why multiple evaluation methods matter

- Accuracy alone can hide weak minority-class performance.
- Segmentation requires pixel-level overlap measures.
- Detection must evaluate both localization and category prediction.
- Fine-grained classes often require per-class error analysis.
- Softmax confidence is not automatically a calibrated probability.
- Latency and model size affect deployment feasibility.
- Browser performance depends on hardware, runtime, and model format.
- Explainability visualizations support inspection but do not prove causal reasoning.

---

## What the Repository Demonstrates

### End-to-End Computer-Vision Delivery

The repository demonstrates the complete path from an idea to a public application:

- problem definition;
- dataset acquisition;
- image validation;
- preprocessing;
- augmentation;
- deterministic splitting;
- architecture selection;
- transfer learning or from-scratch training;
- checkpoint preservation;
- evaluation;
- error analysis;
- saved artifacts;
- reusable inference code;
- testing;
- CI validation;
- application development;
- public deployment;
- documentation;
- responsible-use communication.

### Model Selection Based on Evidence

The projects do not assume that the largest or newest architecture is automatically the best.

Examples include:

- transfer-learning models evaluated against task-specific baselines;
- Project 07 comparison of four CNN strategies;
- deployment selection based on macro F1 rather than only accuracy;
- robustness testing before presenting the selected model;
- ONNX and browser artifacts validated before deployment;
- model-size and latency trade-offs documented.

### Reliable and Reusable Engineering

The repository includes practices needed for dependable experimentation and inference:

- modular source files;
- reusable preprocessing;
- deterministic seeds;
- consistent feature and label mappings;
- safe handling of invalid inputs;
- metadata and configuration recording;
- training-history preservation;
- checkpoint and artifact verification;
- project-specific tests;
- project-specific GitHub Actions workflows;
- one controlled combined GitHub Pages deployment;
- large-file and virtual-environment protection through `.gitignore`;
- deployment assets separated from training environments.

---

## Deployment Diversity

The seven projects intentionally use two deployment approaches.

| Platform | Projects | Purpose |
|---|---:|---|
| Vercel | 4 | Interactive web applications for segmentation, detection, and classification |
| GitHub Pages | 3 | Static TensorFlow.js and ONNX browser-inference applications |

### Live Applications

| Project | Platform | URL |
|---|---|---|
| Project 01 | Vercel | https://medical-image-segmentation-unet.vercel.app/ |
| Project 02 | Vercel | https://cnn-object-detection.vercel.app/ |
| Project 03 | GitHub Pages | https://unit-mole.github.io/cnn-projects/03-densenet-medical-image-classification/ |
| Project 04 | GitHub Pages | https://unit-mole.github.io/cnn-projects/04-image-classification-resnet/ |
| Project 05 | Vercel | https://vgg16-fine-grained-image-classifica.vercel.app/ |
| Project 06 | Vercel | https://satellite-image-segmentation-unet.vercel.app/ |
| Project 07 | GitHub Pages | https://unit-mole.github.io/cnn-projects/ |

This demonstrates the ability to select a deployment method based on runtime needs rather than using one platform for every project.

---

## GitHub Pages Deployment Architecture

Projects 03, 04, and 07 share one repository-level GitHub Pages site.

```text
Combined GitHub Pages artifact
│
├── /                                      → Project 07
├── /03-densenet-medical-image-classification/
├── /04-image-classification-resnet/
└── /07-image-classification-alexnet-transfer-learning/
```

Only the combined workflow performs the Pages deployment:

```text
.github/workflows/cnn-projects-pages.yml
```

The individual Project 03, 04, and 07 workflows perform validation only. This prevents one project from overwriting the other deployed applications.

---

## Repository Convention

The repository is organized as a monorepo.

```text
cnn-projects/
├── .github/
│   └── workflows/
│       ├── 01-image-segmentation-unet-medical-imaging.yml
│       ├── 02-object-detection-using-cnn.yml
│       ├── 03-densenet-medical-image-classification.yml
│       ├── 04-image-classification-resnet.yml
│       ├── 05-fine-grained-image-classification-vgg16.yml
│       ├── 06-satellite-image-segmentation-unet.yml
│       ├── 07-image-classification-alexnet-transfer-learning.yml
│       └── cnn-projects-pages.yml
│
├── 01-image-segmentation-unet-medical-imaging/
├── 02-object-detection-using-cnn/
├── 03-densenet-medical-image-classification/
├── 04-image-classification-resnet/
├── 05-fine-grained-image-classification-vgg16/
├── 06-satellite-image-segmentation-unet/
├── 07-image-classification-alexnet-transfer-learning/
├── .gitignore
├── LICENSE
└── README.md
```

A typical individual project may contain:

```text
project-folder/
├── data/
├── images/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── web/
├── deployment configuration
├── README.md
├── requirements.txt
└── supporting metadata and reports
```

The exact files differ by task, but the standards remain consistent:

- reproducible workflows;
- modular code;
- task-appropriate evaluation;
- public deployment;
- automated validation;
- safe repository practices;
- transparent limitations;
- portfolio-quality documentation.

---

## Continuous Integration

The repository uses project-specific GitHub Actions workflows.

Depending on the project, CI validates:

- required folder and file structure;
- Python source syntax;
- JavaScript syntax;
- JSON validity;
- notebook JSON validity;
- pytest test suites;
- model configuration;
- static application assets;
- README image references;
- TensorFlow.js model presence;
- ONNX model presence and size;
- browser deployment paths;
- oversized files;
- accidental checkpoint or secret inclusion.

Project workflows run only when their relevant project folders or workflow files change.

The repository-level Pages workflow:

1. checks out the repository;
2. prepares the Project 03 browser model when required;
3. verifies browser assets for Projects 03, 04, and 07;
4. assembles one combined `_site` directory;
5. uploads one Pages artifact;
6. deploys that artifact through the `github-pages` environment.

[![Open GitHub Actions](https://img.shields.io/badge/Open-GitHub%20Actions-2088ff?style=for-the-badge)](https://github.com/unit-mole/cnn-projects/actions)

---

## Run a Project Locally

Each project contains detailed setup instructions. The general workflow is:

### 1. Clone the repository

```bash
git clone https://github.com/unit-mole/cnn-projects.git
cd cnn-projects
```

### 2. Enter a project

```bash
cd 07-image-classification-alexnet-transfer-learning
```

Replace the folder name with the project you want to run.

### 3. Create a virtual environment

**Windows**

```bat
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 4. Install project dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some training workflows use a separate file such as:

```bash
python -m pip install -r requirements-training.txt
```

### 5. Follow the selected project README

Projects may run through:

- Jupyter notebooks;
- Python scripts;
- local HTTP servers;
- static browser applications;
- Vercel development servers.

Always follow the instructions in the selected project's `README.md`.

---

## Responsible Use

This repository is intended for education, experimentation, technical demonstration, and portfolio presentation.

General limitations include:

- image datasets may not represent real deployment populations;
- pretrained models can inherit biases from their original training data;
- medical-image demonstrations are not clinical diagnostic systems;
- segmentation masks may miss or falsely include regions;
- object detectors may miss small, occluded, or unsupported objects;
- confidence scores may not be calibrated probabilities;
- visual models may fail on out-of-distribution images;
- browser performance varies by device and runtime;
- benchmark results should not be generalized beyond the evaluated configuration;
- portfolio models are not automatically production-ready;
- no application should be used as the sole basis for medical, safety-critical, financial, legal, hiring, insurance, quality-release, or production decisions.

Important outputs should be verified through trusted sources, domain expertise, and additional validation.

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| Medical-image segmentation | Project 01 |
| Object detection | Project 02 |
| Medical-image classification | Project 03 |
| Dense connectivity | Project 03 |
| Residual learning | Project 04 |
| TensorFlow.js browser inference | Projects 03 and 04 |
| Fine-grained image classification | Project 05 |
| Satellite-image segmentation | Project 06 |
| U-Net architecture | Projects 01 and 06 |
| CNN training from scratch | Project 07 |
| AlexNet-style architecture | Project 07 |
| MobileNetV2 fine-tuning | Project 07 |
| Controlled model comparison | Project 07 |
| Calibration analysis | Project 07 |
| Grad-CAM explainability | Project 07 |
| Robustness evaluation | Project 07 |
| ONNX conversion | Project 07 |
| ONNX Runtime Web | Project 07 |
| Vercel deployment | Projects 01, 02, 05, and 06 |
| GitHub Pages deployment | Projects 03, 04, and 07 |
| CI/CD | All seven projects |

---

## Core Skills Demonstrated

`Convolutional Neural Networks` · `Computer Vision` · `Deep Learning` · `Python` · `TensorFlow` · `Keras` · `PyTorch` · `torchvision` · `Image Segmentation` · `Object Detection` · `Image Classification` · `Medical Imaging` · `Satellite Imaging` · `U-Net` · `DenseNet` · `ResNet50` · `VGG16` · `AlexNet` · `MobileNetV2` · `Transfer Learning` · `Fine-Tuning` · `Data Augmentation` · `Class-Weighted Training` · `Accuracy` · `Precision` · `Recall` · `F1-Score` · `Dice Coefficient` · `Intersection over Union` · `Confusion Matrices` · `Calibration` · `Robustness Testing` · `Grad-CAM` · `TensorFlow.js` · `ONNX` · `ONNX Runtime Web` · `JavaScript` · `HTML` · `CSS` · `Vercel` · `GitHub Pages` · `Testing` · `GitHub Actions` · `CI/CD` · `Responsible AI Communication`

---

## Portfolio Positioning

**One-line description:** Seven end-to-end CNN projects spanning medical and satellite segmentation, object detection, DenseNet, ResNet50, VGG16, AlexNet-style networks, MobileNetV2 transfer learning, explainability, TensorFlow.js, ONNX browser inference, Vercel, and GitHub Pages deployment.

**Pinned repository description:** Professional computer-vision portfolio featuring seven deployed CNN projects across segmentation, detection, classification, transfer learning, from-scratch architecture development, model comparison, Grad-CAM, robustness evaluation, TensorFlow.js, ONNX Runtime Web, project-specific CI, Vercel, and GitHub Pages.

This portfolio connects naturally to a Quality Data Scientist background because computer-vision systems can support:

- visual inspection;
- defect localization;
- product categorization;
- medical-image research demonstrations;
- satellite-image analysis;
- automated quality checks;
- image-based anomaly detection;
- inspection-workflow support;
- structured model evaluation and release governance.

---

## License

This repository is distributed under the [MIT License](LICENSE).

Individual models, datasets, and third-party libraries remain subject to their original licenses and usage conditions.

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Computer Vision | Analytics Engineering | Quality Analytics

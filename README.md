# CNN Projects

A structured computer-vision portfolio containing seven Convolutional Neural Network projects across medical image segmentation, object detection, medical image classification, residual learning, fine-grained classification, satellite image segmentation, transfer learning, and client-side browser inference.

**Portfolio status:** 6 completed projects · 1 final project in progress  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)  
**Repository:** [unit-mole/cnn-projects](https://github.com/unit-mole/cnn-projects)

---

## Portfolio Objective

This repository demonstrates how Convolutional Neural Networks and established computer-vision architectures can be applied to practical image-based problems. Each project is developed as an end-to-end case study containing:

- a clearly defined computer-vision, analytical, or applied-AI problem;
- reproducible image, mask, annotation, and label preprocessing;
- leakage-aware training, validation, and test design;
- CNN, U-Net, DenseNet, ResNet, VGG16, AlexNet, and transfer-learning workflows;
- task-appropriate baseline comparison and evaluation;
- saved model, metadata, label-mapping, preprocessing, and browser-inference artifacts;
- modular and reusable training, evaluation, prediction, and visualization code;
- interactive demonstrations using Vercel, TensorFlow.js, GitHub Pages, Hugging Face Spaces, or Gradio;
- Kaggle-ready notebooks where training evidence and GPU execution are useful;
- automated tests and project-specific GitHub Actions CI;
- local execution, Docker, and deployment guidance;
- honest discussion of assumptions, synthetic-data constraints, limitations, responsible use, and future improvements.

The portfolio is designed to demonstrate skills relevant to Data Science, Machine Learning, Applied AI, Computer Vision, Deep Learning, Quality Analytics, Visual Inspection Analytics, Remote Sensing Analytics, Analytics Engineering, and image-based automation roles.

---

## CNN Portfolio Projects

| No. | Project | Architecture / Problem Type | Deployment | Status |
|---:|---|---|---|---|
| 1 | [Medical Image Segmentation using U-Net](01-image-segmentation-unet-medical-imaging/) | U-Net binary medical-image segmentation and pixel-level mask prediction | Hugging Face Spaces + Gradio | Completed |
| 2 | [Object Detection using CNN](02-object-detection-using-cnn/) | CNN-based object localization, bounding-box prediction, and browser visualization | [Vercel + TensorFlow.js](https://cnn-object-detection.vercel.app/) | Completed and deployed |
| 3 | [DenseNet Medical Image Classification](03-densenet-medical-image-classification/) | DenseNet-based medical-image classification with reusable feature propagation | Hugging Face Spaces + Gradio · Kaggle training notebook | Completed |
| 4 | [Image Classification using ResNet](04-image-classification-resnet/) | Residual CNN image classification with TensorFlow.js browser inference | [GitHub Pages + TensorFlow.js](https://unit-mole.github.io/cnn-projects/) | Completed and deployed |
| 5 | [Fine-Grained Image Classification using VGG16](05-fine-grained-image-classification-vgg16/) | VGG16 transfer learning for fine-grained visual classification | [Vercel + TensorFlow.js](https://vgg16-fine-grained-image-classifica.vercel.app/) | Completed and deployed |
| 6 | [Satellite Image Segmentation using U-Net](06-satellite-image-segmentation-unet/) | U-Net binary semantic segmentation for synthetic satellite-style urban regions | [Vercel + TensorFlow.js](https://satellite-image-segmentation-unet.vercel.app/) | Completed and deployed |
| 7 | [Image Classification using AlexNet Transfer Learning](07-image-classification-alexnet-transfer-learning/) | AlexNet-inspired transfer learning and browser-based image classification | [Planned GitHub Pages + TensorFlow.js demo](https://unit-mole.github.io/cnn-projects/07-image-classification-alexnet-transfer-learning/) | In progress |

> Project 07 is included in the portfolio structure and deployment roadmap while its final trained model and production browser assets are being completed. The listed Project 07 URL is the planned GitHub Pages location and may remain unavailable until the final deployment is published.

---

## What the Portfolio Covers

The seven projects are intentionally varied so that the repository demonstrates multiple forms of CNN-based computer vision rather than repeating one image-classification workflow.

### Image Segmentation

- **Medical Image Segmentation using U-Net** predicts a binary mask for a target region in a medical-style image and demonstrates image-mask alignment, segmentation preprocessing, overlap metrics, and visual overlays.
- **Satellite Image Segmentation using U-Net** produces pixel-level urban-region masks from synthetic satellite-style RGB tiles and demonstrates browser-based segmentation, probability maps, mask overlays, threshold adjustment, and downloadable predictions.

These projects demonstrate encoder-decoder architectures, skip connections, image-mask pairing, nearest-neighbor mask resizing, pixel normalization, sigmoid outputs, mask thresholding, Dice coefficient, IoU, probability maps, overlays, error analysis, and responsible interpretation of pixel-level predictions.

### Object Detection and Localization

- **Object Detection using CNN** identifies target regions through bounding boxes and presents the result through an interactive browser application.

This project demonstrates image annotation handling, bounding-box representation, localization logic, confidence output, visual detection overlays, browser inference, and deployment of a lightweight computer-vision interface.

### Medical Image Classification

- **DenseNet Medical Image Classification** demonstrates densely connected convolutional blocks, feature reuse, multi-class prediction, confidence interpretation, and healthcare-specific responsible-use communication.

This project demonstrates image classification, DenseNet architecture, class mapping, data augmentation, confusion-matrix analysis, class-wise evaluation, model persistence, Gradio deployment, and Kaggle-based training evidence.

### Residual and Fine-Grained Classification

- **ResNet Image Classification** uses residual connections to improve gradient flow and supports static TensorFlow.js inference through GitHub Pages.
- **VGG16 Fine-Grained Image Classification** uses transfer learning and browser inference to distinguish visually similar categories.

These projects demonstrate residual learning, skip connections, pretrained feature extraction, fine-tuning, image resizing, normalization, Top-K probabilities, client-side inference, static hosting, and model conversion for the browser.

### Transfer Learning with AlexNet

- **AlexNet Transfer Learning** is the final portfolio project and is designed to demonstrate classic CNN design, transfer learning, data augmentation, classification evaluation, TensorFlow.js export, and static browser deployment.

The project is already represented in the monorepo and portfolio roadmap while final model training and deployment work is completed.

---

## What the Repository Demonstrates

### End-to-End Computer-Vision Delivery

Every project is structured to move beyond notebook-only experimentation. The repository demonstrates:

- business and analytical problem definition;
- reproducible image, mask, annotation, and label preparation;
- image resizing, color conversion, normalization, and augmentation;
- train, validation, and test separation;
- CNN architecture development and evaluation;
- saved Keras models, TensorFlow.js models, metadata, labels, thresholds, and preprocessing settings;
- reusable classification, detection, segmentation, and visualization pipelines;
- manual upload and safe sample-image inference;
- downloadable masks or visual prediction outputs where appropriate;
- local execution;
- cloud and static-site deployment;
- CI-based validation.

### CNN Architecture Diversity

The repository covers several major CNN design patterns:

- U-Net encoder-decoder segmentation;
- skip connections for spatial-detail preservation;
- DenseNet feature reuse through dense connectivity;
- ResNet residual learning;
- VGG16 transfer learning;
- AlexNet-inspired convolutional classification;
- custom CNN localization and object-detection logic;
- Keras-to-TensorFlow.js browser-model conversion.

### Correct Image and Mask Processing

Computer-vision systems require consistent preprocessing. The projects emphasize:

- RGB and grayscale handling;
- fixed input-shape validation;
- image normalization;
- aspect-ratio and resizing decisions;
- nearest-neighbor resizing for segmentation masks;
- label and class-order consistency;
- aligned geometric augmentation for images and masks;
- safe handling of invalid uploads and unsupported formats;
- consistent training and inference preprocessing;
- saved preprocessing configuration in model metadata.

### Evaluation Based on the Actual Problem

The projects use metrics that match each computer-vision task rather than relying on one headline accuracy value.

Examples include:

- Dice coefficient, IoU, pixel accuracy, precision, recall, and F1 for segmentation;
- confidence scores, class-wise precision, recall, F1, confusion matrices, and ROC analysis for classification;
- localization quality, bounding-box visualization, confidence interpretation, and detection review for object detection;
- Top-1 and Top-K predictions for browser classifiers;
- baseline comparisons to determine whether a CNN adds measurable value;
- visual overlays, probability maps, error maps, and weak-prediction examples;
- inference-time and browser-runtime reporting.

### Reliable and Reusable Engineering

The repository includes practices required for dependable inference:

- training and inference preprocessing consistency;
- saved image size, channel order, normalization method, label mapping, and threshold configuration;
- modular source files rather than notebook-only logic;
- pretrained application startup without automatic retraining;
- safe sample data for public demonstrations;
- automated preprocessing and inference tests;
- project-specific GitHub Actions workflows;
- lightweight CI that avoids full model retraining;
- GitHub-safe data and model-artifact management;
- Git LFS or external model hosting guidance where needed;
- static browser deployment without a Python backend where feasible.

### Deployment Diversity

The portfolio intentionally demonstrates multiple delivery approaches:

| Platform | Demonstrated Through |
|---|---|
| Vercel | Object Detection, VGG16 Fine-Grained Classification, Satellite U-Net Segmentation |
| TensorFlow.js | Browser-based classification, detection, and segmentation |
| GitHub Pages | ResNet browser inference and planned AlexNet deployment |
| Hugging Face Spaces | Gradio-based model demonstrations |
| Gradio | Upload-driven image prediction interfaces |
| Kaggle Notebooks | Training, evaluation, GPU execution, and reproducibility |
| GitHub Actions | Project-specific CI and deployment validation |
| Docker | Reproducible local or container-based execution where useful |

This diversity shows that the portfolio covers not only model development but also practical model packaging and deployment.

---

## Project Highlights

### 01 — Medical Image Segmentation using U-Net

```text
Medical-style input image
          │
          ▼
Image and mask preprocessing
          │
          ▼
U-Net encoder
          │
          ▼
Bottleneck
          │
          ▼
U-Net decoder with skip connections
          │
          ▼
Predicted binary segmentation mask
```

**Core capabilities**

- Image-mask pairing
- Binary segmentation
- U-Net architecture
- Dice and IoU evaluation
- Predicted-mask overlays
- Gradio-ready inference
- Medical responsible-use communication

### 02 — Object Detection using CNN

```text
Uploaded image
      │
      ▼
Image preprocessing
      │
      ▼
CNN feature extraction
      │
      ▼
Object localization
      │
      ▼
Bounding box and confidence output
      │
      ▼
Vercel browser visualization
```

**Live demo:** https://cnn-object-detection.vercel.app/

### 03 — DenseNet Medical Image Classification

```text
Medical-style image
        │
        ▼
Preprocessing and augmentation
        │
        ▼
Dense convolution blocks
        │
        ▼
Feature transition layers
        │
        ▼
Global feature aggregation
        │
        ▼
Class probabilities
```

**Core capabilities**

- Dense connectivity
- Transfer learning
- Class-wise evaluation
- Confidence interpretation
- Gradio application
- Kaggle-ready training workflow

### 04 — ResNet Image Classification

```text
Uploaded image
      │
      ▼
Browser preprocessing
      │
      ▼
Residual CNN blocks
      │
      ▼
TensorFlow.js inference
      │
      ▼
Top prediction and confidence
```

**Live demo:** https://unit-mole.github.io/cnn-projects/

### 05 — VGG16 Fine-Grained Image Classification

```text
Uploaded image
      │
      ▼
Resize and normalize
      │
      ▼
VGG16 feature extractor
      │
      ▼
Fine-grained classification head
      │
      ▼
Browser probability output
```

**Live demo:** https://vgg16-fine-grained-image-classifica.vercel.app/

### 06 — Satellite Image Segmentation using U-Net

```text
Satellite-style RGB tile
          │
          ▼
64 × 64 RGB preprocessing
          │
          ▼
Compact U-Net
          │
          ▼
Pixel probability map
          │
          ▼
Binary mask and visual overlay
          │
          ▼
TensorFlow.js browser inference
```

**Live demo:** https://satellite-image-segmentation-unet.vercel.app/

The current portfolio model uses procedurally generated satellite-style tiles. Its near-perfect synthetic test metrics validate the pipeline but should not be interpreted as evidence of real-world remote-sensing performance.

### 07 — AlexNet Transfer Learning

```text
Input image
     │
     ▼
Image augmentation and normalization
     │
     ▼
AlexNet-inspired convolution blocks
     │
     ▼
Transfer-learning classifier
     │
     ▼
TensorFlow.js model export
     │
     ▼
Static browser deployment
```

**Planned demo:** https://unit-mole.github.io/cnn-projects/07-image-classification-alexnet-transfer-learning/

**Current status:** Final model training, evaluation, browser artifact export, and production deployment are in progress.

---

## Business and Quality Analytics Translation

The applications do not stop at raw model outputs. Depending on the project, they provide:

- predicted image classes;
- Top-K class probabilities;
- object bounding boxes;
- localization confidence;
- medical-style segmentation masks;
- satellite-style urban-region masks;
- probability maps;
- mask overlays;
- segmented-area measurements;
- browser inference timing;
- downloadable prediction artifacts;
- model and baseline comparisons;
- visual error interpretations;
- limitations and responsible-use statements.

This demonstrates the ability to translate technical computer-vision outputs into information that can be understood by data scientists, quality engineers, analysts, operations teams, managers, and other business stakeholders.

The portfolio also connects naturally to the user's current Quality Data Scientist background through:

- automated visual inspection;
- defect and anomaly localization;
- region-of-interest detection;
- image-based monitoring;
- affected-area measurement;
- segmentation-based quality assessment;
- classification-based triage;
- automated image review;
- spatial and visual analytics;
- applied AI for quality workflows.

---

## Responsible Model Communication

Each project documents its intended scope and limitations. The repository avoids presenting educational portfolio models as production-ready medical, geospatial, surveillance, industrial inspection, legal, public-safety, or operational decision systems without additional:

- domain validation;
- representative real-world data;
- bias and robustness analysis;
- calibration;
- monitoring;
- security controls;
- privacy safeguards;
- governance;
- human oversight.

Synthetic and demonstration datasets are clearly identified where used. Public demos should not receive confidential, private, restricted, or sensitive images.

---

## Repository Convention

The repository is organized as a monorepo. Each project generally follows this structure:

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
│       └── 07-image-classification-alexnet-transfer-learning.yml
│
├── 01-image-segmentation-unet-medical-imaging/
├── 02-object-detection-using-cnn/
├── 03-densenet-medical-image-classification/
├── 04-image-classification-resnet/
├── 05-fine-grained-image-classification-vgg16/
├── 06-satellite-image-segmentation-unet/
├── 07-image-classification-alexnet-transfer-learning/
│
├── .gitignore
├── LICENSE
└── README.md
```

A project may contain:

```text
project-folder/
├── app/
├── archive/
├── assets/
├── data/
├── images/
├── kaggle/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── tfjs_model/
├── app.py
├── gradio_app.py
├── index.html
├── Dockerfile
├── README.md
├── README_HOSTING.md
├── README_HUGGINGFACE.md
├── README_VERCEL.md
├── requirements.txt
├── requirements-dev.txt
└── train_model.py
```

The exact files vary by project, but the standards remain consistent:

- reproducible workflows;
- modular code;
- deployable pretrained inference;
- automated validation;
- clear documentation;
- safe repository practices;
- transparent model assumptions and limitations.

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| Binary medical image segmentation | Project 01 U-Net |
| Object detection and localization | Project 02 CNN |
| Dense feature reuse | Project 03 DenseNet |
| Medical-style image classification | Project 03 DenseNet |
| Residual learning | Project 04 ResNet |
| Fine-grained visual classification | Project 05 VGG16 |
| Satellite-style semantic segmentation | Project 06 U-Net |
| Transfer learning | DenseNet, VGG16, and AlexNet projects |
| Encoder-decoder modeling | U-Net segmentation projects |
| Skip connections | U-Net and ResNet projects |
| Image-mask pairing | Medical and satellite segmentation |
| Binary mask preprocessing | U-Net projects |
| Data augmentation | Classification and segmentation training |
| Browser inference | Projects 02, 04, 05, 06, and planned Project 07 |
| Client-side privacy | TensorFlow.js browser applications |
| Model conversion | Keras to TensorFlow.js |
| Static deployment | Vercel and GitHub Pages |
| Interactive Python deployment | Hugging Face Spaces + Gradio |
| Training evidence | Kaggle-ready notebooks |
| Segmentation evaluation | Dice, IoU, precision, recall, F1 |
| Classification evaluation | Accuracy, precision, recall, F1, confusion matrices |
| Detection review | Bounding boxes, confidence, visual localization |
| Visual interpretation | Masks, overlays, probability maps, and prediction panels |
| Testing and CI/CD | pytest, validation scripts, syntax checks, and GitHub Actions |
| Reproducible packaging | Requirements files, Docker, metadata, and run scripts |

---

## Core Skills Demonstrated

`Convolutional Neural Networks` · `CNN` · `Computer Vision` · `Deep Learning` · `Image Classification` · `Image Segmentation` · `Semantic Segmentation` · `Object Detection` · `Object Localization` · `Bounding Boxes` · `Medical Image Analysis` · `Satellite Image Analysis` · `Remote Sensing Analytics` · `U-Net` · `DenseNet` · `ResNet` · `VGG16` · `AlexNet` · `Transfer Learning` · `Fine-Tuning` · `Encoder-Decoder Networks` · `Skip Connections` · `Dense Connections` · `Residual Learning` · `Image Preprocessing` · `Mask Preprocessing` · `Data Augmentation` · `Image-Mask Alignment` · `Pixel-Level Prediction` · `Probability Maps` · `Mask Overlays` · `Dice Coefficient` · `Intersection over Union` · `Precision` · `Recall` · `F1-score` · `Confusion Matrix` · `Baseline Comparison` · `Error Analysis` · `Responsible AI Communication` · `TensorFlow` · `Keras` · `TensorFlow.js` · `OpenCV` · `NumPy` · `pandas` · `scikit-learn` · `Matplotlib` · `Pillow` · `Gradio` · `Hugging Face Spaces` · `Kaggle Notebooks` · `Vercel` · `GitHub Pages` · `JavaScript` · `HTML` · `CSS` · `Docker` · `Testing` · `GitHub Actions` · `CI/CD` · `Browser Inference` · `Privacy-Aware Deployment` · `Quality Analytics` · `Visual Inspection Analytics` · `Business Translation`

---

## Local Setup

Clone the repository:

```bash
git clone https://github.com/unit-mole/cnn-projects.git
cd cnn-projects
```

Open the required project:

```bash
cd 06-satellite-image-segmentation-unet
```

Create and activate a virtual environment.

**Windows**

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the project-specific tests:

```bash
python -m pytest -q
```

Launch instructions differ by project and are documented inside each project-level `README.md`.

---

## Portfolio Roadmap

| Project | Current Stage | Next Milestone |
|---|---|---|
| 01 Medical U-Net | Completed | Add or confirm public Hugging Face Spaces link |
| 02 Object Detection CNN | Deployed | Maintain Vercel browser demo |
| 03 DenseNet Medical Classification | Completed | Add or confirm public Hugging Face Spaces link |
| 04 ResNet Classification | Deployed | Maintain GitHub Pages demo |
| 05 VGG16 Fine-Grained Classification | Deployed | Maintain Vercel browser demo |
| 06 Satellite U-Net Segmentation | Deployed | Maintain Vercel TensorFlow.js demo |
| 07 AlexNet Transfer Learning | In progress | Finish training, export TensorFlow.js artifacts, and publish planned static demo |

---

## GitHub Repository Description

```text
End-to-end CNN computer-vision portfolio featuring U-Net segmentation, object detection, DenseNet medical classification, ResNet browser inference, VGG16 fine-grained classification, satellite segmentation, AlexNet transfer learning, TensorFlow/Keras models, automated testing, and deployment-ready demos.
```

### Suggested GitHub Topics

```text
cnn
convolutional-neural-networks
computer-vision
deep-learning
image-segmentation
semantic-segmentation
object-detection
image-classification
medical-imaging
satellite-image-segmentation
remote-sensing
unet
densenet
resnet
vgg16
alexnet
transfer-learning
tensorflow
keras
tensorflowjs
gradio
huggingface-spaces
vercel
github-pages
kaggle-notebooks
machine-learning
data-science
portfolio-projects
```

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Computer Vision | Analytics Engineering | Quality Analytics

---

## License

This repository is released under the MIT License. Individual datasets or third-party model assets may have separate licenses and usage requirements. See the project-level documentation for details.

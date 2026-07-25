# CNN Projects — Computer Vision Portfolio

[![Project 01 CI](https://github.com/unit-mole/cnn-projects/actions/workflows/01-image-segmentation-unet-medical-imaging.yml/badge.svg)](https://github.com/unit-mole/cnn-projects/actions/workflows/01-image-segmentation-unet-medical-imaging.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.20-ff7c00.svg)](https://www.gradio.app/)

A professional CNN and computer vision portfolio featuring image segmentation, object detection, medical image classification, transfer learning, satellite segmentation, model evaluation, and interactive Hugging Face Spaces / Gradio demos.

I currently work as a **Quality Data Scientist** and am building this repository to demonstrate applied computer vision skills relevant to visual inspection, defect localization, automated quality checks, measurement automation, medical AI proof-of-concepts, and image-based analytics workflows.

## Repository roadmap

| # | Project | Core architecture | Status | Live demo |
|---|---|---|---|---|
| 01 | [Medical Image Segmentation](01-image-segmentation-unet-medical-imaging/) | U-Net | ✅ Complete | [Add Hugging Face Space URL](https://huggingface.co/spaces/YOUR_HF_USERNAME/medical-image-segmentation-unet) |
| 02 | Object Detection | CNN detector | 🧭 Planned | — |
| 03 | Medical Image Classification | DenseNet | 🧭 Planned | — |
| 04 | Image Classification | ResNet | 🧭 Planned | — |
| 05 | Fine-Grained Image Classification | VGG16 | 🧭 Planned | — |
| 06 | Satellite Image Segmentation | U-Net | 🧭 Planned | — |
| 07 | Image Classification | AlexNet transfer learning | 🧭 Planned | — |

## Completed project

### 01 — Medical Image Segmentation using U-Net

An end-to-end binary segmentation project built from the supplied notebook and trained model. It uses a compact U-Net to segment deterministic **synthetic MRI-style grayscale images**, compares the network against an intensity-threshold baseline, evaluates Dice and IoU, and provides a Gradio demo ready for Hugging Face Spaces.

> **Important:** The current model was trained on synthetic image-mask pairs, not real patient scans. The project is a technical portfolio demonstration and is not a medical device.

**Key measured results on the synthetic test set**

| Approach | Dice | IoU |
|---|---:|---:|
| Intensity threshold baseline | 0.9659 | 0.9360 |
| U-Net | 0.9977 | 0.9954 |

See the [project README](01-image-segmentation-unet-medical-imaging/README.md) for dataset context, architecture, results, limitations, deployment, and responsible-use notes.

## Tech stack

- Python 3.11
- TensorFlow / Keras
- NumPy, pandas, scikit-learn
- Pillow and Matplotlib
- Gradio
- Hugging Face Spaces
- pytest and GitHub Actions
- Docker

## Repository organization

```text
cnn-projects/
├── .github/workflows/
│   └── 01-image-segmentation-unet-medical-imaging.yml
├── 01-image-segmentation-unet-medical-imaging/
├── 02-object-detection-using-cnn/
├── 03-densenet-medical-image-classification/
├── 04-image-classification-resnet/
├── 05-fine-grained-image-classification-vgg16/
├── 06-satellite-image-segmentation-unet/
├── 07-image-classification-alexnet-transfer-learning/
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.md
```

## Quick start for Project 01

```bash
git clone https://github.com/unit-mole/cnn-projects.git
cd cnn-projects/01-image-segmentation-unet-medical-imaging
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
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

## GitHub repository description

> End-to-end CNN computer vision portfolio featuring U-Net segmentation, object detection, DenseNet, ResNet, VGG16, AlexNet transfer learning, TensorFlow/Keras models, and Hugging Face Spaces demos.

## Recommended repository topics

`cnn` · `convolutional-neural-networks` · `computer-vision` · `deep-learning` · `image-segmentation` · `medical-image-segmentation` · `unet` · `tensorflow` · `keras` · `gradio` · `huggingface-spaces` · `machine-learning` · `data-science` · `portfolio-projects`

## Responsible use

Each medical-imaging project must clearly distinguish demonstration data from clinical data, document limitations, exclude protected health information, and avoid diagnostic claims.

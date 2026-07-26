# Project Audit

## Supplied assets reviewed

- Executed U-Net notebook
- Saved Keras 3 model (`unet_medical.keras`)
- Metrics JSON
- Existing modular Python/Gradio portfolio package

## Actual experiment identified

- Data: deterministic synthetic MRI-style grayscale images
- Samples: 2,500 at 64×64×1
- Task: binary segmentation of a bright synthetic elliptical region
- Model: compact two-level U-Net, 470,977 parameters
- Test Dice: 0.9976925 on synthetic test data
- Test IoU: 0.9954064 on synthetic test data

## Vercel conversion completed

- Added a responsive static HTML/CSS/JavaScript application.
- Reproduced grayscale resizing and normalization in JavaScript.
- Reconstructed the exact U-Net topology using TensorFlow.js Layers.
- Exported 22 trained Conv2D tensors from the Keras 3 archive.
- Added a deterministic 1,883,908-byte float32 browser weight bundle.
- Added SHA-256, offsets, shapes, and byte-count validation.
- Added Vercel configuration that skips Python installation.
- Added safe samples, optional ground-truth scoring, downloads, disclaimer, and limitations.
- Added browser-bundle unit tests and GitHub Actions validation.

## Important positioning correction

The project must remain labeled as a synthetic medical-imaging proof of concept. The recorded scores do not establish clinical performance.

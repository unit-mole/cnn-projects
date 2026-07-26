# 02 — Object Detection Using CNN

A Vercel-ready browser application for a custom single-object handwritten-digit detector. The original Keras weights are converted to a compact browser bundle and inference runs locally with TensorFlow.js.

## Live demo

Add the Vercel URL after deployment.

## Scope

- One handwritten digit per image
- Classes 0–9
- One normalized XYXY bounding box
- Synthetic 64×64 MNIST canvases
- No NMS because the model predicts one box

## Results

| Metric | Result |
|---|---:|
| CNN class accuracy | 93.23% |
| Top-3 accuracy | 98.90% |
| CNN mean IoU | 0.349 |
| Baseline mean IoU | 0.368 |

## Local test

```cmd
python -m http.server 8000
```

Open `http://localhost:8000`.

## Vercel settings

- Framework Preset: Other
- Root Directory: `02-object-detection-using-cnn`
- Build Command: empty
- Output Directory: empty
- Install Command: empty

## Responsible use

Educational portfolio demonstration only. Do not use for surveillance, safety-critical, medical, legal, security, or production decisions.

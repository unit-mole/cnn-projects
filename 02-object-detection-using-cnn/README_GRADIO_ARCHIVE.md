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

# CNN Handwritten Digit Object Detector

Upload one handwritten digit and receive a predicted digit class, confidence
score, normalized bounding box, and annotated image.

This Space uses a compact custom CNN trained on synthetic 64×64 canvases built
from MNIST. It is a single-object educational detector—not a general detector
for photographs or multiple objects.

## Responsible use

The model may miss a digit, predict the wrong class, or localize the digit
poorly. Do not use it as the sole basis for safety-critical, surveillance,
medical, security, legal, or production decisions. Do not upload private,
confidential, copyrighted, or personally identifiable images.

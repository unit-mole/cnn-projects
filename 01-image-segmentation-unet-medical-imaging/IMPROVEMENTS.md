# Improvements

## Completed

- Converted the notebook-centric experiment into modular Python code and tests.
- Added a local Gradio reference app.
- Added a production-style Vercel static website.
- Added TensorFlow.js browser inference with the trained U-Net weights.
- Added deterministic Keras 3 weight export without requiring TensorFlow.
- Added probability map, binary mask, overlay, optional Dice/IoU, and mask download.
- Added responsible medical disclaimer and synthetic-data framing.
- Added web-asset validation to CI.

## Recommended next steps

1. Train and evaluate on an appropriately licensed, de-identified public medical segmentation dataset.
2. Use patient-level non-overlapping splits and document data provenance.
3. Add modality-specific preprocessing and clinically meaningful augmentations.
4. Benchmark calibration, boundary metrics, and external-domain robustness.
5. Quantize the browser model only after measuring parity and segmentation quality.
6. Replace the placeholder Vercel link after production deployment.

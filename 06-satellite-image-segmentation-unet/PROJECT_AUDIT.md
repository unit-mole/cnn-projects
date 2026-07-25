# Project audit

## Source artifacts reviewed

- original 178-cell notebook,
- exported `.keras` model,
- exported JSON metrics.

## What the original project did correctly

- deterministic seed and repeatable dataset generation,
- explicit train/validation/test split,
- threshold baseline,
- compact U-Net with skip connections,
- Dice and IoU metrics,
- model saving and reload check,
- best/worst visual examples,
- threshold sweep and area diagnostics.

## Gaps found

1. **Dataset-description mismatch:** the introduction referred to SpaceNet/DeepGlobe-style sources, but the executable code generated synthetic data.
2. **Overstated applicability:** near-perfect synthetic metrics were described as high-precision geospatial performance.
3. **Baseline result not interpreted:** the threshold baseline slightly outperformed U-Net, revealing that the generator made the task too easy.
4. **No production inference package:** preprocessing and model loading lived only in the notebook.
5. **No deployable app:** no Gradio or Hugging Face entry point.
6. **No metadata contract:** image size, threshold, classes, and limitations were not packaged with the model.
7. **No tests or CI:** preprocessing, artifact integrity, imports, and inference were not automatically validated.
8. **No responsible-use communication:** geospatial limitations and upload safety were missing.
9. **No monorepo integration guide:** workflow placement and root README updates were absent.

## Corrections made

- reframed the task as binary synthetic urban-structure segmentation,
- preserved the supplied model and exact metrics,
- created model metadata and safe synthetic samples,
- split code into reusable preprocessing, metrics, model, inference, and visualization modules,
- added a polished Gradio application and Hugging Face Space metadata,
- created cleaned local and Kaggle notebooks while archiving the original,
- extracted real notebook figures into `outputs/figures`,
- added tests, validation scripts, Docker, requirements, local run scripts, and CI,
- documented the baseline-vs-U-Net result honestly,
- added responsible-use and data-safety guidance.

## Residual limitation

The supplied model has not been retrained on a real satellite dataset. That is the largest remaining technical gap and should be the first major future improvement.


## Vercel + TensorFlow.js deployment audit

- Added a static `index.html` entrypoint and responsive recruiter-facing interface.
- Exported 471,553 inference parameters into a 1,886,212-byte browser weight file.
- Added both conventional LayersModel loading and a deterministic architecture/weight fallback.
- Added browser-side preprocessing, mask thresholding, probability rendering, overlays, optional overlap metrics, and PNG download.
- Added `vercel.json`, `.vercelignore`, local web launchers, export tooling, validation, tests, and CI checks.
- Browser uploads remain on-device; no image API or server-side storage is used.

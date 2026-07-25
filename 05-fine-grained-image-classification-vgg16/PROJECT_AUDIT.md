# Project Audit

## Supplied assets reviewed

- Complete Jupyter notebook
- Saved Keras VGG16 model
- Metrics JSON
- Detailed project specification

## Actual experiment identified

- Dataset: CIFAR-10 filtered to cats and dogs
- Task: balanced binary image classification
- Native size: 32×32 RGB
- Model: frozen ImageNet VGG16 backbone and dense classifier
- Test accuracy: 86.95%
- Baseline test accuracy: 57.20%
- Browser target: Vercel + TensorFlow.js

## Issues found and addressed

| Original condition | Improvement |
|---|---|
| Notebook-centric implementation | Split into reusable `src/`, `scripts/`, `tests/`, and `web/` modules |
| Training-only augmentation embedded in model | Created flattened browser inference model |
| Browser preprocessing absent | Reproduced VGG16 preprocessing in JavaScript |
| No TensorFlow.js/Vercel app | Added full static responsive frontend and deployment config |
| No lightweight CI | Added project-scoped workflow without retraining |
| Top-2 result could be overstated | Documented that it is trivial for two classes |
| Generic fine-grained framing could overstate scope | Documented actual binary cat-vs-dog dataset |
| High-confidence mistakes not surfaced | Added explicit error-analysis and responsible-use sections |
| Dataset redistribution risk | Kept full dataset out of Git and added dataset card |

## Validation completed during packaging

- Source model loaded successfully.
- Browser inference model built from supplied learned weights.
- Source/browser maximum absolute output difference: `0.0` on the packaging parity batch.
- TensorFlow.js topology and manifest generated.
- Sixteen model shards validated against manifest byte counts.
- Python source compiled.
- Lightweight test suite and artifact validators included.

## Remaining deployment verification

A real browser runtime smoke test must still be completed after serving or deploying the files. This environment validated files and parity but did not execute the final TensorFlow.js graph in a browser.

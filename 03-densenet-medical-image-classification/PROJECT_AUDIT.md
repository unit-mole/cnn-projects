# Project Audit

## Scope

This audit covers the executable notebook, bundled Keras artifact, browser deployment model, TensorFlow.js interface, GitHub Pages workflow, documentation, and public-use disclosures.

## Confirmed model facts

- Original model input: `28 × 28 × 3`, scaled to `[0, 1]`.
- Internal training-model resize: `96 × 96`.
- Backbone: DenseNet121 with ImageNet initialization.
- Backbone state in attached experiment: frozen.
- Head: global average pooling, Dense(256), batch normalization, dropout, Dense(2, softmax).
- Browser inference model input: preprocessed `96 × 96 × 3`.
- Browser deployment format: TensorFlow.js LayersModel generated from HDF5.

## Critical dataset finding

The notebook narrative references pneumonia and chest X-rays, but the executed code loads Fashion-MNIST and converts source classes `2`, `4`, and `6` into a synthetic positive class. Therefore:

- The bundled artifact is not a clinical chest-X-ray classifier.
- Public labels remain `normal_like` and `pneumonia_like`.
- Recorded metrics are described only as synthetic-proxy results.
- The web interface and README include prominent medical and data-safety disclaimers.

## Browser-model preparation

The original model contains resize, augmentation, and DenseNet preprocessing layers. For browser compatibility:

1. Training-only augmentation is removed from the exported inference model.
2. The DenseNet backbone graph is flattened into the browser HDF5 model.
3. Deterministic preprocessing is implemented in JavaScript.
4. An equivalence test compares original and browser-model predictions before export.
5. The model is converted to TensorFlow.js during GitHub Actions.

## Deployment safety

The Project 03 workflow publishes to:

```text
03-densenet-medical-image-classification/
```

inside the existing `gh-pages` branch with `keep_files: true`. This is designed to preserve the existing root website and Project 07 subfolder.

## Validation included

- Python project-structure validation
- Keras archive validation
- Unit tests
- Browser source validation
- JavaScript syntax validation
- Generated TensorFlow.js manifest and shard validation
- Explicit no-retraining CI design

## Remaining limitations

- The converter executes in GitHub Actions rather than committing generated model shards to `main`.
- DenseNet121 is larger than an intentionally mobile-first architecture, so first-load time depends on network and browser hardware.
- Browser preprocessing can differ slightly across rendering engines.
- The synthetic proxy is not suitable for clinical interpretation.
- A real medical project requires licensed data, patient-level splitting, external validation, calibration, subgroup analysis, and clinical review.

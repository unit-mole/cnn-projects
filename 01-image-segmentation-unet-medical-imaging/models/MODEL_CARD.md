# Model Card — Compact Synthetic Medical U-Net

## Model details

- Architecture: compact two-level U-Net
- Framework: TensorFlow / Keras
- Input: `(64, 64, 1)` normalized grayscale image
- Output: `(64, 64, 1)` sigmoid probability map
- Parameters: 470,977
- Artifact: `unet_medical.keras`
- SHA-256: `f1627ed203bb80d22365b9e77d0a9c25ef569889e2aa265a7807cec9d8847cbc`

## Intended use

Educational demonstration of binary semantic-segmentation engineering, model packaging, inference, evaluation, and deployment.

## Out-of-scope use

- diagnosis, treatment, triage, prognosis, or patient management
- use on private or identifiable medical images
- claims about tumor, lesion, organ, tissue, or cell segmentation
- clinical or regulatory deployment

## Training data

2,500 deterministic synthetic grayscale images with aligned elliptical binary masks. No real patients or external medical dataset were used.

## Evaluation

The recorded soft Dice of 0.9977 and soft IoU of 0.9954 apply only to the synthetic test split. The intensity construction makes the task easier than real medical segmentation.

## Limitations

See `model_metadata.json` and the project README for complete limitations and responsible-use guidance.

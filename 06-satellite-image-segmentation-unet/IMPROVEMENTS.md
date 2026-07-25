# Improvement roadmap

## Highest priority

- replace procedural data with a licensed real urban-mapping dataset,
- use geographic scene-level train/validation/test splits,
- report class balance by pixel and by scene,
- evaluate on a separate city, sensor, or acquisition period.

## Modeling

- compare U-Net with a plain CNN, U-Net++, DeepLabV3+, and a pretrained encoder,
- combine binary cross-entropy with Dice or focal loss,
- add boundary-aware metrics,
- calibrate probabilities and expose uncertainty,
- tune the decision threshold only on validation data.

## Data

- add synchronized flips, rotations, crops, brightness, and contrast changes,
- support multi-class masks,
- document RGB versus multispectral band handling,
- validate CRS/resolution outside the neural-network input pipeline,
- remove or anonymize sensitive metadata from public samples.

## Deployment

- export a smaller inference model when needed,
- add cached model download from the Hugging Face Model Hub,
- add app analytics that do not retain uploaded images,
- add end-to-end browser tests.


## Browser deployment roadmap

- Add browser weight quantization after verifying metric parity.
- Add Playwright tests that load a sample, run inference, and compare summary statistics.
- Evaluate WebGPU when browser support is sufficiently consistent.
- Add a side-by-side threshold-baseline browser comparison for educational value.

# Project Audit

## Source assets reviewed

- Training notebook: `Image_Classification_ResNet_Residual_Learning_FULL_ELITE(1).ipynb`
- Metrics: `metrics(3).json`
- Trained model: `resnet50_cifar100.keras`

## Confirmed objective

The project is a **100-class CIFAR-100 image classifier** using a frozen ImageNet-pretrained ResNet50 backbone and a custom softmax head. It is multi-class—not binary—and the browser demo must preserve the notebook's ResNet preprocessing.

## Confirmed results

- Baseline validation accuracy: `0.1262`
- Baseline test accuracy: `0.1257`
- ResNet validation accuracy: `0.6707000136375427`
- ResNet test accuracy: `0.6708999872207642`
- Top-5 accuracy: `0.912`
- Seed: `42`

## Improvements made

1. Separated reusable loading, preprocessing, model, training, evaluation, inference, visualization, and conversion modules.
2. Preserved the attached notebook as the authoritative experiment record.
3. Added exact CIFAR-100 fine-label metadata and browser-safe top-k inference.
4. Flattened the deployable inference model so browser preprocessing occurs outside the model graph.
5. Added TensorFlow.js assets, a responsive static UI, input validation, privacy messaging, and responsible-use guidance.
6. Added lightweight tests and structural model-manifest validation.
7. Added GitHub Actions CI and Pages publication from the nested `web/` directory.
8. Avoided inventing unavailable precision, recall, F1, ROC, or confusion-matrix results.

## Known constraints

- Source images are only 32×32, which limits fine visual detail.
- The frozen-backbone experiment can be improved through controlled fine-tuning.
- ResNet50 is a large browser model and has a noticeable initial download and warm-up cost.
- Normal photographs may be out of distribution relative to CIFAR-100.

## TensorFlow.js bundle validation scope

The included browser bundle was generated from the flattened HDF5 inference model using the repository's deterministic fallback serializer. Its topology, manifest, shard presence, byte ordering, and weight-stream equality were validated. The flattened Python model produced exactly the same probabilities as the supplied `.keras` model on the equivalence test recorded in `outputs/metrics/browser_model_equivalence.json`. The official `tensorflowjs_converter` remains the recommended regeneration path and is implemented in `scripts/convert_to_tfjs.py`.

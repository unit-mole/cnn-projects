# Fine-Grained Image Classification using VGG16

[![CI](https://github.com/ADD_GITHUB_USERNAME/cnn-projects/actions/workflows/05-fine-grained-image-classification-vgg16.yml/badge.svg)](https://github.com/ADD_GITHUB_USERNAME/cnn-projects/actions/workflows/05-fine-grained-image-classification-vgg16.yml)
[![Vercel](https://img.shields.io/badge/live%20demo-Vercel-black)](ADD_VERCEL_URL)
[![TensorFlow.js](https://img.shields.io/badge/inference-TensorFlow.js-ff6f00)](https://www.tensorflow.org/js)

A complete CNN portfolio project that converts the supplied VGG16 transfer-learning notebook and trained Keras model into a modular Python package, tested TensorFlow.js model bundle, responsive static browser app, Vercel deployment, and optional Gradio fallback.

> **Responsible-use notice:** This project is for educational and portfolio demonstration only. It may confuse visually similar categories, especially when images are unclear, cropped, low-quality, or outside the training distribution. Do not use it as the sole basis for medical, legal, safety-critical, security, hiring, insurance, financial, or production decisions. Do not upload private, sensitive, confidential, copyrighted, or personally identifiable images. Predictions are machine-learning outputs, not guaranteed truth.

## Live links

- **Vercel browser demo:** `ADD_VERCEL_URL`
- **Optional Hugging Face fallback:** `ADD_HUGGING_FACE_SPACE_URL`
- **Training notebook:** `ADD_COLAB_OR_KAGGLE_URL`
- **GitHub repository:** `ADD_GITHUB_PROJECT_URL`

## Project objective

The practical question is:

> Given an image, can a pretrained CNN distinguish between two visually similar animal categories and communicate uncertainty responsibly?

The supplied implementation is specifically a **binary CIFAR-10 cat-versus-dog classifier**. It is a useful fine-grained-style discrimination problem because both classes share visual features such as fur, four-legged body structure, eyes, ears, and household backgrounds. It is not presented as a many-species or breed-level dataset; the documentation deliberately preserves the actual scope of the uploaded files.

## Why fine-grained classification is harder

Standard image classification often separates visibly different categories. Fine-grained classification separates visually similar categories and must rely on subtle texture, shape, color pattern, object-part, or small-defect cues. This makes probability ranking, confusion analysis, visual error review, and limitation reporting especially important.

## Actual dataset

The notebook filters CIFAR-10 to source class `3` (cat) and source class `5` (dog).

| Property | Value |
|---|---|
| Source | CIFAR-10, filtered |
| Native format | RGB arrays |
| Native image size | 32×32×3 |
| Classes | `cat`, `dog` |
| Training images | 8,000 |
| Validation images | 2,000 |
| Test images | 2,000 |
| Training distribution | 4,005 cat / 3,995 dog |
| Test distribution | 1,000 cat / 1,000 dog |
| Task | Binary image classification |

The classes are effectively balanced, so no aggressive oversampling is needed. The modular loader nevertheless supports a reproducible stratified split and includes class-weight calculation for future datasets.

## Results

| Model | Validation accuracy | Test accuracy | Macro F1 | Notes |
|---|---:|---:|---:|---|
| Flattened pixels + logistic regression | 56.65% | 57.20% | — | Simple baseline |
| **VGG16 transfer learning** | **85.55%** | **86.95%** | **86.95%** | Frozen ImageNet backbone |

Per-class test results:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cat | 87.59% | 86.10% | 86.84% | 1,000 |
| Dog | 86.33% | 87.80% | 87.06% | 1,000 |

Confusion matrix:

| Actual \ Predicted | Cat | Dog |
|---|---:|---:|
| Cat | 861 | 139 |
| Dog | 122 | 878 |

The uploaded metrics report top-2 accuracy of 100%. Because this experiment has exactly two classes, top-2 always includes every class and is not a meaningful ranking achievement. The project therefore highlights test accuracy, macro F1, per-class recall, and confusion patterns instead.

## VGG16 architecture

VGG16 is a convolutional neural network known for repeated small 3×3 filters. Early layers learn edges and textures; deeper layers combine them into more complex object-part patterns. Transfer learning reuses ImageNet features and adapts them to the custom cat-versus-dog task.

```text
32×32 RGB image
    ↓
Safe training augmentation
    ↓
Resize to 96×96
    ↓
VGG16 ImageNet preprocessing
    ↓
Frozen VGG16 convolutional backbone
    ↓
Flatten
    ↓
Dense 256 + Batch Normalization + Dropout 0.50
    ↓
Dense 128 + Dropout 0.40
    ↓
2-unit Softmax output
```

| Model property | Value |
|---|---|
| Total parameters | 15,928,770 |
| Trainable parameters | 1,213,570 |
| Frozen parameters | 14,715,200 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Categorical cross-entropy |
| Requested epochs | 15 |
| Batch size | 128 |
| Backbone | ImageNet VGG16, frozen |

The original experiment did not unfreeze the backbone. An optional conservative block-5 fine-tuning helper is included for future experimentation, but no unmeasured fine-tuning result is claimed.

## Image preprocessing

Training/Python source model:

1. Decode as RGB.
2. Resize to 32×32.
3. Normalize pixels to `[0, 1]`.
4. Inside the model, resize to 96×96.
5. Multiply back to `[0, 255]` and apply `tf.keras.applications.vgg16.preprocess_input`.

Browser model:

1. Decode uploaded image as RGB with `tf.browser.fromPixels`.
2. Resize to 32×32, then 96×96 with bilinear interpolation.
3. Reverse RGB to BGR.
4. Subtract VGG16 ImageNet means `[103.939, 116.779, 123.68]`.
5. Run the flattened TensorFlow.js model.

The same class order (`cat`, then `dog`) and preprocessing metadata are saved in `models/model_metadata.json` and `web/metadata.json`.

## Data augmentation

The notebook uses deliberately restrained augmentation:

- horizontal flip,
- small rotation (`0.06` fraction),
- small zoom (`10%`).

These transformations provide variation without heavy blur, crop, or color distortion that could remove the subtle visual signals needed for similar-class discrimination.

## Evaluation approach

The project includes:

- accuracy,
- macro and weighted F1,
- per-class precision, recall, and F1,
- confusion matrix,
- training/validation curves,
- correct and misclassified prediction galleries,
- high-confidence wrong examples,
- close top-two probability warning,
- baseline comparison.

The notebook includes very confident mistakes, including several wrong predictions above 99% confidence. This is why the interface treats confidence as model output—not calibrated certainty—and includes a responsible-use warning.

## TensorFlow.js conversion

The original Keras model embeds augmentation and Keras 3 preprocessing operations. To make deployment more robust:

1. The learned VGG16 backbone and dense-head weights were transferred into a browser-only inference graph.
2. Training-only augmentation and preprocessing layers were removed from the exported graph.
3. Equivalent preprocessing was implemented explicitly in `web/app.js`.
4. Source-model and browser-model predictions were checked in Python; the maximum observed absolute difference was `0.0` for the packaging validation batch.
5. The TensorFlow.js bundle was written as `model.json` plus sixteen weight shards.

The browser bundle is about 60.8 MiB, so the first load may be slower than a lightweight MobileNet demo. The included Gradio app is the fallback when full-browser VGG16 performance is unsuitable.

## Browser prediction output

The static app displays:

- input image preview,
- predicted class,
- confidence score,
- all available class probabilities,
- a warning when the top-two probability gap is below `0.15`,
- prediction interpretation,
- model card and limitations.

The interface says **class probabilities**, not “top five,” because this actual model has only two output classes.

## Project structure

```text
05-fine-grained-image-classification-vgg16/
├── app/                         # Optional Gradio fallback
├── archive/
├── data/
│   ├── README_data.md
│   └── sample_images/
├── images/
├── models/
│   ├── vgg16_fine_grained_classification_model.keras
│   ├── vgg16_browser_inference.keras
│   ├── class_mapping.json
│   ├── model_metadata.json
│   └── tfjs_model/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── web/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── metadata.json
│   ├── sample_images/
│   └── tfjs_model/
├── app.py
├── Dockerfile
├── FILE_MANIFEST.csv
├── package.json
├── requirements.txt
├── vercel.json
└── README_VERCEL.md
```

The project-specific GitHub Actions workflow is correctly placed at the monorepo root:

```text
cnn-projects/.github/workflows/05-fine-grained-image-classification-vgg16.yml
```

## Run the browser demo locally

```bash
git clone ADD_GITHUB_REPOSITORY_URL
cd cnn-projects/05-fine-grained-image-classification-vgg16
python scripts/validate_project.py
python scripts/run_local_web_server.py --port 8000
```

Open `http://127.0.0.1:8000`. Do not open `web/index.html` directly with a `file://` URL because model-shard requests require an HTTP server.

Node alternative:

```bash
npm install
npm run validate
npm run dev
```

## Run Python inference

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -c "from src.classification_pipeline import classify_image; print(classify_image('data/sample_images/cat_sample.png'))"
```

## Retrain and evaluate

```bash
python scripts/train_model.py
python scripts/evaluate_model.py
```

VGG16 training is compute-intensive. A GPU-enabled Colab/Kaggle environment is preferable to GitHub Actions or a small local CPU.

## Reconvert to TensorFlow.js

```bash
python scripts/export_model.py
python scripts/convert_to_tfjs.py
python scripts/validate_tfjs_artifacts.py
```

The packaged TensorFlow.js bundle is already present; reconversion is needed only after retraining or changing the architecture.

## Deploy on Vercel

See [`README_VERCEL.md`](README_VERCEL.md). Configure the Vercel project root as:

```text
05-fine-grained-image-classification-vgg16
```

The included `vercel.json` publishes the `web/` directory.

## Optional Hugging Face fallback

See [`README_HUGGINGFACE.md`](README_HUGGINGFACE.md). The fallback loads the full Keras model on a Python backend and serves a Gradio interface.

## Error analysis and limitations

Observed error themes include:

- low native CIFAR-10 resolution,
- visually similar ears, fur, face shapes, and backgrounds,
- partial or cropped animals,
- background bias,
- high-confidence wrong predictions,
- domain shift from arbitrary web or phone images,
- no probability calibration study,
- binary rather than many-class fine-grained labeling,
- a large browser model and slower initial load.

## Future improvements

1. Add a true multi-breed or product-variant dataset with verified redistribution rights.
2. Fine-tune VGG16 block 5 with a low learning rate and report measured results.
3. Add temperature scaling or another probability-calibration analysis.
4. Compare VGG16 with MobileNetV2 and EfficientNet under the same split.
5. Quantize a browser bundle and measure accuracy, download size, and latency.
6. Add Grad-CAM only after validating the implementation against the exported model.
7. Add automated browser smoke tests with Playwright in a suitable CI environment.

## Portfolio positioning

**One-line description**

> Built and deployed a VGG16 transfer-learning image classifier with TensorFlow.js browser inference, confusion analysis, responsible uncertainty messaging, and Vercel-ready static hosting.

**Pinned-repository description**

> End-to-end CNN portfolio with VGG16 transfer learning, CIFAR-10 cat/dog classification, 86.95% test accuracy, TensorFlow.js conversion, browser-side predictions, CI validation, and Vercel deployment.

**Skills demonstrated**

CNN modeling · transfer learning · VGG16 · image preprocessing · safe augmentation · baseline comparison · class-wise evaluation · confusion/error analysis · Keras serialization · TensorFlow.js · JavaScript · static deployment · Vercel · GitHub Actions

For a Quality Data Scientist, the same workflow maps naturally to product-variant recognition, defect-type classification, visual inspection analytics, and image-assisted quality review.

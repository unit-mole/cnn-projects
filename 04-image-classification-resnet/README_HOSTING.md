# GitHub Pages Deployment Guide

## Recommended deployment method

Use the repository workflow at:

```text
.github/workflows/04-image-classification-resnet.yml
```

It uploads only `04-image-classification-resnet/web/` as the Pages artifact. This is cleaner than copying files into a root `docs/` directory and avoids maintaining duplicate site source.

## One-time repository setting

1. Push the project and workflow to GitHub.
2. Open **Settings → Pages**.
3. Under **Build and deployment**, choose **GitHub Actions** as the source.
4. Push to `main` or run the workflow manually.
5. Open the deployment URL shown in the workflow summary.

## Model conversion

Install the full requirements, export the browser-compatible inference model, and convert it:

```bash
python scripts/export_model.py --keras-model models/resnet50_cifar100.keras --output models/resnet50_cifar100_browser.h5
python scripts/convert_to_tfjs.py --keras-model models/resnet50_cifar100_browser.h5 --output web/tfjs_model
```

The preferred source for conversion is the browser-inference HDF5 model whose input is already a 96×96 preprocessed tensor. `web/app.js` performs resizing and ImageNet ResNet preprocessing before prediction.

The official command-line equivalent is:

```bash
tensorflowjs_converter --input_format=keras   models/resnet50_cifar100_browser.h5   web/tfjs_model
```

## Test locally

Browsers usually block model fetches from a `file://` page. Serve the folder:

```bash
python scripts/run_local_web_server.py --port 8000
```

Then open `http://localhost:8000` and confirm:

- the status changes to **Model ready**;
- a sample or uploaded image is previewed;
- prediction returns 100 class probabilities;
- the result and top-three bars are visible;
- the browser console has no failed shard requests.

## Large-model guidance

The full ResNet50 browser weights are substantial. The current bundle uses multiple shards so each request remains manageable and cacheable. Before publishing:

- retain all `.bin` files referenced by `model.json`;
- do not rename shards without updating the manifest;
- avoid Git LFS pointer files inside the deployed `web/` folder unless Pages resolves them to actual binary content;
- test the production URL after deployment;
- consider post-training quantization only after comparing accuracy and browser latency.

## Troubleshooting

### `Failed to fetch model.json`

Confirm the page is served over HTTP(S), the path is `./tfjs_model/model.json`, and the filename case matches exactly.

### A shard returns 404

Run:

```bash
python scripts/validate_project.py --project-dir .
```

The validator checks every path in the weight manifest.

### Model loads but prediction fails

Check that `metadata.json` has browser input size 96×96, that `app.js` converts RGB to BGR, and that the loaded model output dimension is 100.

### The first prediction is slow

The first run includes model initialization and backend warm-up. Subsequent predictions are typically faster. Keep the model status visible and avoid triggering multiple predictions simultaneously.

### GitHub Pages shows the repository README instead of the app

Make sure the Pages source is **GitHub Actions**, not a branch folder, and confirm the workflow uploads `04-image-classification-resnet/web`.

## Share the final link

Add the deployed URL to:

- this project's `README.md`;
- the root `cnn-projects/README.md`;
- the repository website field;
- your portfolio project card and résumé.

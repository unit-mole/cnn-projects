# Vercel + TensorFlow.js Deployment Guide

Project 01 now includes a static browser application that runs the trained compact U-Net with **TensorFlow.js 4.22.0**. Vercel only serves HTML, CSS, JavaScript, safe synthetic examples, and the exported inference weights. The visitor's browser performs the segmentation.

## Deployment architecture

```text
Vercel static hosting
        │
        └── web/
            ├── index.html
            ├── style.css
            ├── app.js
            ├── metadata.json
            ├── sample_images/
            ├── sample_masks/
            └── tfjs_model/
                ├── model.json
                ├── weights_manifest.json
                ├── weights.bin
                └── model_metadata.json

Browser
  └── TensorFlow.js → U-Net → probability map → mask → overlay
```

No Python backend, API route, database, GPU server, secret, or environment variable is required.

## Validate locally

From `01-image-segmentation-unet-medical-imaging`:

```bat
run_vercel_local.bat
```

Or:

```bash
node scripts/validate-web.mjs
python -m http.server 8000 --directory web
```

Open `http://127.0.0.1:8000`. Do not double-click `web/index.html`; browser security blocks model-file fetches when a page is opened through `file://`.

A successful local test should show **Model ready**, allow a safe sample to be selected, and generate the probability map, binary mask, overlay, Dice, and IoU.

## Vercel dashboard settings

1. Sign in to Vercel using the GitHub account connected to `unit-mole/cnn-projects`.
2. Choose **Add New → Project**.
3. Import `unit-mole/cnn-projects`.
4. Project name: `medical-image-segmentation-unet`.
5. Root Directory: `01-image-segmentation-unet-medical-imaging`.
6. Framework Preset: **Other**.
7. Keep the configuration read from `vercel.json`:
   - Install Command: empty
   - Build Command: `npm run build`
   - Output Directory: `web`
8. No environment variables are required.
9. Select **Deploy**.
10. Open the production URL and confirm that the status becomes **Model ready**.

The empty `installCommand` is intentional. It prevents Vercel from detecting `requirements.txt` and trying to install TensorFlow/Python packages for a deployment that is entirely static.

## Model export

The source model is a Keras 3 `.keras` archive. To regenerate the browser bundle without installing TensorFlow:

```bash
python -m pip install h5py numpy
python scripts/export_tfjs_assets.py
python scripts/validate_tfjs_export.py
npm run build
```

The exporter reads only the 22 Conv2D inference tensors, writes 470,977 float32 parameters to `web/tfjs_model/weights.bin`, and records exact offsets, shapes, byte counts, and SHA-256 in `weights_manifest.json`.

## Troubleshooting

### Vercel tries to install Python
Confirm that `vercel.json` is in the Project 01 root and contains `"installCommand": ""`. Also confirm the Vercel Root Directory is Project 01—not the monorepo root.

### Site returns 404
Confirm `vercel.json` uses `"outputDirectory": "web"` and that `web/index.html` exists.

### Page loads but says Model error
Open browser developer tools → Network. Confirm these return HTTP 200:

```text
/tfjs_model/model.json
/tfjs_model/weights_manifest.json
/tfjs_model/weights.bin
/tfjs_model/model_metadata.json
```

Also ensure a browser extension or company network filter is not blocking the TensorFlow.js CDN.

### Local page cannot fetch weights
Use an HTTP server; do not use a `file://` URL.

### First prediction is slower
TensorFlow.js warms up WebGL on the first model execution. Later predictions are normally faster. The app falls back to the available TensorFlow.js backend.

## After deployment

Replace `ADD_VERCEL_URL` in:

```text
01-image-segmentation-unet-medical-imaging/README.md
cnn-projects/README.md
```

Recommended label: **Live Browser Demo — Vercel + TensorFlow.js**.

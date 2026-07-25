# Vercel + TensorFlow.js Deployment Guide

This project includes a static browser application that runs the trained compact U-Net with **TensorFlow.js 4.22.0**. Vercel serves the HTML, CSS, JavaScript, sample tiles, `model.json`, and `weights.bin`; all segmentation inference runs on the visitor's device.

## Deployment architecture

```text
Vercel static hosting
        │
        ├── index.html
        ├── assets/css/styles.css
        ├── assets/js/app.js
        ├── assets/samples/
        └── tfjs_model/
            ├── model.json
            ├── weights_manifest.json
            ├── weights.bin
            └── model_metadata.json

Browser
  └── TensorFlow.js → U-Net inference → mask, overlay, probability map
```

No Python API, serverless function, database, environment variable, or GPU server is required.

## Validate locally before deployment

From `06-satellite-image-segmentation-unet`:

### Windows

```bat
run_vercel_local.bat
```

### macOS or Linux

```bash
./run_vercel_local.sh
```

Then open:

```text
http://localhost:8000
```

Do not open `index.html` directly using a `file://` URL because browsers block model-file fetches from local files. Use the local HTTP server.

Run static validation:

```bash
python scripts/validate_tfjs_export.py
node --check assets/js/app.js
```

## Push the update to GitHub

Open Git Bash or the VS Code terminal in the root `cnn-projects` repository:

```bash
git add .
git commit -m "Add Vercel TensorFlow.js deployment for Project 06"
git pull --rebase origin main
git push origin main
```

## Deploy through the Vercel dashboard

1. Sign in to Vercel using the GitHub account connected to `unit-mole/cnn-projects`.
2. Select **Add New → Project**.
3. Find `cnn-projects` and select **Import**. The same Git repository can be imported as a separate Vercel project for this numbered folder.
4. Set the project name, for example:

   ```text
   satellite-image-segmentation-unet
   ```

5. Under **Root Directory**, select:

   ```text
   06-satellite-image-segmentation-unet
   ```

6. Use these build settings:

   | Setting | Value |
   |---|---|
   | Framework Preset | Other |
   | Build Command | Leave empty |
   | Output Directory | Leave empty |
   | Install Command | Leave empty |
   | Environment Variables | None required |

7. Select **Deploy**.
8. After the deployment finishes, open the generated Vercel URL.
9. Confirm that the status changes to **Model ready**, select a sample tile, and generate a segmentation.
10. In **Project Settings → Domains**, optionally assign a cleaner Vercel domain.

## Monorepo setting that matters

The selected Vercel Root Directory must be the numbered project folder—not the repository root and not a nested web folder:

```text
cnn-projects/06-satellite-image-segmentation-unet
```

The `vercel.json` file is already inside that folder, which makes it the configuration root for this deployment.

## Automatic deployments

Once the Vercel project is connected to the `main` branch:

- every relevant GitHub push creates a new deployment;
- pull requests can receive preview deployments;
- a successful production deployment updates the public project URL.

## Add the live URL to the portfolio

After deployment, replace the placeholder in:

```text
06-satellite-image-segmentation-unet/README.md
cnn-projects/README.md
06-satellite-image-segmentation-unet/models/model_metadata.json
```

Recommended link label:

```text
Live Browser Demo — Vercel + TensorFlow.js
```

## Browser-model export

The tracked browser model contains only inference weights and is approximately 1.8 MB. To regenerate it from the supplied Keras archive:

```bash
pip install h5py numpy
python scripts/export_tfjs_assets.py
python scripts/validate_tfjs_export.py
```

The web app first attempts `tf.loadLayersModel`. It also includes a deterministic fallback that reconstructs the compact U-Net and loads the same weights through `tf.io.loadWeights`, which makes the demo more resilient to Keras/TensorFlow.js serialization-version differences.

## Troubleshooting

### The page deploys but shows `Model error`

Confirm these files exist in GitHub:

```text
tfjs_model/model.json
tfjs_model/weights_manifest.json
tfjs_model/weights.bin
tfjs_model/model_metadata.json
```

Also confirm that ad blockers, network filters, or browser extensions are not blocking the TensorFlow.js CDN.

### The page shows a 404

Verify the Vercel Root Directory is exactly:

```text
06-satellite-image-segmentation-unet
```

and confirm that `index.html` is located directly inside that folder.

### Inference is slow

The app prefers WebGL and falls back to CPU. The first prediction includes browser/model warmup costs; later predictions are normally faster. Older devices or browsers without WebGL may take longer.

### `index.html` works locally but the model does not load

Run a local web server rather than opening the HTML file directly:

```bash
python -m http.server 8000
```

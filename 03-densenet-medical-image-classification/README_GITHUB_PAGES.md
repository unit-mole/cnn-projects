# GitHub Pages Deployment Guide

This project publishes a TensorFlow.js browser demo into the existing `cnn-projects` GitHub Pages site without replacing the root page or Project 07.

## Final URL

```text
https://unit-mole.github.io/cnn-projects/03-densenet-medical-image-classification/
```

## Files used for deployment

```text
.github/workflows/03-densenet-medical-image-classification.yml
03-densenet-medical-image-classification/web/
03-densenet-medical-image-classification/models/densenet121_medical_browser.h5
03-densenet-medical-image-classification/scripts/convert_browser_model.py
03-densenet-medical-image-classification/requirements-pages.txt
```

## What the workflow does

1. Validates the project structure.
2. Runs unit tests.
3. Checks JavaScript syntax.
4. Installs the TensorFlow.js Python converter on GitHub Actions.
5. Converts the inference-only HDF5 model into `model.json` and binary weight shards.
6. Validates the generated model manifest and shards.
7. Publishes the `web/` folder to the `gh-pages` branch under:

```text
03-densenet-medical-image-classification/
```

The workflow uses `destination_dir` and `keep_files: true`, so existing deployed pages remain in place.

## Repository setting

Because this repository already hosts Project 04 and Project 07, GitHub Pages is likely configured correctly. Confirm once:

1. Open the GitHub repository.
2. Select **Settings**.
3. Select **Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Select branch **gh-pages**.
6. Select folder **/(root)**.
7. Save.

Do not switch to a workflow that deploys a single full-site artifact unless all existing project pages are included in that same artifact, because a full-site deployment can replace the current site.

## First deployment

After pushing the project:

1. Open **Actions**.
2. Select `03-densenet-medical-image-classification.yml`.
3. Wait for both jobs to become green:
   - `Lightweight project validation`
   - `Publish browser demo to gh-pages`
4. Wait approximately one to three minutes for GitHub Pages caching.
5. Open the final URL.
6. Hard refresh once with `Ctrl + F5` if an older page is cached.

## Manual rerun

The workflow supports `workflow_dispatch`:

1. Open **Actions**.
2. Select the Project 03 workflow.
3. Select **Run workflow**.
4. Choose `main` and run it.

## Local browser test

```bash
cd 03-densenet-medical-image-classification
python -m venv .venv-pages
```

Windows:

```bat
.venv-pages\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-pages.txt
python scripts/convert_browser_model.py
python -m http.server 8000 --directory web
```

Then open:

```text
http://localhost:8000
```

## Troubleshooting

### The workflow passes validation but conversion fails

Review the `Convert Keras browser model to TensorFlow.js` log. Confirm:

```text
models/densenet121_medical_browser.h5
requirements-pages.txt
```

are committed and that the workflow is using Python 3.11.

### The page opens but says the model is missing

Confirm that the publish job produced:

```text
web/model/model.json
web/model/*.bin
```

in the temporary Action workspace and that the `gh-pages` branch contains:

```text
03-densenet-medical-image-classification/model/model.json
```

### Project 04 or Project 07 disappears

Do not manually delete the `gh-pages` branch. Confirm the workflow contains:

```yaml
destination_dir: 03-densenet-medical-image-classification
keep_files: true
```

Then rerun the affected projects’ deployment workflows if necessary.

### Browser loads slowly the first time

DenseNet121 is substantially larger than a compact web model. The workflow applies two-byte weight quantization, and the browser caches model files after the first successful load. WebGL is used when available, with CPU fallback.

### A blank page appears

Use the exact URL ending in a slash:

```text
https://unit-mole.github.io/cnn-projects/03-densenet-medical-image-classification/
```

Open the browser developer console and check for blocked CDN or model-file requests.

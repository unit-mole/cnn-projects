# Deploy Project 07 to GitHub Pages

## Live URL

```text
https://unit-mole.github.io/cnn-projects/07-image-classification-alexnet-transfer-learning/
```

## Publishing arrangement

The repository uses one `gh-pages` branch for the GitHub Pages website:

- Project 04 remains at the repository Pages root.
- Project 07 is published into its own subdirectory.
- Project 07's workflow uses `keep_files: true`, so publishing Project 07 does not remove Project 04.

No root-level `github-pages/` folder is required.

## Project 07 workflow

```text
.github/workflows/07-image-classification-alexnet-transfer-learning.yml
```

On a push to `main`, the workflow:

1. validates Python and browser files,
2. checks the TensorFlow.js manifest and binary shard,
3. runs the unit tests,
4. publishes `07-image-classification-alexnet-transfer-learning/web/` to the matching directory on `gh-pages`.

## GitHub Pages setting

Under **Settings → Pages**, use:

```text
Source: Deploy from a branch
Branch: gh-pages
Folder: /(root)
```

## Local verification

```bash
cd 07-image-classification-alexnet-transfer-learning/web
python -m http.server 8000
```

Open `http://localhost:8000`. Do not open `index.html` directly using `file://`, because browser security rules can block model and metadata requests.

## Browser InputLayer fix

The bundled demo model must use this TensorFlow.js-compatible field:

```json
"batch_input_shape": [null, 227, 227, 3]
```

Do not change it to the Keras 3 field `batch_shape`. The validation script rejects that incompatible field before deployment.

The browser app also contains a programmatic smoke-model fallback. It is used only when the bundled untrained smoke manifest cannot be loaded. A trained artifact does not use the fallback.

## Replace the smoke-test model

After training, run:

```bash
python scripts/convert_to_tfjs.py
```

Commit:

```text
web/tfjs_model/model.json
web/tfjs_model/*.bin
web/metadata.json
```

The conversion script changes `artifact_status` to `trained`, so the browser loads the trained TensorFlow.js export directly.

## Troubleshooting

- A 404 means the Project 07 publishing job has not completed or Pages is not serving `gh-pages` from `/(root)`.
- An InputLayer error means `model.json` probably contains `batch_shape` instead of `batch_input_shape` or `input_shape`.
- A shard error means a `.bin` file referenced by `model.json` is absent or has different letter casing.
- Use `Ctrl + F5` after deployment to bypass an older cached `app.js` or model manifest.
- Keep all browser paths relative: `./metadata.json`, `./tfjs_model/model.json`, and `./sample_images/...`.

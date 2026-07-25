# Deploy Project 07 to GitHub Pages

## Live URL

```text
https://unit-mole.github.io/cnn-projects/07-image-classification-alexnet-transfer-learning/
```

## Combined repository deployment

The repository has one GitHub Pages site, so one combined workflow publishes both browser demos:

- Project 04 remains at the repository Pages root.
- Project 04 is also available at `/04-image-classification-resnet/`.
- Project 07 is available at `/07-image-classification-alexnet-transfer-learning/`.

No separate root-level `github-pages/` folder is needed.

## Workflow files

```text
.github/workflows/04-image-classification-resnet.yml
.github/workflows/07-image-classification-alexnet-transfer-learning.yml
```

The first workflow deploys the combined static site. The second performs lightweight Project 07 code and asset validation without retraining the model.

## One-time GitHub setup

1. Open the repository on GitHub.
2. Select **Settings → Pages**.
3. Set the publishing source to **GitHub Actions**.
4. Push to `main` or manually run **Deploy ResNet and AlexNet Browser Demos**.

## Local verification

```bash
cd 07-image-classification-alexnet-transfer-learning/web
python -m http.server 8000
```

Open `http://localhost:8000`. Do not open `index.html` directly using `file://`, because browser security rules can block model and metadata requests.

## Replace the smoke-test model

After training, run:

```bash
python scripts/convert_to_tfjs.py
```

Commit `web/tfjs_model/model.json`, every generated `.bin` shard, and `web/metadata.json`.

## Troubleshooting

- A 404 usually means the combined deployment workflow has not completed or Pages is not set to GitHub Actions.
- A model-loading error usually means a shard referenced by `model.json` is missing or has different letter casing.
- Keep relative paths such as `./metadata.json`, `./tfjs_model/model.json`, and `./sample_images/...`.
- Only the combined Project 04 workflow should deploy to the `github-pages` environment.

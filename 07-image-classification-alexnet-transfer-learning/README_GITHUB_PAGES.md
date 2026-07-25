# GitHub Pages fallback deployment

The same static `web/` folder can be published through GitHub Pages.

## Recommended workflow approach

1. Train and convert the model.
2. Confirm all paths in `web/index.html`, `web/app.js`, and `web/metadata.json` are relative.
3. Add a GitHub Pages deployment workflow that uploads `07-image-classification-alexnet-transfer-learning/web` as the Pages artifact.
4. In repository settings, set **Pages → Source** to **GitHub Actions**.
5. Run the workflow and open the generated Pages URL.

## Manual branch approach

You may copy the contents of `web/` to a dedicated `gh-pages` branch. Publish the branch root and keep `model.json` next to its referenced `.bin` files.

## Subfolder path warning

GitHub Pages often serves a repository under `/<repository-name>/`. This app uses relative paths such as `./tfjs_model/model.json`, so it remains compatible with a subfolder deployment. Avoid root-absolute paths beginning with `/`.

## Verification checklist

- `index.html` loads without console errors,
- TensorFlow.js CDN loads,
- `metadata.json` returns HTTP 200,
- `tfjs_model/model.json` returns HTTP 200,
- every weight shard returns HTTP 200,
- image upload works,
- top-k results render,
- smoke-test or trained-artifact status is visible,
- responsible-use notice is visible.

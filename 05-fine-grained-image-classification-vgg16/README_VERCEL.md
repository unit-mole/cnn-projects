# Vercel Deployment Guide

This project is a static TensorFlow.js application. No Python backend or model training runs on Vercel.

## Included deployment files

```text
web/index.html
web/style.css
web/app.js
web/metadata.json
web/tfjs_model/model.json
web/tfjs_model/group1-shard*.bin
package.json
vercel.json
```

## Validate locally first

```bash
cd 05-fine-grained-image-classification-vgg16
python scripts/validate_project.py
node scripts/validate-web.mjs
python scripts/run_local_web_server.py --port 8000
```

Open `http://127.0.0.1:8000`, wait until the status says **Model ready**, test both sample buttons, and upload a supported image.

## Deploy from GitHub

1. Push the completed `cnn-projects` monorepo to GitHub.
2. Sign in to Vercel and choose **Add New → Project**.
3. Import the `cnn-projects` GitHub repository.
4. Set **Root Directory** to `05-fine-grained-image-classification-vgg16`.
5. Keep the framework preset as **Other** or allow Vercel to detect the static configuration.
6. The included `vercel.json` sets `outputDirectory` to `web`.
7. Deploy.
8. Open the production URL and verify that all sixteen `.bin` shards return HTTP 200.
9. Run predictions with the packaged cat and dog samples.
10. Replace `ADD_VERCEL_URL` in both README files with the production URL.

## Model loading details

`web/app.js` loads:

```javascript
await tf.loadLayersModel("./tfjs_model/model.json");
```

The manifest references sixteen relative shard files. Keep `model.json` and every `.bin` file together inside `web/tfjs_model/`.

## Large-model considerations

The VGG16 weights are roughly 60.8 MiB before transfer compression. Initial load time therefore depends on the visitor's network and browser. The UI displays download progress. Long-term caching headers are configured for model shards.

Options after measuring real deployment performance:

- use the official converter's 16-bit or 8-bit weight quantization,
- keep VGG16 as the primary model but offer a lightweight MobileNetV2 browser demo,
- use the included Gradio fallback for server-side inference,
- retain several shards rather than one very large binary file.

Always measure prediction parity and test accuracy after quantization before replacing the current bundle.

## Troubleshooting

### Model stays on “Loading model”

Open browser developer tools → **Network** and inspect `model.json` and `.bin` requests. Confirm filenames exactly match the manifest.

### `file://` or CORS error

Do not double-click `index.html`. Use the supplied local HTTP server or Vercel.

### 404 for the site root

Confirm Vercel Root Directory is the project folder and `vercel.json` remains at that folder's root.

### Prediction shape error

Confirm `web/metadata.json` contains model size `[96, 96]` and class order `["cat", "dog"]`. The model expects preprocessed `96×96×3` BGR tensors.

### Git does not include model files

Check whether Git LFS is configured. Run `git lfs install`, verify tracked files, and confirm Vercel can access the LFS objects. The packaged web shards may also be committed as standard Git files when repository limits allow.

## Post-deployment checklist

- [ ] Page loads over HTTPS
- [ ] TensorFlow.js runtime loads
- [ ] Model reaches 100% download progress
- [ ] Both sample images produce predictions
- [ ] User upload works on desktop and mobile
- [ ] No console errors
- [ ] Responsible-use notice is visible
- [ ] GitHub and notebook links are updated
- [ ] Final URL is added to the root repository README

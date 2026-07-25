# Model artifacts

- `satellite_unet_segmentation_model.keras` — supplied compact U-Net, approximately 5.5 MB.
- `model_metadata.json` — inference dimensions, threshold, classes, preprocessing, reported metrics, and limitations.
- `metrics.json` — metrics exported by the supplied notebook.

The `.keras` model is loaded with `compile=False` during inference. That avoids requiring the custom Dice and IoU training functions merely to run predictions. Training scripts still define those metrics for model fitting and evaluation.

Because the model is below GitHub's normal 100 MB per-file limit, Git LFS is not required for this artifact. A larger replacement model should be stored with Git LFS or on the Hugging Face Hub.


## TensorFlow.js browser export

The Vercel demo uses the inference-only assets in `../tfjs_model/`:

```text
model.json
weights_manifest.json
weights.bin
model_metadata.json
```

The browser file excludes optimizer state and is regenerated with:

```bash
python scripts/export_tfjs_assets.py
```

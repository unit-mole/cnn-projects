# Kaggle notebook publishing guide

## Notebook to upload

```text
notebooks/satellite_image_segmentation_unet_kaggle.ipynb
```

## Publish the supplied synthetic benchmark

1. Sign in to Kaggle and select **Create → New Notebook**.
2. Use **File → Import Notebook** and upload the Kaggle-ready `.ipynb` file.
3. Set the language to Python.
4. Internet access is not required.
5. GPU is optional; the compact 64×64 model can run on CPU, while GPU reduces training time.
6. Select **Run All**.
7. Confirm that the model and metrics are written under:

```text
/kaggle/working/artifacts_unet_satellite/models/
```

8. Open the notebook output/files panel and download:

```text
satellite_unet_segmentation_model.keras
metrics.json
```

9. Rename or copy the model to:

```text
models/satellite_unet_segmentation_model.keras
```

10. Update `models/model_metadata.json` if dimensions, classes, threshold, or dataset change.
11. Save a version, add a clear title/description, and publish the notebook.
12. Copy the public Kaggle URL into the GitHub README and Gradio app.

## Attaching a real dataset later

The supplied notebook generates synthetic data and therefore needs no attached dataset. For a real extension:

1. Select **Add Input** in Kaggle.
2. Attach a licensed public dataset or a private dataset you are authorized to use.
3. Inspect the mounted path under `/kaggle/input/`.
4. Replace the synthetic generator with deterministic image-mask discovery.
5. Split by geographic scene rather than random patches.
6. Document the license, classes, channels, spatial resolution, and pixel distribution.
7. Do not publish restricted, sensitive, or non-redistributable imagery.

## Recommended notebook introduction

State clearly that outputs are educational machine-learning predictions. Mention cloud, shadow, seasonal, sensor, resolution, and distribution-shift limitations. Do not imply that synthetic metrics measure real-world operational capability.

## Links to add

```text
Training Notebook: https://www.kaggle.com/code/YOUR_USERNAME/satellite-image-segmentation-unet
Live Demo: https://huggingface.co/spaces/YOUR_USERNAME/satellite-image-segmentation-unet
GitHub: https://github.com/YOUR_USERNAME/cnn-projects/tree/main/06-satellite-image-segmentation-unet
```

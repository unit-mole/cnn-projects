# Hosting summary

## Primary deployment

- **Platform:** Hugging Face Spaces
- **SDK:** Gradio
- **Entry point:** `app.py`
- **Model:** `models/satellite_unet_segmentation_model.keras`
- **Training proof:** Kaggle notebook

Detailed instructions:

- Hugging Face: [`README_HUGGINGFACE.md`](README_HUGGINGFACE.md)
- Kaggle: [`kaggle/README_KAGGLE.md`](kaggle/README_KAGGLE.md)
- Local: see the main [`README.md`](README.md)

The app loads a pre-trained model and never trains during startup.

# Optional Hugging Face Spaces + Gradio Fallback

Use this path when the full VGG16 browser bundle is too slow for the target audience or device. Vercel + TensorFlow.js remains the primary deployment.

## Files used

```text
app.py
gradio_app.py
app/gradio_app.py
requirements-huggingface.txt
models/vgg16_fine_grained_classification_model.keras
```

## Local test

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-huggingface.txt
python app.py
```

## Create the Space

1. Create a new Hugging Face Space.
2. Choose the **Gradio** SDK.
3. Copy the project files into the Space repository.
4. Rename or copy `requirements-huggingface.txt` to `requirements.txt` for the Space.
5. Ensure the `.keras` model is available. Use Git LFS when required.
6. Wait for the Space build to complete.
7. Test sample and uploaded images.
8. Add the Space URL to the project README as the fallback demo.

The fallback sends the uploaded image to the Space runtime. Therefore, the privacy wording must make clear that inference is no longer entirely local to the visitor's browser.

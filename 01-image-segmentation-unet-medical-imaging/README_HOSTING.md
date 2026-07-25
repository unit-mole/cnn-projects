# Hosting Guide — Hugging Face Spaces + Gradio

## Recommended Space settings

- Owner: your Hugging Face account
- Space name: `medical-image-segmentation-unet`
- License: MIT
- SDK: Gradio
- Hardware: CPU Basic
- Visibility: Public for a portfolio demo
- Python: 3.11
- App file: `app.py`

## Browser upload method

1. Sign in to Hugging Face.
2. Choose **New Space**.
3. Select **Gradio** as the SDK.
4. Create the Space.
5. Upload the contents of this project folder so that `app.py`, `requirements.txt`, `README.md`, `models/`, `src/`, and `data/` are at the Space repository root.
6. Wait for the build to finish.
7. Open the **App** tab and test one of the synthetic examples.
8. Replace every `YOUR_HF_USERNAME` placeholder in the GitHub README with the final Space URL.

## Git method

```bash
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/medical-image-segmentation-unet
cd medical-image-segmentation-unet
# Copy the contents of this project folder here.
git add .
git commit -m "Deploy U-Net segmentation demo"
git push
```

Hugging Face Spaces rebuilds the app after each pushed commit.

## Gradio CLI method

From this project directory after installing Gradio:

```bash
gradio deploy
```

Follow the prompts to create or update the Space.

## Runtime dependency strategy

`requirements.txt` contains only the packages needed by the live app. Training, plotting, and notebook dependencies are isolated in `requirements-training.txt` so the free Space build stays smaller and faster.

## Required root files for the Space

```text
app.py
requirements.txt
README.md
models/unet_medical.keras
models/model_metadata.json
src/
data/sample_images/
```

## Build troubleshooting

- Confirm that `models/unet_medical.keras` exists in the Space repository.
- Confirm that `app_file: app.py` appears in the README YAML header.
- Open the Space **Logs** tab for the Python traceback.
- Use CPU Basic; the model is small and does not require a GPU.
- Do not add training to `app.py`; the committed model should load directly.
- If a future model exceeds ordinary repository limits, store it with Git LFS or in a Hugging Face model repository and download it at startup.

## Final links to share

- Space page: `https://huggingface.co/spaces/YOUR_HF_USERNAME/medical-image-segmentation-unet`
- Embedded app: `https://YOUR_HF_USERNAME-medical-image-segmentation-unet.hf.space`

Share the Space page on GitHub, LinkedIn, your resume, and your portfolio. Use the embedded URL only when embedding the app in another webpage.

# Hugging Face Spaces deployment guide

This project folder is directly deployable as a **Gradio Space**. The project `README.md` already contains the required YAML metadata and points to `app.py`.

## 1. Create the Space

1. Sign in to Hugging Face.
2. Select **New Space**.
3. Enter a name such as `satellite-image-segmentation-unet`.
4. Choose **Gradio** as the SDK.
5. Choose a public Space for portfolio visibility.
6. Select the free CPU hardware tier.
7. Choose the MIT license.

## 2. Required Space-root files

Copy these items from this project folder into the root of the Space repository:

```text
app.py
gradio_app.py
requirements.txt
README.md
models/
src/
data/sample_images/
```

The model is only about 5.5 MB, so normal Git upload is sufficient. Git LFS becomes relevant only for a future artifact that approaches or exceeds GitHub/Hugging Face repository file-size constraints.

## 3. Replace placeholders

Search for `YOUR_USERNAME` in:

- `README.md`
- `gradio_app.py`
- root `cnn-projects/README.md`

Replace it with your GitHub, Hugging Face, and Kaggle usernames or full URLs.

## 4. Upload with Git

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/satellite-image-segmentation-unet
cd satellite-image-segmentation-unet
# copy the required project files into this folder
git add .
git commit -m "Deploy satellite U-Net Gradio demo"
git push
```

The Space builder installs `requirements.txt`, reads the YAML metadata from `README.md`, and starts `app.py`.

## 5. Test

After the build completes:

1. Open the **App** tab.
2. Select a synthetic sample.
3. Run segmentation.
4. Confirm input, mask, overlay, probability map, JSON metadata, and mask download.
5. Review **Logs** if the build or runtime fails.

## 6. Shareable URL

Your public link will have this pattern:

```text
https://huggingface.co/spaces/YOUR_USERNAME/satellite-image-segmentation-unet
```

Add it to GitHub, your resume, LinkedIn, and your portfolio website.

## 7. Common fixes

- **Model missing:** confirm `models/satellite_unet_segmentation_model.keras` exists in the Space.
- **Memory pressure:** keep CPU concurrency low and avoid loading training libraries or datasets at startup.
- **Dependency error:** rebuild after checking pinned TensorFlow/Keras/Gradio versions.
- **Custom metric error:** inference uses `compile=False`, so the app should not require custom metric deserialization.
- **Blank examples:** confirm `data/sample_images/*.png` was uploaded.

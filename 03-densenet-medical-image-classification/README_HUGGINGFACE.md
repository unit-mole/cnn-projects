# Hugging Face Spaces Deployment

## Required Space files

Copy these items to the root of a new Gradio Space:

```text
app.py
gradio_app.py
requirements.txt
README.md
models/
src/
data/sample_images/
```

The project `README.md` already contains the Space YAML metadata block. `app.py` is the entry point.
The app loads the saved `.keras` model only when inference is requested and does not train at startup.

## Web workflow

1. Create a new Hugging Face Space.
2. Select **Gradio** as the SDK.
3. Choose a public or private visibility level.
4. Upload the required files while preserving their folder structure.
5. Wait for the build logs to finish.
6. Open the App tab and test a synthetic sample image.
7. Add the final Space URL to the project and root READMEs.

## Git workflow

```bash
git clone https://huggingface.co/spaces/<username>/<space-name>
cd <space-name>
# Copy the deployment files into this folder.
git add .
git commit -m "Deploy DenseNet Gradio portfolio app"
git push
```

The bundled model is about 32 MB, below GitHub's 100 MiB single-object block, but Hugging Face recommends
large-file tooling for machine-learning artifacts. A Space repository created on the Hub includes the
appropriate large-file configuration; follow its generated `.gitattributes` guidance when needed.

## Important presentation rule

Keep the visible audit warning until the bundled synthetic proxy model is replaced with a model trained
and evaluated on a properly documented chest-X-ray dataset.

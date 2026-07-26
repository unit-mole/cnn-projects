# Hosting Project 01

The primary public deployment is **Vercel + TensorFlow.js**.

- Entry point: `web/index.html`
- Browser code: `web/app.js`
- Browser model: `web/tfjs_model/`
- Vercel configuration: `vercel.json`
- Vercel Root Directory: `01-image-segmentation-unet-medical-imaging`
- Output Directory: `web`
- Python backend: not required
- Environment variables: not required

See [`README_VERCEL.md`](README_VERCEL.md) for local validation, GitHub integration, Vercel settings, model loading, and troubleshooting.

The existing Python/Gradio files remain in the repository as an optional local reference implementation. They are not executed by Vercel.

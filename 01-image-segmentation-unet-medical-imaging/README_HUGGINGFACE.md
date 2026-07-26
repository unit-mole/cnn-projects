# Optional Hugging Face fallback

The project was originally prepared for a Python/Gradio Space. Because compute-backed Gradio Spaces may require a paid Hugging Face plan, the primary free portfolio deployment has been changed to **Vercel + TensorFlow.js**.

The retained `app.py`, `gradio_app.py`, and Python inference modules can still be used locally or deployed to a compatible Python host later. They are not required by the Vercel website.

Primary deployment instructions: [`README_VERCEL.md`](README_VERCEL.md).

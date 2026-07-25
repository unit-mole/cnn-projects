"""Hugging Face Spaces and local Gradio entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from src.config import MEDICAL_DISCLAIMER, MODEL_METADATA_PATH, PROJECT_ROOT
from src.inference_pipeline import InferenceEngine

ENGINE = InferenceEngine()
APP_THEME = gr.themes.Soft()
APP_CSS = """
.hero {text-align: center; padding: 0.4rem 0 0.8rem 0;}
.disclaimer {border: 1px solid #d97706; border-radius: 10px; padding: 12px; background: #fff7ed;}
.footer-note {font-size: 0.92rem; opacity: 0.82;}
"""
SAMPLE_IMAGES = sorted((PROJECT_ROOT / "data" / "sample_images").glob("*.png"))


def _model_summary() -> str:
    try:
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        model = metadata["model"]
        evaluation = metadata["evaluation"]
        return f"""
### Model snapshot

| Item | Value |
|---|---|
| Architecture | {model['architecture']} |
| Parameters | {model['parameter_count']:,} |
| Input | 64×64 grayscale |
| Output | Binary probability mask |
| Synthetic test Dice | {evaluation['unet']['test_soft_dice']:.4f} |
| Synthetic test IoU | {evaluation['unet']['test_soft_iou']:.4f} |

Results are from the notebook's deterministic synthetic test set and must not be interpreted as clinical performance.
"""
    except Exception:
        return "Model metadata is unavailable."


def run_segmentation(image, threshold, ground_truth=None):
    if image is None:
        raise gr.Error("Upload an image or choose one of the safe synthetic examples.")
    try:
        result = ENGINE.predict(image, threshold=float(threshold), ground_truth=ground_truth)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    return (
        result["original"],
        result["mask"],
        result["overlay"],
        result["probability"],
        result["metrics"],
        result["download_path"],
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Medical Image Segmentation using U-Net",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as demo:
        gr.Markdown(
            """
<div class="hero">
<h1>🩺 Medical Image Segmentation using U-Net</h1>
<p>Upload a grayscale-style image and inspect the U-Net probability map, binary mask, and overlay.</p>
</div>
"""
        )
        gr.Markdown(
            f"<div class='disclaimer'><strong>Medical disclaimer:</strong> {MEDICAL_DISCLAIMER}</div>"
        )
        gr.Markdown(
            """
This portfolio demo uses a compact U-Net trained on **synthetic MRI-style 64×64 images with elliptical target masks**. It demonstrates the engineering workflow for pixel-level segmentation; it does not validate segmentation of real anatomy, tumors, lesions, organs, or tissues.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Input image",
                    type="pil",
                    sources=["upload"],
                )
                ground_truth_input = gr.Image(
                    label="Optional ground-truth mask (for Dice/IoU)",
                    type="pil",
                    sources=["upload"],
                )
                threshold = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=0.50,
                    step=0.05,
                    label="Mask threshold",
                )
                predict_button = gr.Button("Generate segmentation", variant="primary")
                clear_button = gr.ClearButton(
                    [image_input, ground_truth_input, threshold], value="Clear"
                )
            with gr.Column(scale=1):
                original_output = gr.Image(label="Original image", format="png")
                mask_output = gr.Image(label="Predicted binary mask", format="png")

        with gr.Row():
            overlay_output = gr.Image(label="Mask overlay", format="png")
            probability_output = gr.Image(label="Probability map", format="png")

        with gr.Row():
            metrics_output = gr.JSON(label="Prediction summary")
            mask_download = gr.File(label="Download predicted mask")

        predict_button.click(
            fn=run_segmentation,
            inputs=[image_input, threshold, ground_truth_input],
            outputs=[
                original_output,
                mask_output,
                overlay_output,
                probability_output,
                metrics_output,
                mask_download,
            ],
            api_name="segment",
        )

        if SAMPLE_IMAGES:
            gr.Examples(
                examples=[[str(path)] for path in SAMPLE_IMAGES],
                inputs=[image_input],
                label="Safe synthetic examples",
            )

        with gr.Accordion("Model details and limitations", open=False):
            gr.Markdown(_model_summary())
            gr.Markdown(
                """
### How U-Net works

The encoder learns progressively more abstract image features. The decoder reconstructs a pixel-level output. Skip connections carry fine spatial information from encoder layers to matching decoder layers, helping the model preserve boundaries.

### Key limitations

- Synthetic training images, not clinical scans.
- Simple elliptical targets, not real anatomy.
- No DICOM support or clinical metadata handling.
- No clinical validation, regulatory review, or deployment approval.
- Uploaded public-demo images should never contain protected health information.
"""
            )

        gr.Markdown(
            """
<div class="footer-note">
Portfolio project by Anmol Tripathi · GitHub: <a href="https://github.com/unit-mole/cnn-projects" target="_blank">unit-mole/cnn-projects</a> · Replace the placeholder Space link in the README after deployment.
</div>
"""
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        max_file_size="10mb",
        theme=APP_THEME,
        css=APP_CSS,
    )

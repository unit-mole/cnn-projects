from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.config import SAMPLE_IMAGE_DIR
from src.segmentation_pipeline import SegmenterService

SERVICE = SegmenterService()
RESPONSIBLE_USE = """
### Responsible use

This public demo is an educational portfolio artifact. The model was trained on **synthetic 64×64 satellite-style tiles**, not a real operational remote-sensing dataset. Predictions may be inaccurate for clouds, shadows, seasonal changes, sensor changes, resolution changes, or unfamiliar imagery. Do not use the masks as the sole basis for emergency response, environmental enforcement, land ownership, military, legal, financial, agricultural, infrastructure, or public-policy decisions. Do not upload private, restricted, confidential, copyrighted, or sensitive geospatial imagery.
"""


def run_segmentation(image, ground_truth=None):
    if image is None:
        raise gr.Error("Upload a satellite-style image or choose a safe synthetic example.")
    try:
        result, metrics, mask_path = SERVICE.segment(image, ground_truth)
        return (
            result.original,
            result.mask_image,
            result.overlay_image,
            result.probability_image,
            metrics,
            str(mask_path),
        )
    except Exception as exc:
        raise gr.Error(f"Segmentation failed: {exc}") from exc


def build_demo() -> gr.Blocks:
    sample_paths = [[str(path)] for path in sorted(SAMPLE_IMAGE_DIR.glob("*.png"))]
    with gr.Blocks(title="Satellite Image Segmentation U-Net") as demo:
        gr.Markdown(
            """
# 🛰️ Satellite Image Segmentation using U-Net

Upload an RGB satellite-style image to generate a binary urban-structure mask, an overlay, and a probability map. The model is a compact U-Net trained on a deterministic synthetic benchmark. The Kaggle notebook documents training, evaluation, and artifact export.

**Live-demo links:** [GitHub](https://github.com/YOUR_USERNAME/cnn-projects/tree/main/06-satellite-image-segmentation-unet) · [Kaggle Notebook](https://www.kaggle.com/code/YOUR_USERNAME/satellite-image-segmentation-unet)
"""
        )
        gr.Markdown(RESPONSIBLE_USE)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Satellite image (RGB)")
                ground_truth_input = gr.Image(
                    type="pil",
                    label="Optional ground-truth mask",
                    image_mode="L",
                )
                predict_button = gr.Button("Generate segmentation", variant="primary")
                gr.Examples(
                    examples=sample_paths,
                    inputs=[image_input],
                    label="Safe synthetic examples",
                )
            with gr.Column(scale=1):
                original_output = gr.Image(type="pil", label="Input image")
                mask_output = gr.Image(type="pil", label="Predicted binary mask")

        with gr.Row():
            overlay_output = gr.Image(type="pil", label="Mask overlay")
            probability_output = gr.Image(type="pil", label="Probability map")

        with gr.Row():
            metrics_output = gr.JSON(label="Metrics and interpretation")
            model_output = gr.JSON(value=SERVICE.model_details(), label="Model details")

        mask_download = gr.DownloadButton(label="Download predicted mask")

        gr.Markdown(
            """
## How U-Net works

The encoder learns increasingly abstract visual features. The decoder restores pixel-level resolution. Skip connections pass fine spatial detail from encoder layers to matching decoder layers, which helps preserve object boundaries.

## Limitations

The procedural generator makes target pixels brighter than background pixels, so a simple threshold baseline performs extremely well and is marginally stronger than the U-Net. The reported test scores prove the end-to-end implementation, not real-world remote-sensing generalization. A production extension should use a licensed benchmark, scene-level geographic splits, stronger augmentation, and cross-sensor evaluation.
"""
        )

        predict_button.click(
            fn=run_segmentation,
            inputs=[image_input, ground_truth_input],
            outputs=[
                original_output,
                mask_output,
                overlay_output,
                probability_output,
                metrics_output,
                mask_download,
            ],
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()

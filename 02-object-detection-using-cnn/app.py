from __future__ import annotations

import json

import gradio as gr

from src.config import METRICS_PATH
from src.inference_pipeline import predict_objects

RESPONSIBLE_USE = """
> **Responsible use:** This educational portfolio model can miss a digit, predict
> the wrong class, or return an inaccurate bounding box. It is not suitable as
> the sole basis for surveillance, safety-critical monitoring, autonomous
> driving, medical, security, legal, or production inspection decisions. Do not
> upload private, confidential, copyrighted, or personally identifiable images.
"""

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def run_detection(image, confidence_threshold, auto_invert):
    if image is None:
        raise gr.Error("Upload an image before running detection.")
    try:
        return predict_objects(
            image,
            confidence_threshold=float(confidence_threshold),
            auto_invert=bool(auto_invert),
        )
    except Exception as exc:
        raise gr.Error(f"Detection failed: {exc}") from exc


with gr.Blocks(title="CNN Handwritten Digit Detector") as demo:
    gr.Markdown(
        """
# 🎯 CNN-Based Object Detection

Upload an image containing **one handwritten digit**. The model predicts the
digit class and one bounding box. It was trained on synthetic 64×64 canvases
created from MNIST, so ordinary photographs and multi-object scenes are outside
its intended scope.
"""
    )
    gr.Markdown(RESPONSIBLE_USE)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input image")
            confidence = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.50,
                step=0.05,
                label="Minimum class confidence",
            )
            auto_invert = gr.Checkbox(
                value=True,
                label="Automatically invert bright backgrounds",
            )
            predict_button = gr.Button("Detect digit", variant="primary")

        with gr.Column():
            annotated_output = gr.Image(label="Annotated prediction")
            summary_output = gr.Dataframe(
                headers=[
                    "detected_class",
                    "confidence",
                    "x1_normalized",
                    "y1_normalized",
                    "x2_normalized",
                    "y2_normalized",
                ],
                label="Detection summary",
                interactive=False,
            )
            details_output = gr.JSON(label="Prediction details")

    gr.Examples(
        examples=[
            ["data/sample_images/sample_digit_2.png", 0.50, True],
            ["data/sample_images/sample_digit_5.png", 0.50, True],
            ["data/sample_images/sample_digit_8.png", 0.50, True],
        ],
        inputs=[image_input, confidence, auto_invert],
    )

    gr.Markdown(
        f"""
## Saved evaluation metrics

| Metric | Value |
|---|---:|
| Baseline class accuracy | {metrics['baseline_test_class_acc']:.3f} |
| Baseline mean IoU | {metrics['baseline_test_iou']:.3f} |
| CNN detector class accuracy | {metrics['detector_test_class_acc']:.3f} |
| CNN detector mean IoU | {metrics['detector_test_iou']:.3f} |
| Top-3 accuracy | {metrics['top3_accuracy']:.3f} |

**Interpretation:** classification is strong on the synthetic test set, while
localization remains a limitation. The detector predicts exactly one box, so
non-maximum suppression is neither implemented nor required.
"""
    )

    predict_button.click(
        fn=run_detection,
        inputs=[image_input, confidence, auto_invert],
        outputs=[annotated_output, summary_output, details_output],
    )

if __name__ == "__main__":
    demo.launch()

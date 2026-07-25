"""Portfolio-friendly Gradio interface for Hugging Face Spaces."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

from src.class_mapping import humanize_label
from src.config import MEDICAL_DISCLAIMER, METRICS_PATH, SAMPLE_IMAGE_DIR
from src.inference_pipeline import create_prediction_summary, get_top_predictions, predict_class


def _load_metrics() -> dict:
    if METRICS_PATH.exists():
        with METRICS_PATH.open(encoding="utf-8") as file:
            return json.load(file)
    return {}


def run_prediction(image):
    if image is None:
        raise gr.Error("Upload or select an image before running the model.")
    try:
        result = predict_class(image, include_gradcam=True)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    top = get_top_predictions(result, top_k=3)
    table = pd.DataFrame(
        [
            {
                "Class": humanize_label(row["class"]),
                "Probability": round(float(row["probability"]), 6),
                "Percentage": f"{float(row['percentage']):.2f}%",
            }
            for row in top
        ]
    )
    label_scores = {humanize_label(name): probability for name, probability in result.probabilities.items()}
    summary_path = create_prediction_summary(result)
    return (
        humanize_label(result.predicted_class),
        f"{result.confidence:.2%}",
        label_scores,
        table,
        result.interpretation,
        result.gradcam_image,
        summary_path,
    )


metrics = _load_metrics()
metric_markdown = f"""
### Bundled artifact metrics — synthetic proxy task

| Metric | Value |
|---|---:|
| DenseNet validation accuracy | {metrics.get('densenet_validation_accuracy', float('nan')):.4f} |
| DenseNet test accuracy | {metrics.get('densenet_test_accuracy', float('nan')):.4f} |
| DenseNet test ROC-AUC | {metrics.get('densenet_test_auc', float('nan')):.4f} |

These values come from the attached Fashion-MNIST-derived proxy experiment, not a clinical chest-X-ray evaluation.
"""

sample_paths = [str(path) for path in sorted(SAMPLE_IMAGE_DIR.glob("*.png"))]

with gr.Blocks(title="DenseNet Medical Image Classification") as demo:
    gr.Markdown(
        """
# 🩺 DenseNet Medical Image Classification

An end-to-end CNN portfolio project demonstrating modular preprocessing, DenseNet121 transfer learning,
classification metrics, explainability hooks, Kaggle training, and Hugging Face Spaces deployment.

> **Artifact audit:** the bundled `.keras` model was trained on a Fashion-MNIST-derived synthetic proxy,
> not on chest X-rays. The live interface therefore uses **normal-like** and **pneumonia-like** labels only.
> Run the included Kaggle notebook on an appropriate public chest-X-ray dataset and replace the model
> artifacts before presenting this as a chest-X-ray classifier.
"""
    )
    gr.Markdown(f"> ⚠️ **Medical disclaimer:** {MEDICAL_DISCLAIMER}")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload image")
            predict_button = gr.Button("Run DenseNet prediction", variant="primary")
            gr.Examples(
                examples=sample_paths,
                inputs=image_input,
                label="Synthetic software-test images (not medical images)",
            )
        with gr.Column(scale=1):
            predicted_class = gr.Textbox(label="Predicted class", interactive=False)
            confidence = gr.Textbox(label="Confidence", interactive=False)
            probability_label = gr.Label(label="Class probabilities", num_top_classes=3)
            probability_table = gr.Dataframe(
                headers=["Class", "Probability", "Percentage"],
                datatype=["str", "number", "str"],
                interactive=False,
                label="Top probabilities",
            )

    interpretation = gr.Markdown(label="Prediction interpretation")
    with gr.Row():
        gradcam_output = gr.Image(label="Optional Grad-CAM overlay", type="pil")
        summary_file = gr.File(label="Download prediction summary (JSON)")

    predict_button.click(
        fn=run_prediction,
        inputs=[image_input],
        outputs=[
            predicted_class,
            confidence,
            probability_label,
            probability_table,
            interpretation,
            gradcam_output,
            summary_file,
        ],
    )

    gr.Markdown(metric_markdown)
    gr.Markdown(
        """
### Why DenseNet?
DenseNet connects each layer to later layers, encouraging feature reuse and improving gradient flow.
This makes it a strong transfer-learning backbone for image-classification experiments, including medical
imaging research when supported by appropriate data, validation, and clinical review.

### Limitations
- Current bundled artifact is a synthetic proxy rather than a clinical model.
- Output confidence is not calibrated for healthcare decisions.
- Grad-CAM is a coarse optional visualization and is not a clinical explanation.
- Performance on one dataset does not establish real-world safety or generalization.

**Project links:** [GitHub repository placeholder](https://github.com/unit-mole/cnn-projects) ·
[Kaggle notebook link placeholder](https://www.kaggle.com/)
"""
    )

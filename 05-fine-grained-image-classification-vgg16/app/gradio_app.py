"""Optional Hugging Face Spaces + Gradio fallback."""

from __future__ import annotations

import gradio as gr

from src.inference_pipeline import load_classification_model, predict_class

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_classification_model()
    return _MODEL


def classify(image):
    if image is None:
        return {}, "Upload an image to begin.", ""
    result = predict_class(get_model(), image, top_k=2)
    labels = {row["class_name"]: row["probability"] for row in result["top_predictions"]}
    warning = result["similar_class_warning"]["message"] if result["similar_class_warning"]["is_close"] else ""
    return labels, result["summary"], warning


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="VGG16 Fine-Grained Classifier") as demo:
        gr.Markdown(
            "# VGG16 Fine-Grained Cat vs Dog Classifier\n"
            "Educational portfolio demo. Do not use predictions for high-stakes decisions."
        )
        with gr.Row():
            image = gr.Image(type="pil", label="Input image")
            with gr.Column():
                labels = gr.Label(num_top_classes=2, label="Class probabilities")
                summary = gr.Textbox(label="Interpretation")
                warning = gr.Textbox(label="Similar-class warning")
        button = gr.Button("Classify")
        button.click(classify, inputs=image, outputs=[labels, summary, warning])
        gr.Markdown(
            "The model may confuse visually similar or out-of-distribution images. "
            "Do not upload private, sensitive, confidential, copyrighted, or personally identifiable content."
        )
    return demo

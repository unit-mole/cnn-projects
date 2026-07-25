# Kaggle Training and Publishing Guide

## Purpose

Use `notebooks/densenet_medical_image_classification_kaggle.ipynb` to train and evaluate a true
folder-based chest-X-ray classifier. The bundled GitHub model is only a synthetic proxy.

## Steps

1. Open Kaggle and create a new notebook.
2. Import the provided Kaggle notebook or upload the `.ipynb` file.
3. Attach a licensed chest-X-ray dataset with `train`, `val`/`validation`, and `test` folders.
4. In notebook settings, select an available GPU accelerator.
5. Update `DATASET_ROOT` if the automatic path detection does not find the dataset.
6. Review the medical disclaimer and dataset-license notes.
7. Run all cells from top to bottom.
8. Inspect class distribution, corrupt-file checks, duplicate checks, and sample images.
9. Train the frozen DenseNet121 stage, then the optional low-learning-rate fine-tuning stage.
10. Review accuracy, precision, recall, F1, macro F1, weighted F1, ROC-AUC, PR-AUC, confusion matrix,
    false positives, false negatives, low-confidence cases, and Grad-CAM examples.
11. Download the output files from `/kaggle/working/densenet_medical_outputs/`.
12. Replace the GitHub/Hugging Face `models/` artifacts only after confirming the class order and metadata.
13. Save and publish a notebook version, then copy the public notebook URL into both READMEs.

## Expected exports

```text
densenet_medical_classification_model.keras
model_metadata.json
model_metrics.json
classification_report.csv
confusion_matrix.csv
training_history.csv
training_accuracy.png
training_loss.png
roc_curve.png
precision_recall_curve.png
gradcam_examples.png
```

## Required disclosure

Do not publish patient-identifying images or metadata. Do not describe the result as a diagnostic tool.
Document the exact dataset source, license, split, class counts, and limitations.

# Reproduced Outputs

This directory contains evaluation artifacts extracted from the supplied notebook and structured CSV/JSON summaries generated during packaging.

Key files:

- `training_accuracy_curve.png` and `training_loss_curve.png`
- `confusion_matrix.png` and `confusion_matrix.csv`
- `classification_report.csv`
- `correct_predictions.png` and `misclassified_examples.png`
- `high_confidence_wrong.png` for error analysis
- `model_metrics.json`
- `tfjs_conversion_summary.json`

The supplied binary experiment has two classes. Its reported top-2 accuracy is therefore always 100% because both available classes are included; the portfolio headline uses test accuracy and macro F1 instead.

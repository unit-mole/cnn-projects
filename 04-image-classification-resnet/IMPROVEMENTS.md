# Improvement Backlog

## High priority

- Re-run evaluation to create a complete classification report and confusion matrix from the exact saved model.
- Fine-tune the final ResNet stage with a low learning rate and compare against the frozen-backbone result.
- Add confidence calibration and an explicit low-confidence warning.

## Browser optimization

- Benchmark WebGL and WebGPU backends.
- Evaluate 16-bit or 8-bit weight quantization and report any accuracy change.
- Add a smaller comparison deployment while retaining ResNet50 as the main experiment.

## Quality-analytics extension

- Replace CIFAR-100 with a permitted product or defect-image dataset.
- Add inspection-specific class definitions and false-negative review.
- Add Grad-CAM to the Python evaluation workflow for visual evidence review.

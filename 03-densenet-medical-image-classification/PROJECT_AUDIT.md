# Project Audit

## What the attached files actually do

The executable notebook loads **Fashion-MNIST**, converts the grayscale images to three channels,
and creates a synthetic binary target. Original Fashion-MNIST classes `2`, `4`, and `6` are mapped
to `pneumonia_like=1`; all other classes are mapped to `normal_like=0`.

The split recorded in the notebook is:

| Split | Images |
|---|---:|
| Train | 52,000 |
| Validation | 8,000 |
| Test | 10,000 |

Training class distribution:

| Class | Count |
|---|---:|
| normal_like | 36,419 |
| pneumonia_like | 15,581 |

The model is DenseNet121 with ImageNet weights, a frozen backbone, 96×96 internal resizing,
global-average pooling, a 256-unit dense layer, batch normalization, dropout, and a two-class
softmax output. The stored artifact accepts 28×28×3 inputs scaled to `[0, 1]`.

## Critical documentation correction

The notebook's prose describes a chest-X-ray pneumonia dataset, but the executed code does not load
chest X-rays. Therefore:

- bundled metrics are **not pneumonia-detection metrics**;
- the model must not be described as clinically validated;
- the app uses `normal_like` and `pneumonia_like`, not diagnostic labels;
- a separate Kaggle notebook is included for training on a properly sourced public chest-X-ray dataset.

## Attached model results

| Metric | Baseline | DenseNet121 |
|---|---:|---:|
| Validation accuracy | 0.9313 | 0.9599 |
| Test accuracy | 0.9282 | 0.9564 |
| Test ROC-AUC | 0.9740 | 0.9934 |

These values are retained as reproducibility evidence for the synthetic proxy experiment only.

# Data Guide

## Bundled artifact data status

The bundled model was trained on a **Fashion-MNIST-derived synthetic proxy**. No clinical chest-X-ray
dataset is included in this repository.

## Real chest-X-ray training

Use the Kaggle notebook with a legally accessible public dataset organized as:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/ or validation/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Do not commit the full dataset. Confirm the dataset license and attribution requirements before use.
Never publish private medical images, DICOM files containing identifiers, patient names, medical-record
numbers, hospital identifiers, or protected health information.

## Sample images

`sample_images/` contains procedural synthetic images created only to test the software interface.
They are not medical images and must not be used to judge model quality.

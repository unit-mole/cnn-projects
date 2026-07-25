# Dataset and Data-Safety Notes

## What the supplied notebook actually uses

The notebook does **not** load an external Kaggle, ISIC, BraTS, lung, CT, MRI, X-ray, or hospital dataset. It generates a deterministic synthetic dataset in memory:

- 2,500 grayscale images
- 64 × 64 pixels
- one channel
- Gaussian background intensity centered around 0.35
- one brighter elliptical region per image
- one aligned binary mask per image
- 70% training, 15% validation, and 15% test split
- random seed 42

The target is therefore a **synthetic high-intensity ellipse**, not an organ, lesion, tumor, cell, or tissue label.

## Public demo assets

`sample_images/` and `sample_masks/` contain eight newly generated synthetic examples using the same algorithm. They contain no patient information, DICOM headers, hospital identifiers, names, record numbers, or protected health information.

`sample_manifest.csv` maps each safe input image to its matching reference mask.

## Extending to a real dataset

Before using a real medical dataset:

1. Confirm its license and redistribution terms.
2. Remove protected health information and DICOM metadata.
3. Split by patient, not merely by image, to reduce leakage.
4. Preserve image-mask alignment.
5. Resize masks with nearest-neighbor interpolation.
6. document modality, acquisition protocol, target definition, annotation process, and clinical limitations.
7. Re-train and re-evaluate the model; the current artifact is not suitable for real scans.

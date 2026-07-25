# Project Audit

## Source files reviewed

- `Medical_Image_Segmentation_UNet_FULL_ELITE(1).ipynb`
- `unet_medical.keras`
- `metrics.json`

## What the original project actually does

The notebook generates 2,500 deterministic synthetic MRI-style grayscale images. Each image contains noise plus a brighter elliptical region, and each mask identifies that ellipse. It is a binary segmentation problem with image and mask tensors shaped `(2500, 64, 64, 1)`.

The split is:

- training: 1,750
- validation: 375
- test: 375

The model is a compact two-level U-Net with 470,977 parameters, two skip connections, a sigmoid output, binary cross-entropy loss, Adam optimizer, soft Dice, and soft IoU.

## Strong elements retained

- deterministic random seed
- explicit train/validation/test split
- intensity-threshold baseline
- Dice and IoU metrics
- U-Net skip connections
- early stopping and learning-rate reduction
- threshold sweep
- best/worst example analysis
- saved `.keras` model and metrics
- model reload check

## Problems corrected

1. **Dataset claims:** The original narrative referred broadly to clinical datasets even though the executable code used only synthetic arrays.
2. **Clinical wording:** Claims about diagnosis and hospital deployment were removed or reframed as future, separately validated applications.
3. **Missing deployment layer:** Added a Gradio application and Hugging Face Spaces configuration.
4. **Notebook-only code:** Split preprocessing, model, evaluation, inference, visualization, and data generation into reusable modules.
5. **Missing artifact metadata:** Added input shape, normalization, threshold, architecture, training settings, dataset context, checksum, limitations, and disclaimer.
6. **Missing safety controls:** Added explicit warnings against private medical uploads and public redistribution of clinical data.
7. **Missing tests and CI:** Added unit tests, import checks, model smoke inference, and a project-scoped GitHub Actions workflow.
8. **Missing reproducible outputs:** Added safe sample images, reference masks, overlays, probability maps, baseline comparison, threshold analysis, and error examples.

## Main limitation

The unusually high test scores are expected because the synthetic task is simple and generated from a consistent intensity rule. They should not be interpreted as evidence of real-world medical segmentation capability.

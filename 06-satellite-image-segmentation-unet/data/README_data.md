# Dataset and sample-data guide

## Actual dataset used by the supplied notebook

The notebook generates **2,500 procedural 64×64 RGB images** and matching **binary masks** in memory. Each mask contains two to five rectangles representing synthetic urban structures. The corresponding image pixels are brightened inside those rectangles.

| Property | Value |
|---|---|
| Dataset type | Synthetic procedural benchmark |
| Number of samples | 2,500 |
| Image size | 64×64 |
| Channels | 3 (RGB) |
| Mask size | 64×64×1 |
| Segmentation type | Binary |
| Classes | Background; synthetic urban structure |
| Split | 1,750 train / 375 validation / 375 test |
| Seed | 42 |
| Average positive-mask rate | Approximately 10.6% in the training split |

The public `sample_images/` and `sample_masks/` files are safe synthetic tiles created from the same generator. They contain no coordinates, EXIF geolocation, private-property metadata, or restricted imagery.

## Important limitation

This is not SpaceNet, DeepGlobe, Sentinel, Landsat, or another real satellite benchmark. The synthetic target is strongly correlated with brightness, which explains the near-perfect threshold baseline and U-Net scores. Use the project as a deployable workflow demonstration, not as evidence of operational remote-sensing accuracy.

## Replacing the synthetic data with a real dataset

1. Confirm the license and redistribution terms.
2. Keep full raw imagery outside GitHub when it is large or restricted.
3. Remove coordinates and sensitive metadata from public samples.
4. Pair images and masks by stable scene/tile ID.
5. Split by geographic scene—not random patches from the same scene—to reduce leakage.
6. Use bilinear interpolation for imagery and nearest-neighbor interpolation for categorical masks.
7. Document selected bands and visualization mapping for multispectral inputs.
8. Rebuild the model when channel count or class count changes.

Never commit API keys, private coordinates, military-sensitive imagery, confidential tiles, or copyrighted tiles without permission.

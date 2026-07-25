# Dataset Notes

The training notebook downloads **CIFAR-100 fine labels** through TensorFlow/Keras. The full dataset is intentionally not stored in this repository.

- 60,000 RGB images total
- 50,000 original training images
- 10,000 official test images
- 100 fine classes
- Native size: 32×32 pixels

The notebook uses the final 10,000 original training examples as validation, leaving 40,000 for training. To keep the repository safe and lightweight, `sample_images/` contains generated browser test patterns rather than redistributed user images.

Do not commit private, confidential, personally identifiable, or unlicensed images. Remove EXIF metadata from any public demonstration image.

# Dataset Notes

The original notebook downloads MNIST through TensorFlow/Keras and constructs a
synthetic object-detection dataset in memory.

For each selected MNIST image:

1. The 28×28 digit is normalized.
2. It is randomly resized between approximately 80% and 140%.
3. It is placed at a random location on a black 64×64 canvas.
4. The digit label is retained as the class target.
5. One normalized bounding box is stored as `[x1, y1, x2, y2]`.

The repository does not redistribute the full generated dataset. Only a few
locally generated sample images are included for interface testing.

# Dataset Card — CIFAR-10 Cats and Dogs

## Actual dataset used

The supplied notebook loads **CIFAR-10** and retains only source class `3` (cat) and source class `5` (dog). Images are RGB arrays with a native resolution of **32×32×3**.

| Split | Images | Cat | Dog |
|---|---:|---:|---:|
| Training | 8,000 | 4,005 | 3,995 |
| Validation | 2,000 | approximately balanced | approximately balanced |
| Test | 2,000 | 1,000 | 1,000 |

The original notebook used 10,000 filtered training-pool images and allocated the final 2,000 to validation. The modular retraining script defaults to a seeded stratified split while preserving the same split sizes.

## Public-repository policy

The full dataset is **not copied into this repository**. TensorFlow/Keras downloads it when the training or evaluation scripts run. Only two small sample crops derived from notebook visualizations are packaged for the browser demo.

Before redistributing additional images, verify the source terms and remove private or personally identifiable content. Do not commit local raw datasets under `data/raw/`.

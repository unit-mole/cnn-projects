# Dataset notes

The experiment downloads CIFAR-10 through `torchvision.datasets.CIFAR10`. The full dataset is not stored in GitHub.

The source images are 32×32 RGB and are regrouped into `living`, `nature`, `transport`, and `urban`. This grouping creates class imbalance because `nature` contains four source classes while the others contain two.

The controlled pipeline uses a fixed seed of 42, a stratified 8,000-image validation split, the untouched official 10,000-image test split, class-weighted training, and macro/per-class evaluation metrics.

# Model artifacts

## Included now

- `model_metadata.json`: honest metadata describing the target classifier and current artifact status.
- `class_indices.json`: CIFAR-10 class order.
- `tfjs_model/`: a tiny smoke-test-only TensorFlow.js model that verifies browser loading and preprocessing.

## Generated after training

- `alexnet_cifar10.keras`
- `mobilenetv2_cifar10.keras`
- trained `tfjs_model/model.json`
- trained weight shard files

The smoke-test model uses global average RGB values and handcrafted weights. Its output is not a trained CIFAR-10 prediction and must not be shown as model performance.

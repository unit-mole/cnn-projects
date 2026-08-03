# Version 3 upgrade

Version 3 replaces TensorFlow training with native-Windows PyTorch CUDA training so the project follows the same local RTX workflow as the Transformer portfolio. Browser deployment now uses ONNX Runtime Web rather than TensorFlow.js. This avoids WSL and preserves a fully static GitHub Pages deployment.

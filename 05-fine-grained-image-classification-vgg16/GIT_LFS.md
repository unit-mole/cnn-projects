# Optional Git LFS Setup

The packaged `.keras` files are each below GitHub's single-file hard limit, but the complete project is large because it contains two Keras files and two TensorFlow.js bundle copies. Git LFS is recommended for long-term model versioning.

Install Git LFS, then run:

```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add models/*.keras
git commit -m "Track Keras model artifacts with Git LFS"
```

Do this only after Git LFS is installed. The downloadable package intentionally does not include an active `.gitattributes` rule so that a normal `git add` does not fail on systems without Git LFS.

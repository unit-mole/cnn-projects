# Monorepo integration

## Required placement

```text
cnn-projects/
├── .github/workflows/
│   └── 07-image-classification-alexnet-transfer-learning.yml
└── 07-image-classification-alexnet-transfer-learning/
```

The workflow must stay at the repository root under `.github/workflows/`; do not move it into the project folder.

## Integration steps

1. Copy this project folder into the root of `cnn-projects`.
2. Copy the workflow YAML into root `.github/workflows/`.
3. Merge, rather than blindly replace, the root README and `.gitignore` if your repository already contains customized content.
4. Commit code first without large training data.
5. Train locally, Kaggle, or Colab.
6. Export the selected model to TensorFlow.js.
7. Review shard sizes before committing. Use Git LFS when repository limits require it.
8. Add final Cloudflare and GitHub Pages URLs to both root and project READMEs.

## Path-scoped CI

The workflow triggers only when this project folder or its matching workflow file changes. This prevents unrelated CNN projects from running the same checks.

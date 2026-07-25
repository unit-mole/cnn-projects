# Monorepo Integration

## Correct location

Copy this complete folder into:

```text
cnn-projects/
└── 01-image-segmentation-unet-medical-imaging/
```

The workflow file belongs at the root of the repository, not inside the project folder:

```text
cnn-projects/
└── .github/
    └── workflows/
        └── 01-image-segmentation-unet-medical-imaging.yml
```

## Recommended integration commands

From your local `cnn-projects` directory:

```bash
git status
git add .
git commit -m "Add U-Net medical image segmentation project"
git branch -M main
git remote add origin https://github.com/unit-mole/cnn-projects.git
git push -u origin main
```

If `origin` already exists, do not add it again:

```bash
git remote -v
git push -u origin main
```

## Model file

The committed model is approximately 5.5 MB, below GitHub's normal 100 MB single-file limit, so Git LFS is not required for this artifact. Use Git LFS or the Hugging Face Model Hub if a future model becomes substantially larger.

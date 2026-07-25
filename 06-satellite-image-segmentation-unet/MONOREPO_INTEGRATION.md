# Monorepo integration

Copy the complete project folder to:

```text
cnn-projects/06-satellite-image-segmentation-unet/
```

Copy the workflow separately to the repository-level path:

```text
cnn-projects/.github/workflows/06-satellite-image-segmentation-unet.yml
```

Do **not** place the YAML workflow inside the numbered project folder.

The downloadable package also contains an updated root `cnn-projects/README.md`, `.gitignore`, and `LICENSE`. Review the root README before overwriting your existing version so that links and status labels remain consistent with projects 01–05.

## Suggested Git commands

```bash
cd cnn-projects
git add 06-satellite-image-segmentation-unet .github/workflows/06-satellite-image-segmentation-unet.yml README.md .gitignore LICENSE
git commit -m "Add satellite U-Net segmentation project"
git push origin main
```

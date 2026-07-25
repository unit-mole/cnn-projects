# Monorepo Integration

Place this folder directly at:

```text
cnn-projects/05-fine-grained-image-classification-vgg16/
```

Place the workflow at:

```text
cnn-projects/.github/workflows/05-fine-grained-image-classification-vgg16.yml
```

Do not move the workflow inside the project folder. Its path filters and working directory are already configured for the monorepo.

After copying:

```bash
cd cnn-projects
git status
git add .github/workflows/05-fine-grained-image-classification-vgg16.yml
git add 05-fine-grained-image-classification-vgg16
git add README.md .gitignore LICENSE GITHUB_REPOSITORY_SETUP.md
git commit -m "Add VGG16 fine-grained browser classification project"
git push
```

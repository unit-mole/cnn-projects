# AlexNet-Style CNN vs MobileNetV2 Transfer Learning

A native-Windows, RTX-accelerated PyTorch portfolio project comparing a Simple CNN, an AlexNet-style CNN built from scratch, frozen MobileNetV2 transfer learning, and partial MobileNetV2 fine-tuning. The selected model is exported to ONNX and deployed as a static GitHub Pages application through ONNX Runtime Web.

## Honest project framing

The project uses CIFAR-10 images regrouped into four semantic categories: `living`, `nature`, `transport`, and `urban`. It is a controlled semantic-group classification benchmark, not a true high-resolution scene dataset.

## Local Windows workflow — same pattern as the Transformer projects

Open CMD inside this project folder and run:

```cmd
py --list
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-training.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python scripts\validate_project.py
python -m pytest -q
jupyter lab notebooks\07_pytorch_cnn_model_comparison.ipynb
```

Or run the full experiment directly:

```cmd
python -u scripts\run_full_experiment.py
```

Export the winning model:

```cmd
python scripts\export_to_onnx.py
python scripts\validate_project.py --require-onnx
python scripts\run_local_web_server.py --port 8000
```

Open `http://127.0.0.1:8000`.

## Generated evidence

Each model receives accuracy, balanced accuracy, macro/weighted F1, per-class metrics, ROC-AUC, top-2 accuracy, calibration metrics, confusion matrix, learning curves, prediction galleries, parameter count, checkpoint size, and latency. The selected model additionally receives Grad-CAM and robustness testing.

## Deployment

The browser application loads `web/model/model.onnx` using ONNX Runtime Web. No Python backend is required. The `web/` folder is suitable for GitHub Pages.

## Responsible use

Educational portfolio demonstration only. Predictions may be incorrect, especially for unclear or out-of-distribution images. Do not use this model as the sole basis for medical, legal, safety, security, hiring, insurance, financial, quality-release, or production decisions.

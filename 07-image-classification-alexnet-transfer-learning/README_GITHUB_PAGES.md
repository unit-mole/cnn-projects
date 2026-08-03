# GitHub Pages deployment

1. Complete training and run `python scripts\export_to_onnx.py`.
2. Confirm `web\model\model.onnx` exists.
3. Test locally with `python scripts\run_local_web_server.py --port 8000`.
4. Commit the project and workflow.
5. In GitHub, open **Settings → Pages → Source → GitHub Actions**.
6. The workflow publishes the static `web/` folder.

# Validation report

Validation completed on 2026-07-25.

## Passed checks

- Python source compilation completed successfully.
- pytest result: 10 passed, 2 skipped.
- TensorFlow-dependent architecture tests were skipped because TensorFlow was not installed in the packaging runtime.
- JavaScript syntax check passed with `node --check web/app.js`.
- `scripts/validate_project.py` passed.
- GitHub Actions YAML parsed successfully.
- Browser metadata and TensorFlow.js model JSON parsed successfully.
- Local static server returned `index.html`, `metadata.json`, `model.json`, and the binary weight shard successfully.
- The included binary shard size matches the smoke-test manifest.

## Important boundary

The bundled TensorFlow.js artifact is an explicitly labeled smoke-test model, not a trained AlexNet-style CIFAR-10 classifier. It validates the static browser wiring only. Run training, evaluation, and conversion before publishing model-performance claims.

## Commands used

```bash
python -m compileall -q src scripts app tests train_model.py
pytest -q
node --check web/app.js
python scripts/validate_project.py
python scripts/run_local_web_server.py --host 127.0.0.1 --port 8765
```

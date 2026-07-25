# Validation report

Validation completed on 2026-07-25 after the TensorFlow.js InputLayer compatibility fix.

## Passed checks

- Python source compilation completed successfully.
- pytest result: 12 passed, 2 skipped.
- TensorFlow-dependent architecture tests were skipped because TensorFlow was not installed in the packaging runtime.
- JavaScript syntax check passed with `node --check web/app.js`.
- `scripts/validate_project.py` passed.
- `scripts/validate_github_pages.py` passed.
- GitHub Actions YAML parsed successfully.
- Browser metadata and TensorFlow.js model JSON parsed successfully.
- The packaged InputLayer uses `batch_input_shape`, not the incompatible Keras 3 `batch_shape` field.
- The TensorFlow.js binary shard exists and its byte size matches the manifest weights.
- The browser app contains a smoke-model fallback used only for the explicitly untrained demo artifact.

## Important boundary

The bundled TensorFlow.js artifact is an explicitly labeled smoke-test model, not a trained AlexNet-style CIFAR-10 classifier. It validates browser loading, preprocessing, prediction, and result rendering only. Run training, evaluation, and conversion before publishing model-performance claims.

## Commands used

```bash
python -m compileall -q src scripts app tests train_model.py
pytest -q
node --check web/app.js
python scripts/validate_project.py
python scripts/validate_github_pages.py
```

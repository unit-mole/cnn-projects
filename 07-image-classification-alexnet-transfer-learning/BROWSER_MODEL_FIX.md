# TensorFlow.js InputLayer browser fix

## Original browser error

```text
An InputLayer should be passed either a `batchInputShape` or an `inputShape`.
```

## Root cause

The manually generated smoke-test `model.json` used the Keras 3 key:

```json
"batch_shape": [null, 227, 227, 3]
```

TensorFlow.js Layers expects the serialized Keras-v2-compatible key:

```json
"batch_input_shape": [null, 227, 227, 3]
```

The original manifest also contained Keras 3 `DTypePolicy` objects and model-prefixed weight names. These were simplified to TensorFlow.js-compatible values.

## Files changed

- `web/tfjs_model/model.json`
- `models/tfjs_model/model.json`
- `web/app.js`
- `web/index.html`
- `scripts/create_smoke_test_tfjs_model.py`
- `scripts/validate_github_pages.py`
- `src/model_conversion.py`
- `tests/test_model_conversion.py`
- `README.md`
- `README_GITHUB_PAGES.md`

## Defensive browser fallback

For the explicitly untrained smoke-test artifact only, `web/app.js` now builds the same tiny TensorFlow.js model programmatically if the manifest cannot be deserialized. This prevents a broken portfolio demo while retaining the honest warning that the bundled model is not trained.

When a real converted model is installed and `artifact_status` equals `trained`, a model-loading failure remains a hard error and the fallback is not used.

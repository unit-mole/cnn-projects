# TensorFlow.js Model Location

The deployable TensorFlow.js model is stored once under:

```text
web/tfjs_model/
```

This avoids duplicating roughly 95 MB of binary shards in the repository. Run `scripts/convert_to_tfjs.py --sync-models-copy` to create a second physical copy here when an external packaging requirement specifically needs both locations.

#!/usr/bin/env python
"""Validate TensorFlow.js topology, manifest paths, and weight byte counts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DTYPE_BYTES = {"float32": 4, "int32": 4, "bool": 1, "uint8": 1, "uint16": 2, "float16": 2}


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def validate(model_json: Path) -> dict:
    document = json.loads(model_json.read_text(encoding="utf-8"))
    if document.get("format") != "layers-model":
        raise AssertionError("model.json is not a layers-model.")
    manifests = document.get("weightsManifest")
    if not manifests:
        raise AssertionError("weightsManifest is empty.")

    expected = 0
    tensors = 0
    paths = []
    for group in manifests:
        paths.extend(group.get("paths", []))
        for weight in group.get("weights", []):
            dtype = weight.get("dtype")
            if dtype not in DTYPE_BYTES:
                raise AssertionError(f"Unsupported validation dtype: {dtype}")
            expected += product(weight["shape"]) * DTYPE_BYTES[dtype]
            tensors += 1

    actual = 0
    for relative in paths:
        shard = model_json.parent / relative
        if not shard.is_file():
            raise AssertionError(f"Missing shard: {shard}")
        actual += shard.stat().st_size

    if expected != actual:
        raise AssertionError(f"Manifest expects {expected:,} bytes, shards contain {actual:,} bytes.")

    return {"model": str(model_json), "shards": len(paths), "weight_tensors": tensors, "bytes": actual}


if __name__ == "__main__":
    for relative in ("web/tfjs_model/model.json", "models/tfjs_model/model.json"):
        print(validate(PROJECT_ROOT / relative))

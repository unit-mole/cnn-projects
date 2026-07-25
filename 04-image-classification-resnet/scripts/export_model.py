from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a browser-friendly HDF5 inference model.")
    parser.add_argument("--keras-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/resnet50_cifar100_browser.h5"))
    args = parser.parse_args()

    import tensorflow as tf

    source = tf.keras.models.load_model(args.keras_model, compile=False, safe_mode=False)
    backbone = source.get_layer("resnet50")
    # Head layer names in the supplied model may be generated; select by position.
    gap, dense, batch_norm, dropout, output_dense = source.layers[-5:]

    inputs = tf.keras.Input(shape=(96, 96, 3), name="browser_preprocessed_image")
    x = backbone(inputs, training=False)
    x = gap(x)
    x = dense(x)
    x = batch_norm(x, training=False)
    x = dropout(x, training=False)
    outputs = output_dense(x)
    browser_model = tf.keras.Model(inputs, outputs, name="resnet50_cifar100_browser")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    browser_model.save(args.output)
    print(f"Saved browser model to {args.output}")


if __name__ == "__main__":
    main()

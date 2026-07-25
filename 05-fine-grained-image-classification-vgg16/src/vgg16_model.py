"""VGG16 transfer-learning architecture."""

from __future__ import annotations

from .config import CHANNELS, MODEL_IMAGE_SIZE, SOURCE_IMAGE_SIZE
from .data_preprocessing import build_safe_augmentation


def build_vgg16_classifier(
    number_of_classes: int = 2,
    *,
    backbone_trainable: bool = False,
    learning_rate: float = 1e-3,
):
    """Build the model architecture used in the supplied notebook."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required to build the VGG16 model.") from exc

    inputs = tf.keras.Input(shape=(*SOURCE_IMAGE_SIZE, CHANNELS), name="image")
    x = build_safe_augmentation()(inputs)
    x = tf.keras.layers.Resizing(*MODEL_IMAGE_SIZE, interpolation="bilinear", name="resize_96")(x)
    x = tf.keras.applications.vgg16.preprocess_input(x * 255.0)

    backbone = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(*MODEL_IMAGE_SIZE, CHANNELS),
    )
    backbone.trainable = backbone_trainable
    x = backbone(x, training=False)
    x = tf.keras.layers.Flatten(name="flatten_features")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization")(x)
    x = tf.keras.layers.Dropout(0.50, name="dropout_50")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.40, name="dropout_40")(x)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="vgg16_fine_grained_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def enable_block5_fine_tuning(model, learning_rate: float = 1e-5):
    """Optionally unfreeze VGG16 block 5 for a conservative second training stage."""
    import tensorflow as tf

    backbone = next(
        (layer for layer in model.layers if isinstance(layer, tf.keras.Model) and "vgg16" in layer.name.lower()),
        None,
    )
    if backbone is None:
        raise ValueError("VGG16 backbone was not found in the model.")

    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = layer.name.startswith("block5_")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

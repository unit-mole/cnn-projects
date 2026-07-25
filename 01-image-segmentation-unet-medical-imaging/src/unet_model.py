"""Exact compact U-Net architecture used by the supplied trained artifact."""

from __future__ import annotations

from typing import Any

from .metrics import dice_coef_tf, iou_tf


def conv_block(x: Any, filters: int) -> Any:
    import tensorflow as tf

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_unet(
    input_shape: tuple[int, int, int] = (64, 64, 1),
    learning_rate: float = 0.001,
    compile_model: bool = True,
) -> Any:
    """Build the 470,977-parameter two-level U-Net from the original notebook."""
    import tensorflow as tf

    inputs = tf.keras.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    bottleneck = conv_block(p2, 128)

    u4 = tf.keras.layers.UpSampling2D(interpolation="nearest")(bottleneck)
    u4 = tf.keras.layers.Concatenate()([u4, c2])
    c4 = conv_block(u4, 64)

    u5 = tf.keras.layers.UpSampling2D(interpolation="nearest")(c4)
    u5 = tf.keras.layers.Concatenate()([u5, c1])
    c5 = conv_block(u5, 32)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c5)
    model = tf.keras.Model(inputs, outputs, name="compact_medical_unet")

    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=[dice_coef_tf, iou_tf],
        )
    return model

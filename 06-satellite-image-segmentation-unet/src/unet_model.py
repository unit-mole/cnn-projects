from __future__ import annotations

from .metrics import dice_coef_tf, iou_tf


def _conv_block(x, filters: int):
    import tensorflow as tf
    x = tf.keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    return x


def build_unet(input_shape: tuple[int, int, int] = (64, 64, 3)):
    import tensorflow as tf
    inputs = tf.keras.Input(shape=input_shape)
    encoder_1 = _conv_block(inputs, 32)
    pooled_1 = tf.keras.layers.MaxPooling2D()(encoder_1)
    encoder_2 = _conv_block(pooled_1, 64)
    pooled_2 = tf.keras.layers.MaxPooling2D()(encoder_2)
    bottleneck = _conv_block(pooled_2, 128)

    up_1 = tf.keras.layers.UpSampling2D(interpolation="nearest")(bottleneck)
    up_1 = tf.keras.layers.Concatenate()([up_1, encoder_2])
    decoder_1 = _conv_block(up_1, 64)

    up_2 = tf.keras.layers.UpSampling2D(interpolation="nearest")(decoder_1)
    up_2 = tf.keras.layers.Concatenate()([up_2, encoder_1])
    decoder_2 = _conv_block(up_2, 32)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(decoder_2)
    return tf.keras.Model(inputs, outputs, name="satellite_unet")


def compile_unet(model, learning_rate: float = 1e-3):
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[dice_coef_tf, iou_tf],
    )
    return model

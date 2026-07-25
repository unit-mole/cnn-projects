from __future__ import annotations

import tensorflow as tf


def build_detector(input_shape: tuple[int, int, int] = (64, 64, 1), num_classes: int = 10) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    class_output = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(x)
    box_output = tf.keras.layers.Dense(4, activation="sigmoid", name="box_output")(x)
    return tf.keras.Model(inputs, [class_output, box_output], name="cnn_digit_detector")


def compile_detector(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={"class_output": "categorical_crossentropy", "box_output": "mse"},
        metrics={"class_output": ["accuracy"]},
    )
    return model

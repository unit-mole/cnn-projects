from __future__ import annotations


def build_alexnet_style(
    input_shape: tuple[int, int, int],
    num_classes: int,
    learning_rate: float = 1e-3,
    dense_units: tuple[int, int] = (512, 256),
    dropout_rate: float = 0.5,
):
    """Build a compact AlexNet-style network for browser-friendly export.

    This is intentionally described as AlexNet-style. It preserves the five
    convolutional stages and large early receptive fields while replacing the
    original 4096-unit fully connected head with global average pooling and a
    smaller classification head.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    import tensorflow as tf

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Conv2D(64, 11, strides=4, padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name="pool1")(x)

    x = tf.keras.layers.Conv2D(192, 5, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name="pool2")(x)

    x = tf.keras.layers.Conv2D(384, 3, padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="conv4")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="conv5")(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name="pool5")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(dense_units[0], activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout1")(x)
    x = tf.keras.layers.Dense(dense_units[1], activation="relu", name="fc2")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = tf.keras.Model(inputs, outputs, name="alexnet_style_browser_aware")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(5, num_classes), name="top_k_accuracy"),
        ],
    )
    return model

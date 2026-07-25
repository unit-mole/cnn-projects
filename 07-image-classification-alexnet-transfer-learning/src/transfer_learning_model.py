from __future__ import annotations


def build_mobilenetv2_transfer_model(
    input_shape: tuple[int, int, int],
    num_classes: int,
    learning_rate: float = 3e-4,
    pretrained: bool = True,
    fine_tune_layers: int = 0,
):
    """Build an honest transfer-learning baseline using MobileNetV2.

    Passing pretrained=False initializes the base randomly and therefore should
    not be described as transfer learning in reports.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    import tensorflow as tf

    weights = "imagenet" if pretrained else None
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling=None,
    )
    base.trainable = False

    if fine_tune_layers > 0:
        base.trainable = True
        for layer in base.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers[-fine_tune_layers:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(5, num_classes), name="top_k_accuracy"),
        ],
    )
    return model

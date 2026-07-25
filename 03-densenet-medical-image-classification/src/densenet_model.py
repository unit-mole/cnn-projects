"""DenseNet121 transfer-learning model builder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseNetConfig:
    image_size: tuple[int, int] = (224, 224)
    channels: int = 3
    num_classes: int = 2
    dropout_rate: float = 0.35
    learning_rate: float = 1e-3
    imagenet_weights: bool = True


def build_densenet121(config: DenseNetConfig = DenseNetConfig()):
    import tensorflow as tf

    from .data_preprocessing import build_medically_conservative_augmentation

    inputs = tf.keras.Input((*config.image_size, config.channels), name="image")
    x = build_medically_conservative_augmentation()(inputs)
    x = tf.keras.applications.densenet.preprocess_input(x)
    backbone = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet" if config.imagenet_weights else None,
        input_shape=(*config.image_size, config.channels),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="dropout")(x)

    if config.num_classes == 2:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="probability")(x)
        loss = "binary_crossentropy"
        metrics = [
            "accuracy",
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    else:
        outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax", name="probabilities")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, config.num_classes))]

    model = tf.keras.Model(inputs, outputs, name="densenet121_medical_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model, backbone


def unfreeze_for_fine_tuning(model, backbone, trainable_tail_layers: int = 40, learning_rate: float = 1e-5):
    import tensorflow as tf

    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - int(trainable_tail_layers))
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=model.loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

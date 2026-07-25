from __future__ import annotations

from typing import Any

from .config import ProjectConfig


def build_augmentation() -> Any:
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="safe_augmentation",
    )


def build_resnet50_classifier(config: ProjectConfig | None = None, weights: str | None = "imagenet") -> Any:
    """Build the architecture used by the attached notebook."""
    import tensorflow as tf

    cfg = config or ProjectConfig()
    tf.keras.utils.set_random_seed(cfg.seed)

    inputs = tf.keras.Input(shape=(*cfg.original_image_size, cfg.channels), name="image")
    x = tf.keras.layers.Resizing(*cfg.model_image_size, name="resize_for_resnet")(inputs)
    x = build_augmentation()(x)
    x = tf.keras.layers.Lambda(
        lambda values: tf.keras.applications.resnet.preprocess_input(values * 255.0),
        name="resnet_preprocess",
    )(x)

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=(*cfg.model_image_size, cfg.channels),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(cfg.dense_units, activation="relu", name="classification_dense")(x)
    x = tf.keras.layers.BatchNormalization(name="classification_batch_norm")(x)
    x = tf.keras.layers.Dropout(cfg.dropout_rate, name="classification_dropout")(x)
    outputs = tf.keras.layers.Dense(cfg.num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="resnet50_cifar100")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    return model


def unfreeze_final_stage(model: Any, trainable_from: str = "conv5_block1_1_conv") -> None:
    """Optional fine-tuning helper; call after frozen-head training."""
    backbone = model.get_layer("resnet50")
    backbone.trainable = True
    make_trainable = False
    for layer in backbone.layers:
        if layer.name == trainable_from:
            make_trainable = True
        layer.trainable = make_trainable and not layer.name.endswith("bn")

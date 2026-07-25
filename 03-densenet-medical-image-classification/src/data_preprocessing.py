"""Training-dataset preprocessing utilities."""

from __future__ import annotations


def configure_for_performance(dataset, cache: bool = False):
    """Apply optional caching and automatic prefetching to a tf.data dataset."""
    import tensorflow as tf

    if cache:
        dataset = dataset.cache()
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_medically_conservative_augmentation():
    """Small transformations intended for chest X-ray experimentation.

    Horizontal flipping is intentionally excluded by default because laterality can
    matter. Users should enable it only after confirming it is appropriate for the
    selected dataset and objective.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.025),
            tf.keras.layers.RandomZoom(0.05),
            tf.keras.layers.RandomTranslation(0.02, 0.02),
            tf.keras.layers.RandomContrast(0.08),
        ],
        name="conservative_augmentation",
    )

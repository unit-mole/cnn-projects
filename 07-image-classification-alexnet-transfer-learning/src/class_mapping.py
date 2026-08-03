from __future__ import annotations

import numpy as np

ORIGINAL_CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
GROUP_CLASS_NAMES = ("living", "nature", "transport", "urban")
ORIGINAL_INDEX_TO_GROUP = {
    0: "transport", 1: "urban", 2: "nature", 3: "living", 4: "nature",
    5: "living", 6: "nature", 7: "nature", 8: "transport", 9: "urban",
}
GROUP_TO_INDEX = {name: index for index, name in enumerate(GROUP_CLASS_NAMES)}


def map_original_labels(labels) -> np.ndarray:
    flat = np.asarray(labels).reshape(-1)
    unknown = sorted(set(int(value) for value in flat) - set(ORIGINAL_INDEX_TO_GROUP))
    if unknown:
        raise ValueError(f"Unexpected CIFAR-10 labels: {unknown}")
    return np.asarray(
        [GROUP_TO_INDEX[ORIGINAL_INDEX_TO_GROUP[int(value)]] for value in flat],
        dtype=np.int64,
    )

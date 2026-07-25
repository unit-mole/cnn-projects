from __future__ import annotations

CLASS_MAPPING = {
    0: "Background",
    1: "Synthetic urban structure",
}


def class_name(class_id: int) -> str:
    try:
        return CLASS_MAPPING[int(class_id)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported class id: {class_id}") from exc

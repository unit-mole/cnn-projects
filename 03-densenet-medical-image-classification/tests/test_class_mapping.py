import json

from src.class_mapping import class_names, humanize_label, load_metadata


def test_metadata_class_order(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps({
            "classes": ["normal_like", "pneumonia_like"],
            "class_to_index": {"normal_like": 0, "pneumonia_like": 1},
            "input_shape": [28, 28, 3],
            "dataset_status": "synthetic_proxy_not_clinical",
        }),
        encoding="utf-8",
    )
    metadata = load_metadata(path)
    assert class_names(metadata) == ["normal_like", "pneumonia_like"]
    assert humanize_label("pneumonia_like") == "Pneumonia Like"

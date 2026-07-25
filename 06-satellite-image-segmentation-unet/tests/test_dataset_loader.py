from pathlib import Path

from src.dataset_loader import pair_image_and_mask_files


def test_sample_images_have_matching_masks():
    root = Path(__file__).resolve().parents[1]
    pairs = pair_image_and_mask_files(root / "data" / "sample_images", root / "data" / "sample_masks")
    assert len(pairs) == 6

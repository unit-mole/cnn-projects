from src.class_mapping import CIFAR100_FINE_LABELS, label_for_index, validate_class_names


def test_cifar100_mapping_has_100_unique_labels():
    assert len(CIFAR100_FINE_LABELS) == 100
    assert len(set(CIFAR100_FINE_LABELS)) == 100
    assert label_for_index(0) == "apple"
    assert label_for_index(99) == "worm"


def test_validate_class_names_rejects_duplicates():
    try:
        validate_class_names(["cat", "cat"])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate labels should be rejected")

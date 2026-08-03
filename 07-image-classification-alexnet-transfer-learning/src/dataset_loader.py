from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms

from .class_mapping import GROUP_CLASS_NAMES, map_original_labels

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SemanticCIFAR10(Dataset):
    def __init__(self, base: CIFAR10, indices: np.ndarray, transform):
        self.base = base
        self.indices = np.asarray(indices)
        self.transform = transform
        self.group_labels = map_original_labels(np.asarray(base.targets)[self.indices])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.base[int(self.indices[index])]
        label = int(self.group_labels[index])
        return self.transform(image), label


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_labels: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray
    class_names: tuple[str, ...] = GROUP_CLASS_NAMES

    def summary(self) -> dict[str, object]:
        def counts(labels):
            values = np.bincount(labels, minlength=len(self.class_names))
            return {name: int(values[i]) for i, name in enumerate(self.class_names)}
        return {
            "source": "CIFAR-10 regrouped into four semantic categories",
            "image_size": [96, 96],
            "train_images": int(len(self.train_labels)),
            "validation_images": int(len(self.val_labels)),
            "test_images": int(len(self.test_labels)),
            "class_names": list(self.class_names),
            "train_distribution": counts(self.train_labels),
            "validation_distribution": counts(self.val_labels),
            "test_distribution": counts(self.test_labels),
            "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        }


def build_transforms(image_size: int = 96):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def load_data(data_dir: Path, validation_size: int, batch_size: int, num_workers: int, image_size: int, seed: int) -> DataBundle:
    train_base = CIFAR10(root=data_dir, train=True, download=True)
    test_base = CIFAR10(root=data_dir, train=False, download=True)
    all_indices = np.arange(len(train_base))
    all_group_labels = map_original_labels(train_base.targets)
    train_idx, val_idx = train_test_split(
        all_indices, test_size=validation_size, random_state=seed,
        stratify=all_group_labels, shuffle=True,
    )
    test_idx = np.arange(len(test_base))
    train_transform, eval_transform = build_transforms(image_size)
    train_ds = SemanticCIFAR10(train_base, train_idx, train_transform)
    val_ds = SemanticCIFAR10(train_base, val_idx, eval_transform)
    test_ds = SemanticCIFAR10(test_base, test_idx, eval_transform)
    generator = torch.Generator().manual_seed(seed)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, persistent_workers=num_workers > 0, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, persistent_workers=num_workers > 0, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, persistent_workers=num_workers > 0, **kwargs)
    return DataBundle(train_loader, val_loader, test_loader, train_ds.group_labels, val_ds.group_labels, test_ds.group_labels)


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32, device=device)

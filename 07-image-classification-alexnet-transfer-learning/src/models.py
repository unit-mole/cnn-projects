from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(192, num_classes))

    def forward(self, x): return self.classifier(self.features(x))
    def gradcam_layer(self): return self.features[-3]


class AlexNetStyle(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), nn.BatchNorm2d(96), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.BatchNorm2d(256), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, num_classes),
        )

    def forward(self, x): return self.classifier(self.features(x))
    def gradcam_layer(self): return self.features[-3]


class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.30), nn.Linear(in_features, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(256, num_classes),
        )

    def forward(self, x): return self.backbone(x)
    def gradcam_layer(self): return self.backbone.features[-1]

    def freeze_backbone(self):
        for p in self.backbone.features.parameters(): p.requires_grad = False

    def unfreeze_last_blocks(self, blocks: int = 4):
        for p in self.backbone.features.parameters(): p.requires_grad = False
        for module in list(self.backbone.features.children())[-blocks:]:
            for p in module.parameters(): p.requires_grad = True


def build_model(name: str, num_classes: int = 4, pretrained: bool = True):
    if name == "simple_cnn": return SimpleCNN(num_classes)
    if name == "alexnet_style": return AlexNetStyle(num_classes)
    if name in {"mobilenetv2_frozen", "mobilenetv2_finetuned"}: return MobileNetClassifier(num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")

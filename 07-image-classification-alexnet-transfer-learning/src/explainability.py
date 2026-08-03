from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .visualization import denormalize


def gradcam(
    model,
    image: torch.Tensor,
    device: torch.device,
    target_class: int | None = None,
):
    activations = []
    gradients = []
    layer = model.gradcam_layer()

    def capture(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = layer.register_forward_hook(capture)
    model.eval()
    model.zero_grad(set_to_none=True)

    x = image.unsqueeze(0).to(device)
    logits = model(x)
    target = int(logits.argmax(1).item()) if target_class is None else int(target_class)
    logits[0, target].backward()

    activation = activations[0]
    gradient = gradients[0]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(1, keepdim=True).relu()
    cam = F.interpolate(
        cam,
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    handle.remove()

    confidence = torch.softmax(logits, 1)[0, target].item()
    return cam.detach().cpu().numpy(), target, confidence


def save_gradcam(
    model,
    loader,
    device,
    class_names,
    path: Path,
    samples: int = 8,
) -> None:
    images, labels = next(iter(loader))
    samples = min(samples, len(images))
    fig = plt.figure(figsize=(12, 6))

    for index in range(samples):
        cam, prediction, confidence = gradcam(model, images[index], device)
        image = denormalize(images[index])
        ax = fig.add_subplot(2, 4, index + 1)
        ax.imshow(image)
        ax.imshow(cam, cmap="jet", alpha=0.42)
        ax.axis("off")
        ax.set_title(
            f"T:{class_names[int(labels[index])]}\n"
            f"P:{class_names[prediction]} {confidence:.1%}",
            fontsize=9,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)

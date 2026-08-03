from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def _corrupt(x, name):
    if name=="gaussian_noise": return torch.clamp(x+torch.randn_like(x)*0.12,-3,3)
    if name=="darkness": return x*0.55
    if name=="brightness": return x*1.25
    if name=="blur": return F.avg_pool2d(x,3,stride=1,padding=1)
    if name=="rotation_90": return torch.rot90(x,1,[2,3])
    return x


def evaluate_robustness(model, loader, device, sample_size=1500):
    names=["clean","gaussian_noise","darkness","brightness","blur","rotation_90"]; results={}
    model.eval()
    for name in names:
        ys=[]; ps=[]; seen=0
        with torch.inference_mode():
            for images,labels in loader:
                images=_corrupt(images.to(device),name); logits=model(images); ys.extend(labels.numpy()); ps.extend(logits.argmax(1).cpu().numpy()); seen+=len(labels)
                if seen>=sample_size: break
        score=f1_score(np.asarray(ys)[:sample_size],np.asarray(ps)[:sample_size],average="macro",zero_division=0); results[name]={"macro_f1":float(score)}
    clean=results["clean"]["macro_f1"]
    for value in results.values(): value["drop_from_clean"]=float(clean-value["macro_f1"])
    return results

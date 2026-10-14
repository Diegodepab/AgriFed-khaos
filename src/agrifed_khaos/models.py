"""Unified Vision Model Factory for AgriFed-KHAOS.

Provides dynamic, domain-agnostic instantiation of deep learning vision architectures
(PyTorch native backbones, Timm Vision Transformers and CNNs, and Ultralytics YOLO).
Decouples client training, model serving, and evaluation from any hardcoded architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    pass


def build_model(
    architecture: str = "resnet18",
    num_classes: int = 9,
    pretrained: bool = False,
) -> Any:
    """Instantiate a computer vision classification backbone with customized output head.

    Supports:
    - Torchvision: resnet18, resnet34, resnet50, mobilenet_v3_small,
      mobilenet_v3_large, efficientnet_b0
    - Timm: convnext_tiny, vit_tiny_patch16_224, mobilenetv4_conv_small,
      and any identifier available in timm.
    """
    try:
        import torch.nn as nn
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and torchvision are required to build vision models. "
            "Install with: pip install 'agrifed-khaos[vision]'"
        ) from exc

    arch = architecture.lower().strip()

    # 1. Native Torchvision ResNets
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return cast(nn.Module, m)
    elif arch == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return cast(nn.Module, m)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return cast(nn.Module, m)

    # 2. Native MobileNetV3 (Edge & Low Bandwidth FL)
    elif arch in {"mobilenet_v3_small", "mobilenetv3_small"}:
        m_mob = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        in_feat = m_mob.classifier[3].in_features
        m_mob.classifier[3] = nn.Linear(in_feat, num_classes)
        return cast(nn.Module, m_mob)
    elif arch in {"mobilenet_v3_large", "mobilenetv3_large"}:
        m_mob = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        )
        in_feat = m_mob.classifier[3].in_features
        m_mob.classifier[3] = nn.Linear(in_feat, num_classes)
        return cast(nn.Module, m_mob)

    # 3. Native EfficientNet-B0
    elif arch in {"efficientnet_b0", "efficientnet"}:
        m_eff = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        in_feat = m_eff.classifier[1].in_features
        m_eff.classifier[1] = nn.Linear(in_feat, num_classes)
        return cast(nn.Module, m_eff)

    # 4. Timm Library Models (ConvNeXt, ViT, Swin, MobileNetV4, etc.)
    try:
        import timm

        m_timm = timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)
        return cast(nn.Module, m_timm)
    except Exception as exc:
        raise ValueError(
            f"Unsupported vision architecture '{architecture}'. "
            f"Supported native: resnet18, resnet34, resnet50, mobilenet_v3_small, "
            f"mobilenet_v3_large, efficientnet_b0, or any valid timm model name "
            f"(e.g. convnext_tiny, vit_tiny_patch16_224)."
        ) from exc


def get_cam_target_layer(model: Any, architecture: str = "resnet18") -> Any:
    """Resolve the deepest convolutional feature layer for Grad-CAM visual explanations."""
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Grad-CAM layer extraction.") from exc

    if hasattr(model, "layer4"):
        return cast(Any, model.layer4)[-1]
    elif hasattr(model, "features"):
        return cast(Any, model.features)[-1]
    elif hasattr(model, "stages"):
        return cast(Any, model.stages)[-1]

    # Dynamic fallback to last 2D convolution or batchnorm in module hierarchy
    for module in reversed(list(model.modules())):
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            return module

    children = list(model.children())
    if children:
        return children[-1]
    return model

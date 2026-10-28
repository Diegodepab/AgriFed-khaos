"""Unified Vision Model Factory and Checkpoint Serialization for AgriFed-KHAOS.

Provides dynamic, domain-agnostic instantiation of deep learning vision architectures
(PyTorch native backbones and Timm Vision Transformers and CNNs).
Manages self-contained checkpoint bundles packaging state_dict and architecture metadata.
"""

from __future__ import annotations

from pathlib import Path
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


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explanations."""

    def __init__(
        self, model: Any, target_layer: Any, temperature: float = 1.0
    ) -> None:
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for GradCAM.") from exc

        self.model = model
        self.target_layer = target_layer
        self.temperature = max(float(temperature), 1e-3)
        self.activations: Any | None = None
        self.gradients: Any | None = None
        self.fwd_handle = self.target_layer.register_forward_hook(self._fwd)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module: Any, inp: Any, out: Any) -> None:
        self.activations = out

    def _bwd(self, module: Any, gin: Any, gout: tuple[Any, ...]) -> None:
        self.gradients = gout[0]

    def remove(self) -> None:
        """Remove forward and backward hooks to prevent memory leaks."""
        if hasattr(self, "fwd_handle") and self.fwd_handle is not None:
            self.fwd_handle.remove()
        if hasattr(self, "bwd_handle") and self.bwd_handle is not None:
            self.bwd_handle.remove()

    def generate(
        self, x: Any, class_idx: int | None = None
    ) -> tuple[Any, int, float]:
        import torch
        import torch.nn.functional as F

        self.model.eval()
        logits = self.model(x)
        probs = F.softmax(logits / self.temperature, dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)
        assert self.gradients is not None, "GradCAM failed: gradients were not captured."
        assert self.activations is not None, "GradCAM failed: activations were not captured."

        act = self.activations
        grad = self.gradients

        if act.dim() == 4:
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            cam = torch.relu(torch.sum(weights * act, dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        elif act.dim() == 3:
            tokens = act[:, 1:, :] if act.shape[1] > 1 else act
            t_grad = grad[:, 1:, :] if grad.shape[1] > 1 else grad
            weights = torch.mean(t_grad, dim=1, keepdim=True)
            cam_tokens = torch.relu(torch.sum(weights * tokens, dim=-1))
            grid_size = int(round(cam_tokens.shape[1] ** 0.5))
            if grid_size * grid_size == cam_tokens.shape[1]:
                cam_2d = cam_tokens.reshape(cam_tokens.shape[0], 1, grid_size, grid_size)
                cam = F.interpolate(cam_2d, size=x.shape[2:], mode="bilinear", align_corners=False)
            else:
                cam = torch.ones((1, 1, x.shape[2], x.shape[3]), device=x.device)
        else:
            cam = torch.ones((1, 1, x.shape[2], x.shape[3]), device=x.device)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam[0, 0].detach().cpu().numpy(), class_idx, float(probs[0, class_idx].item())


def save_checkpoint_bundle(
    path: Path | str,
    state_dict: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model weights and metadata atomically into a self-contained bundle.

    Writes to a temporary file in the same directory and commits via POSIX os.replace
    so concurrent readers (e.g. serve.py hot-reload watcher) never observe a partial write.
    """
    import os
    import uuid

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to save checkpoint bundles.") from exc

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": state_dict,
        "metadata": metadata or {},
    }
    tmp_path = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(bundle, str(tmp_path))
        os.replace(tmp_path, p)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def load_checkpoint_bundle(
    path: Path | str,
    device: Any = "cpu",
    allow_unsafe_legacy: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a checkpoint supporting both bundled dictionaries and legacy raw state_dicts.

    Args:
        path: Path to the checkpoint file (.pt)
        device: Device to map tensors onto
        allow_unsafe_legacy: If True, falls back to weights_only=False if weights_only=True fails.
            Default is False to strictly prevent arbitrary code execution (CWE-502)
            from untrusted client-supplied checkpoints.

    Returns:
        tuple (state_dict, metadata)
    """
    import json

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load checkpoint bundles.") from exc

    try:
        checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    except Exception as exc:
        if allow_unsafe_legacy:
            checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        else:
            raise ValueError(
                f"Failed to safely load checkpoint '{path}' with weights_only=True. "
                "Arbitrary code execution protection is active. If this is a trusted local legacy "
                "artifact, set allow_unsafe_legacy=True explicitly."
            ) from exc

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
        and isinstance(checkpoint["state_dict"], dict)
    ):
        state_dict: dict[str, Any] = checkpoint["state_dict"]
        metadata: dict[str, Any] = checkpoint.get("metadata", {})
        return state_dict, metadata

    # Legacy fallback: checkpoint is a raw state_dict
    state_dict = checkpoint
    metadata = {}
    p = Path(path)
    meta_candidates = [
        p.with_suffix(".meta.json"),
        p.parent / f"{p.name}.meta.json",
        p.parent / "metadata.json",
    ]
    for meta_cand in meta_candidates:
        if meta_cand.is_file():
            try:
                metadata = json.loads(meta_cand.read_text(encoding="utf-8"))
                break
            except Exception:
                pass
    return state_dict, metadata


def init_deterministic_classifier_row(
    in_features: int,
    class_name: str | None = None,
    row_idx: int = 0,
    dtype: Any = None,
    device: Any = "cpu",
    has_bias: bool = True,
) -> tuple[Any, Any | None]:
    """Deterministically initialize a single row of classifier weight and bias.

    Uses a dedicated torch.Generator seeded with zlib.crc32(class_name.encode('utf-8'))
    (or deterministic index fallback) to guarantee bit-for-bit identical initialization
    between client local models and server aggregated states.
    """
    import math
    import zlib

    import torch

    if class_name is not None and str(class_name) != "":
        seed = zlib.crc32(str(class_name).encode("utf-8")) & 0xFFFFFFFF
    else:
        seed = (133742 + row_idx * 65537) & 0xFFFFFFFF

    gen = torch.Generator(device="cpu").manual_seed(seed)
    bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0

    w_row = torch.empty((in_features,), dtype=torch.float32, device="cpu")
    w_row.uniform_(-bound, bound, generator=gen)

    b_val: Any | None = None
    if has_bias:
        b_val = torch.empty((1,), dtype=torch.float32, device="cpu")
        b_val.uniform_(-bound, bound, generator=gen)

    target_dtype = dtype if dtype is not None else torch.float32
    return w_row.to(dtype=target_dtype, device=device), (
        b_val[0].to(dtype=target_dtype, device=device) if b_val is not None else None
    )


def expand_model_classifier(
    model: Any,
    new_num_classes: int,
    class_names: list[str] | None = None,
) -> tuple[Any, str, int]:
    """Dynamically expand the final classification layer to accommodate new classes.

    Preserves weights and biases for existing classes (top C_old rows) and
    deterministically initializes newly added class rows using seeds derived
    from their canonical class names.

    Returns:
        (model, classifier_param_name, old_num_classes)
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to expand model classifier.") from exc

    last_linear_name: str | None = None
    old_linear: nn.Linear | None = None

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_linear_name = name
            old_linear = mod

    if last_linear_name is None or old_linear is None:
        raise ValueError("Could not locate a final nn.Linear classifier layer to expand.")

    old_num_classes = int(old_linear.out_features)
    if new_num_classes <= old_num_classes:
        return model, last_linear_name, old_num_classes

    in_features = int(old_linear.in_features)
    has_bias = old_linear.bias is not None
    new_linear = nn.Linear(in_features, new_num_classes, bias=has_bias)

    with torch.no_grad():
        new_linear.weight[:old_num_classes] = old_linear.weight
        if has_bias and old_linear.bias is not None and new_linear.bias is not None:
            new_linear.bias[:old_num_classes] = old_linear.bias

        for idx in range(old_num_classes, new_num_classes):
            c_name = class_names[idx] if class_names and idx < len(class_names) else None
            w_row, b_val = init_deterministic_classifier_row(
                in_features=in_features,
                class_name=c_name,
                row_idx=idx,
                dtype=old_linear.weight.dtype,
                device=old_linear.weight.device,
                has_bias=has_bias,
            )
            new_linear.weight[idx] = w_row
            if has_bias and new_linear.bias is not None and b_val is not None:
                new_linear.bias[idx] = b_val

    parts = last_linear_name.split(".")
    parent: Any = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last_p = parts[-1]
    if last_p.isdigit():
        parent[int(last_p)] = new_linear
    else:
        setattr(parent, last_p, new_linear)

    return model, last_linear_name, old_num_classes


def expand_state_dict_classifier(
    state_dict: dict[str, Any],
    new_num_classes: int,
    class_names: list[str] | None = None,
) -> tuple[dict[str, Any], str, int]:
    """Dynamically expand the classification head weight and bias in a state_dict dictionary.

    Deterministically initializes newly added class rows using seeds derived
    from their canonical class names, ensuring bit-for-bit identical initialization
    between client local models and server aggregated states.

    Returns:
        (state_dict, head_weight_key, old_num_classes)
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to expand state_dict classifier.") from exc

    head_prefixes = ("fc.", "classifier.", "head.", "model.head.")
    w_key: str | None = None
    b_key: str | None = None

    for k in state_dict:
        if (
            any(k.startswith(p) for p in head_prefixes)
            and k.endswith(".weight")
            and state_dict[k].dim() == 2
        ):
            w_key = k
            break
    if w_key is None:
        for k in reversed(list(state_dict.keys())):
            if k.endswith(".weight") and state_dict[k].dim() == 2:
                w_key = k
                break

    if w_key is None:
        raise ValueError("Could not identify classification head weight in state_dict.")

    prefix = w_key[: -len(".weight")]
    if f"{prefix}.bias" in state_dict:
        b_key = f"{prefix}.bias"

    old_w = state_dict[w_key]
    old_num_classes, in_features = int(old_w.shape[0]), int(old_w.shape[1])
    if new_num_classes <= old_num_classes:
        return state_dict, w_key, old_num_classes

    new_w = torch.empty((new_num_classes, in_features), dtype=old_w.dtype, device=old_w.device)
    with torch.no_grad():
        new_w[:old_num_classes] = old_w

    new_b: torch.Tensor | None = None
    if b_key is not None:
        old_b = state_dict[b_key]
        new_b = torch.empty((new_num_classes,), dtype=old_b.dtype, device=old_b.device)
        with torch.no_grad():
            new_b[:old_num_classes] = old_b

    with torch.no_grad():
        for idx in range(old_num_classes, new_num_classes):
            c_name = class_names[idx] if class_names and idx < len(class_names) else None
            w_row, b_val = init_deterministic_classifier_row(
                in_features=in_features,
                class_name=c_name,
                row_idx=idx,
                dtype=old_w.dtype,
                device=old_w.device,
                has_bias=(b_key is not None),
            )
            new_w[idx] = w_row
            if new_b is not None and b_val is not None:
                new_b[idx] = b_val

    state_dict[w_key] = new_w
    if b_key is not None and new_b is not None:
        state_dict[b_key] = new_b

    return state_dict, w_key, old_num_classes



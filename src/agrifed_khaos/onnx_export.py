"""Edge Export Utility for Multiplatform Federated Learning in AgriFed-KHAOS.

Deconstructs vision classification models into:
1. `backbone_feature_extractor.onnx`: Feature extractor with frozen BatchNorm,
   optimized for fast WebAssembly / WebGPU execution in onnxruntime-web.
2. `head_weights.json`: Lightweight classification head weights, canonical classes,
   embedding dimension, and preprocessing configuration.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from agrifed_khaos.crypto import compute_sha256
from agrifed_khaos.models import build_model, load_checkpoint_bundle
from agrifed_khaos.wrappers import freeze_bn_stats

logger = logging.getLogger("agrifed.onnx_export")

HEAD_PREFIXES = ("fc.", "classifier.", "head.", "model.head.", "linear.")


@dataclass(slots=True)
class EdgeModelBundleResult:
    """Descriptor of exported edge artifacts synchronized by base model hash."""

    model_slug: str
    base_model_sha256: str
    architecture: str
    num_classes: int
    classes: list[str]
    embedding_dim: int
    onnx_path: Path
    head_weights_path: Path
    preprocessing: dict[str, Any] = field(default_factory=dict)
    cached: bool = False


def isolate_backbone_and_head(
    model: nn.Module,
    architecture: str,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, str, str]:
    """Isolate the classification head from the feature backbone.

    Replaces the classification layer with nn.Identity() and returns:
    (backbone, head_weight, head_bias, weight_key, bias_key).
    """
    arch = architecture.lower().strip()
    backbone = copy.deepcopy(model)
    backbone.eval()
    freeze_bn_stats(backbone)

    # 1. Native Torchvision ResNets (model.fc)
    if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
        w = backbone.fc.weight.detach().clone()
        b = (
            backbone.fc.bias.detach().clone()
            if backbone.fc.bias is not None
            else torch.zeros(w.shape[0])
        )
        backbone.fc = nn.Identity()
        return backbone, w, b, "fc.weight", "fc.bias"

    # 2. Native EfficientNets (model.classifier[1])
    if (
        hasattr(backbone, "classifier")
        and isinstance(backbone.classifier, (nn.Sequential, list, tuple))
        and len(backbone.classifier) > 1
        and isinstance(backbone.classifier[1], nn.Linear)
    ):
        lin = backbone.classifier[1]
        w = lin.weight.detach().clone()
        b = lin.bias.detach().clone() if lin.bias is not None else torch.zeros(w.shape[0])
        backbone.classifier[1] = nn.Identity()
        return backbone, w, b, "classifier.1.weight", "classifier.1.bias"

    # 3. Native MobileNetV3 (model.classifier[3])
    if (
        hasattr(backbone, "classifier")
        and isinstance(backbone.classifier, (nn.Sequential, list, tuple))
        and len(backbone.classifier) > 3
        and isinstance(backbone.classifier[3], nn.Linear)
    ):
        lin = backbone.classifier[3]
        w = lin.weight.detach().clone()
        b = lin.bias.detach().clone() if lin.bias is not None else torch.zeros(w.shape[0])
        backbone.classifier[3] = nn.Identity()
        return backbone, w, b, "classifier.3.weight", "classifier.3.bias"

    # 4. Single Linear Classifier fallback (model.classifier or model.head)
    for attr in ("classifier", "head", "heads"):
        if hasattr(backbone, attr):
            sub = getattr(backbone, attr)
            if isinstance(sub, nn.Linear):
                w = sub.weight.detach().clone()
                b = sub.bias.detach().clone() if sub.bias is not None else torch.zeros(w.shape[0])
                setattr(backbone, attr, nn.Identity())
                return backbone, w, b, f"{attr}.weight", f"{attr}.bias"
            if isinstance(sub, nn.Sequential) and len(sub) > 0 and isinstance(sub[-1], nn.Linear):
                lin = sub[-1]
                w = lin.weight.detach().clone()
                b = lin.bias.detach().clone() if lin.bias is not None else torch.zeros(w.shape[0])
                sub[-1] = nn.Identity()
                idx = len(sub) - 1
                return backbone, w, b, f"{attr}.{idx}.weight", f"{attr}.{idx}.bias"

    # 5. Timm / General model inspection: find the last nn.Linear
    last_linear_name: str | None = None
    last_linear_mod: nn.Linear | None = None
    for name, mod in backbone.named_modules():
        if isinstance(mod, nn.Linear):
            last_linear_name = name
            last_linear_mod = mod

    if last_linear_name is not None and last_linear_mod is not None:
        w = last_linear_mod.weight.detach().clone()
        b = (
            last_linear_mod.bias.detach().clone()
            if last_linear_mod.bias is not None
            else torch.zeros(w.shape[0])
        )
        # Replace using parent hierarchy
        parts = last_linear_name.split(".")
        parent: Any = backbone
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        last_part = parts[-1]
        if last_part.isdigit():
            parent[int(last_part)] = nn.Identity()
        else:
            setattr(parent, last_part, nn.Identity())
        return (
            backbone,
            w,
            b,
            f"{last_linear_name}.weight",
            f"{last_linear_name}.bias",
        )

    raise ValueError(
        f"Unable to isolate classification head for architecture '{arch}'. "
        "No terminal nn.Linear layer discovered."
    )


def export_edge_model_bundle(
    checkpoint_path: Path,
    output_dir: Path,
    model_slug: str = "model",
    metadata_path: Path | None = None,
    opset_version: int = 17,
) -> EdgeModelBundleResult:
    """Export a vision checkpoint into a synchronized ONNX backbone and JSON head."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    base_sha256 = compute_sha256(checkpoint_path)
    # Safe loading only (weights_only=True): checkpoints may originate from federated artifacts, so
    # pickle fallback (CWE-502) must never be enabled here. Detection (YOLO) checkpoints are rejected
    # upstream by the endpoint guard and are not exportable as classification edge bundles.
    state_dict, bundle_meta = load_checkpoint_bundle(checkpoint_path, device="cpu")

    num_classes = bundle_meta.get("num_classes")
    classes = bundle_meta.get("classes")
    arch = bundle_meta.get("architecture")
    task = bundle_meta.get("task")
    if task == "detection" or (arch and "yolo" in str(arch).lower()):
        raise ValueError(
            f"Edge model bundle export is only supported for vision classification architectures, "
            f"not object detection models (arch='{arch}', task='{task}')."
        )
    img_size = bundle_meta.get("img_size", 224)

    if metadata_path and metadata_path.is_file():
        try:
            m_data = json.loads(metadata_path.read_text(encoding="utf-8"))
            if num_classes is None:
                num_classes = m_data.get("num_classes")
            if classes is None:
                classes = m_data.get("classes")
            if not arch:
                arch = m_data.get("architecture")
            if "img_size" in m_data:
                img_size = m_data["img_size"]
            calibrated_t = m_data.get("temperature", 1.0)
            ood_thresh = m_data.get("ood_threshold")
        except Exception as exc:
            logger.warning("Failed loading sidecar metadata %s: %s", metadata_path, exc)
            calibrated_t = 1.0
            ood_thresh = None
    else:
        calibrated_t = bundle_meta.get("temperature", 1.0)
        ood_thresh = bundle_meta.get("ood_threshold")

    arch = arch or "resnet18"
    num_classes = num_classes or (len(classes) if classes else 9)
    if classes is None:
        classes = [f"class_{i}" for i in range(num_classes)]

    model = build_model(architecture=arch, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    backbone, head_w, head_b, w_key, b_key = isolate_backbone_and_head(model, architecture=arch)

    # Numerical verification on dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        original_logits = model(dummy_input)
        embedding = backbone(dummy_input)
        if embedding.dim() > 2:
            embedding = torch.flatten(embedding, 1)
        reconstructed_logits = embedding @ head_w.T + head_b
        max_diff = float((original_logits - reconstructed_logits).abs().max().item())

    if max_diff > 1e-4:
        logger.warning(
            "Numerical divergence detected during head extraction for %s: %e", arch, max_diff
        )
    else:
        logger.debug("Head extraction exact: max diff %e for %s", max_diff, arch)

    embedding_dim = int(embedding.shape[1])

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="agrifed_edge_bundle_"))
    temp_onnx = temp_dir / "backbone_feature_extractor.onnx"
    temp_head_json = temp_dir / "head_weights.json"

    try:
        # Wrap backbone to guarantee flattened 2D embedding output [batch, D]
        class _FlattenedBackbone(nn.Module):
            def __init__(self, bb: nn.Module) -> None:
                super().__init__()
                self.bb = bb

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.bb(x)
                if out.dim() > 2:
                    out = torch.flatten(out, 1)
                return out

        wrapped_bb = _FlattenedBackbone(backbone)
        wrapped_bb.eval()

        export_kwargs: dict[str, Any] = {
            "export_params": True,
            "opset_version": opset_version,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": ["embedding"],
            "dynamic_axes": {
                "input": {0: "batch_size"},
                "embedding": {0: "batch_size"},
            },
        }
        # Explicit legacy TorchScript exporter avoids dynamo/onnxscript compatibility issues
        try:
            torch.onnx.export(
                wrapped_bb,
                dummy_input,
                str(temp_onnx),
                dynamo=False,
                **export_kwargs,
            )
        except TypeError:
            torch.onnx.export(
                wrapped_bb,
                dummy_input,
                str(temp_onnx),
                **export_kwargs,
            )

        preprocessing_meta = {
            "input_size": [img_size, img_size],
            "resize_shortest_edge": 256,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "temperature": float(calibrated_t or 1.0),
            "ood_threshold": float(ood_thresh) if ood_thresh is not None else None,
        }

        head_payload = {
            "schema_version": "1.0",
            "model_slug": model_slug,
            "base_model_sha256": base_sha256,
            "architecture": arch,
            "num_classes": num_classes,
            "classes": classes,
            "embedding_dim": embedding_dim,
            "head_tensor_keys": {
                "weight": w_key,
                "bias": b_key,
            },
            "preprocessing": preprocessing_meta,
            "head_weights": {
                "weight": head_w.cpu().tolist(),
                "bias": head_b.cpu().tolist(),
            },
        }

        temp_head_json.write_text(json.dumps(head_payload, indent=2), encoding="utf-8")

        # Atomic commit into final output_dir
        dest_onnx = output_dir / "backbone_feature_extractor.onnx"
        dest_head = output_dir / "head_weights.json"

        shutil.move(str(temp_onnx), str(dest_onnx))
        shutil.move(str(temp_head_json), str(dest_head))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return EdgeModelBundleResult(
        model_slug=model_slug,
        base_model_sha256=base_sha256,
        architecture=arch,
        num_classes=num_classes,
        classes=classes,
        embedding_dim=embedding_dim,
        onnx_path=dest_onnx,
        head_weights_path=dest_head,
        preprocessing=preprocessing_meta,
        cached=False,
    )


def get_or_export_edge_bundle(
    checkpoint_path: Path,
    model_slug: str,
    models_dir: Path = Path("models"),
    metadata_path: Path | None = None,
) -> EdgeModelBundleResult:
    """Retrieve existing edge bundle from cache or export on-demand."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    current_sha256 = compute_sha256(checkpoint_path)
    bundle_dir = models_dir / "edge_bundles" / model_slug / current_sha256[:16]
    dest_onnx = bundle_dir / "backbone_feature_extractor.onnx"
    dest_head = bundle_dir / "head_weights.json"

    if dest_onnx.is_file() and dest_head.is_file():
        try:
            head_data = json.loads(dest_head.read_text(encoding="utf-8"))
            if head_data.get("base_model_sha256") == current_sha256:
                return EdgeModelBundleResult(
                    model_slug=model_slug,
                    base_model_sha256=current_sha256,
                    architecture=head_data.get("architecture", "resnet18"),
                    num_classes=int(
                        head_data.get("num_classes", len(head_data.get("classes", [])))
                    ),
                    classes=list(head_data.get("classes", [])),
                    embedding_dim=int(head_data.get("embedding_dim", 512)),
                    onnx_path=dest_onnx,
                    head_weights_path=dest_head,
                    preprocessing=head_data.get("preprocessing", {}),
                    cached=True,
                )
        except Exception as exc:
            logger.warning("Corrupt edge bundle cache in %s: %s; re-exporting.", bundle_dir, exc)

    return export_edge_model_bundle(
        checkpoint_path=checkpoint_path,
        output_dir=bundle_dir,
        model_slug=model_slug,
        metadata_path=metadata_path,
    )

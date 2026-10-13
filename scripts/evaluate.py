#!/usr/bin/env python3
"""Domain-Agnostic Evaluator for AgriFed-KHAOS.

Evaluates trained global checkpoints on field test datasets or single image samples.
Generates performance metrics, confusion matrices, and Grad-CAM attention visualizations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (  # type: ignore[import-untyped]
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgriFed Domain-Agnostic Model Evaluator")
    parser.add_argument(
        "--model", type=str, default="models/global_model.pt", help="Path to model weights (.pt)"
    )
    parser.add_argument(
        "--metadata", type=str, default="models/metadata.json", help="Path to metadata json"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Path to evaluation test split directory"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to a single image file for inference"
    )
    parser.add_argument(
        "--out-dir", type=str, default="docs/images", help="Output directory for plots"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    return parser.parse_args()


def get_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return cast(nn.Module, model)


def get_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.target_layer.register_forward_hook(self._fwd)
        self.target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self.activations = out

    def _bwd(self, module: nn.Module, gin: Any, gout: tuple[torch.Tensor, ...]) -> None:
        self.gradients = gout[0]

    def generate(
        self, x: torch.Tensor, class_idx: int | None = None
    ) -> tuple[np.ndarray, int, float]:
        self.model.eval()
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()
        assert self.gradients is not None
        assert self.activations is not None

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.relu(torch.sum(weights * self.activations, dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam[0, 0].detach().cpu().numpy(), class_idx, float(probs[0, class_idx].item())


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[!] Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Resolve classes from metadata or dataset
    classes: list[str] = []
    meta_path = Path(args.metadata)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        classes = meta.get("classes", [])

    if not classes and args.data_dir:
        classes = sorted([d.name for d in Path(args.data_dir).iterdir() if d.is_dir()])

    num_classes = len(classes) if classes else 9
    model = get_model(num_classes=num_classes).to(device)
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    eval_tf = get_transforms()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("AGRIFED-KHAOS: MODEL EVALUATION & INFERENCE")
    print("=" * 65)
    print(f"[*] Model: {model_path} ({num_classes} classes)")

    # Mode A: Evaluate Single Image
    if args.image:
        img_path = Path(args.image)
        if not img_path.is_file():
            print(f"[!] Error: Image not found: {img_path}", file=sys.stderr)
            sys.exit(1)

        raw_img = Image.open(str(img_path)).convert("RGB")
        tensor = eval_tf(raw_img).unsqueeze(0).to(device)

        target_layer = cast(nn.Module, cast(Any, model).layer4[-1])
        cam = GradCAM(model, target_layer)
        mask, pred_idx, conf = cam.generate(tensor)
        pred_label = (
            classes[pred_idx] if classes and pred_idx < len(classes) else f"Class_{pred_idx}"
        )

        print(f"[*] Image: {img_path.name}")
        print(f"[*] Predicted:  {pred_label} (Confidence: {conf * 100:.2f}%)")

        # Render 3-panel figure
        raw_np = np.array(raw_img.resize((224, 224))) / 255.0
        heat = plt.get_cmap("jet")(mask)[:, :, :3]
        overlay = np.clip(0.55 * raw_np + 0.45 * heat, 0, 1)
        bin_mask = (mask > 0.45).astype(np.float32)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        axes[0].imshow(raw_np)
        axes[0].set_title(f"Input Sample: {img_path.name}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(
            f"Grad-CAM Attention ({pred_label}: {conf * 100:.1f}%)", fontsize=11, fontweight="bold"
        )
        axes[1].axis("off")

        axes[2].imshow(bin_mask, cmap="gray")
        axes[2].set_title("Region of Interest (ROI) Mask", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        plot_file = out_dir / "inference_sample.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[*] Visual explanation saved: {plot_file}")

    # Mode B: Full Dataset Evaluation & Confusion Matrix
    if args.data_dir:
        data_path = Path(args.data_dir)
        dataset = datasets.ImageFolder(str(data_path), transform=eval_tf)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        total_loss = 0.0
        correct = 0
        total = 0
        y_true: list[int] = []
        y_pred: list[int] = []
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        avg_loss = total_loss / max(total, 1)
        acc = (correct / max(total, 1)) * 100
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        print(f"\n[*] Evaluated {total} samples in {data_path}")
        print(f"  • Accuracy:        {acc:.2f}%")
        print(f"  • Cross-Ent Loss:  {avg_loss:.4f}")
        print(f"  • Macro F1-Score:  {f1 * 100:.2f}%")
        print(f"  • Macro Precision: {prec * 100:.2f}%")
        print(f"  • Macro Recall:    {rec * 100:.2f}%")

        # Confusion matrix plot
        class_names = dataset.classes
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Clase Predicha", fontweight="bold")
        ax.set_ylabel("Clase Real (Ground Truth)", fontweight="bold")
        ax.set_title("Matriz de Confusión - Evaluación Test Set", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        cm_file = out_dir / "confusion_matrix.png"
        plt.savefig(cm_file, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[*] Confusion matrix saved: {cm_file}")

    print("=" * 65)


if __name__ == "__main__":
    main()

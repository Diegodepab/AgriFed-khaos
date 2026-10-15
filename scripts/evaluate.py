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
from typing import Any

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
from torchvision import datasets, transforms

from agrifed_khaos.quality import (
    CanaryAlarmLevel,
    CanaryAlarmReport,
    CanaryRoundRecord,
    audit_canary_telemetry,
    load_canary_history,
    save_canary_history,
)


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
    parser.add_argument(
        "--canary-history",
        type=str,
        default=None,
        help="Path to JSON file tracking historical round telemetry for canary alarm monitoring",
    )
    parser.add_argument(
        "--round-id",
        type=str,
        default=None,
        help="Identifier for this federated round (e.g. round_0, round_1, round_11)",
    )
    parser.add_argument(
        "--warn-drop",
        type=float,
        default=0.08,
        help="Accuracy drop threshold to trigger WARNING alarm (default: 0.08 = 8%%)",
    )
    parser.add_argument(
        "--critical-drop",
        type=float,
        default=0.20,
        help="Accuracy drop threshold to trigger CRITICAL incident alarm (default: 0.20 = 20%%)",
    )
    parser.add_argument(
        "--fail-on-alarm",
        action="store_true",
        help="Exit with non-zero status code if a CRITICAL canary alarm is triggered",
    )
    parser.add_argument(
        "--calibrate-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute calibrated decision thresholds (optimal F1 and Youden's J)",
    )
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=None,
        help="Explicit decision threshold for binary classification (default: argmax/0.50)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Model architecture (default: resolved from metadata.json or 'resnet18')",
    )
    return parser.parse_args()


def get_transforms(img_size: tuple[int, int] | int = (224, 224)) -> transforms.Compose:
    size = (img_size, img_size) if isinstance(img_size, int) else img_size
    return transforms.Compose(
        [
            transforms.Resize(size),
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

        act = self.activations
        grad = self.gradients

        if act.dim() == 4:
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            cam = torch.relu(torch.sum(weights * act, dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        elif act.dim() == 3:
            # Transformer tokens: [B, N, D] (e.g. 197 = 1 CLS token + 14x14 patches)
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


def calibrate_binary_thresholds(
    y_true: list[int],
    y_probs: list[list[float]],
    positive_class_idx: int = 1,
) -> dict[str, Any]:
    """Sweep decision thresholds for binary classification to optimize F1 and Youden's J."""
    y_arr = np.array(y_true, dtype=int)
    probs_arr = np.array(y_probs, dtype=float)
    p_pos = probs_arr[:, positive_class_idx]

    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1 = -1.0
    best_tau_f1 = 0.50
    best_youden = -2.0
    best_tau_youden = 0.50

    nominal_tau = 0.50
    nominal_preds = (p_pos >= nominal_tau).astype(int)
    nom_p, nom_r, nom_f1, _ = precision_recall_fscore_support(
        y_arr, nominal_preds, average="binary", pos_label=positive_class_idx, zero_division=0
    )
    nom_acc = float(np.mean(nominal_preds == y_arr))

    sweep_records = []
    for tau in thresholds:
        preds = (p_pos >= tau).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_arr, preds, average="binary", pos_label=positive_class_idx, zero_division=0
        )
        tpr = float(rec)
        tn = int(np.sum((preds == 0) & (y_arr == 0)))
        actual_neg = int(np.sum(y_arr == 0))
        tnr = float(tn / max(actual_neg, 1))
        youden_j = tpr + tnr - 1.0

        if f1 > best_f1:
            best_f1 = float(f1)
            best_tau_f1 = float(tau)
        if youden_j > best_youden:
            best_youden = float(youden_j)
            best_tau_youden = float(tau)

        sweep_records.append(
            {
                "threshold": round(float(tau), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "youden_j": round(float(youden_j), 4),
            }
        )

    cal_preds_f1 = (p_pos >= best_tau_f1).astype(int)
    cal_p, cal_r, cal_f1, _ = precision_recall_fscore_support(
        y_arr, cal_preds_f1, average="binary", pos_label=positive_class_idx, zero_division=0
    )
    cal_acc = float(np.mean(cal_preds_f1 == y_arr))

    return {
        "nominal": {
            "threshold": nominal_tau,
            "accuracy": round(nom_acc, 4),
            "precision": round(float(nom_p), 4),
            "recall": round(float(nom_r), 4),
            "f1_score": round(float(nom_f1), 4),
        },
        "calibrated_f1": {
            "optimal_threshold": round(best_tau_f1, 4),
            "accuracy": round(cal_acc, 4),
            "precision": round(float(cal_p), 4),
            "recall": round(float(cal_r), 4),
            "f1_score": round(float(cal_f1), 4),
        },
        "calibrated_youden": {
            "optimal_threshold": round(best_tau_youden, 4),
            "youden_j": round(best_youden, 4),
        },
        "sweep": sweep_records,
    }


def calibrate_multiclass_ovr_thresholds(
    y_true: list[int],
    y_probs: list[list[float]],
    class_names: list[str],
) -> dict[str, Any]:
    """Perform One-vs-Rest (OvR) decision threshold sweep per class."""
    y_arr = np.array(y_true, dtype=int)
    probs_arr = np.array(y_probs, dtype=float)
    thresholds = np.linspace(0.05, 0.95, 91)
    results: dict[str, Any] = {}

    for c_idx, c_name in enumerate(class_names):
        p_c = probs_arr[:, c_idx]
        y_c = (y_arr == c_idx).astype(int)

        best_f1 = -1.0
        best_tau = 0.50
        for tau in thresholds:
            preds_c = (p_c >= tau).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_c, preds_c, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = float(f1)
                best_tau = float(tau)

        nom_preds = (p_c >= 0.50).astype(int)
        nom_p, nom_r, nom_f1, _ = precision_recall_fscore_support(
            y_c, nom_preds, average="binary", zero_division=0
        )
        cal_preds = (p_c >= best_tau).astype(int)
        cal_p, cal_r, cal_f1, _ = precision_recall_fscore_support(
            y_c, cal_preds, average="binary", zero_division=0
        )

        results[c_name] = {
            "nominal_threshold": 0.50,
            "nominal_f1": round(float(nom_f1), 4),
            "optimal_threshold": round(best_tau, 4),
            "calibrated_f1": round(float(cal_f1), 4),
            "calibrated_precision": round(float(cal_p), 4),
            "calibrated_recall": round(float(cal_r), 4),
        }
    return results


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[!] Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Resolve classes and architecture from bundle, metadata, or dataset
    from agrifed_khaos.models import (
        build_model,
        get_cam_target_layer,
        load_checkpoint_bundle,
        save_checkpoint_bundle,
    )

    state, bundle_meta = load_checkpoint_bundle(model_path, device=device)

    classes: list[str] = bundle_meta.get("classes", [])
    arch = args.arch or bundle_meta.get("architecture")
    meta_path = Path(args.metadata)
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not classes:
                classes = meta.get("classes", [])
            if not arch:
                arch = meta.get("architecture")
        except Exception:
            meta = {}

    if not classes and args.data_dir:
        classes = sorted([d.name for d in Path(args.data_dir).iterdir() if d.is_dir()])

    num_classes = len(classes) if classes else (bundle_meta.get("num_classes") or 9)
    arch = arch or "resnet18"
    model = build_model(architecture=arch, num_classes=num_classes).to(device)
    model.load_state_dict(state)
    model.eval()

    raw_img_size = bundle_meta.get("img_size") or (meta.get("img_size") if meta else 224)
    if isinstance(raw_img_size, int):
        img_size_tuple = (raw_img_size, raw_img_size)
    elif isinstance(raw_img_size, (list, tuple)) and len(raw_img_size) == 2:
        img_size_tuple = (int(raw_img_size[0]), int(raw_img_size[1]))
    else:
        img_size_tuple = (224, 224)

    eval_tf = get_transforms(img_size=img_size_tuple)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("AGRIFED-KHAOS: MODEL EVALUATION & INFERENCE")
    print("=" * 65)
    print(f"[*] Model: {model_path} ({arch}, {num_classes} classes)")

    # Mode A: Evaluate Single Image
    if args.image:
        img_path = Path(args.image)
        if not img_path.is_file():
            print(f"[!] Error: Image not found: {img_path}", file=sys.stderr)
            sys.exit(1)

        raw_img = Image.open(str(img_path)).convert("RGB")
        tensor = eval_tf(raw_img).unsqueeze(0).to(device)

        target_layer = get_cam_target_layer(model, architecture=arch)
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
        if classes:
            canonical_set = set(classes)
            dataset.samples = [
                (p, classes.index(dataset.classes[t]))
                for p, t in dataset.samples
                if dataset.classes[t] in canonical_set
            ]
            dataset.targets = [t for _, t in dataset.samples]
            class_names = list(classes)
        else:
            class_names = list(dataset.classes)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        total_loss = 0.0
        correct = 0
        total = 0
        y_true: list[int] = []
        y_pred: list[int] = []
        y_probs: list[list[float]] = []
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * x.size(0)
                probs = F.softmax(out, dim=1)
                if args.eval_threshold is not None and out.shape[1] == 2:
                    preds = (probs[:, 1] >= args.eval_threshold).long()
                else:
                    preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_probs.extend(probs.cpu().tolist())

        avg_loss = total_loss / max(total, 1)
        acc = (correct / max(total, 1)) * 100

        # Macro metrics
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Per-class metrics
        class_prec, class_rec, class_f1, class_supp = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=list(range(len(class_names))), zero_division=0
        )

        # 1. Rich Table Display for Supreme Readability
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(
                title=f"EVALUACIÓN DETALLADA POR CLASE ({data_path.name.upper()})",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Índice", justify="right", style="dim")
            table.add_column("Clase / Patología", style="bold")
            table.add_column("Precision", justify="right")
            table.add_column("Recall", justify="right")
            table.add_column("F1-Score", justify="right")
            table.add_column("Muestras", justify="right", style="green")

            for i, cname in enumerate(class_names):
                table.add_row(
                    str(i),
                    cname,
                    f"{class_prec[i] * 100:.2f}%",
                    f"{class_rec[i] * 100:.2f}%",
                    f"{class_f1[i] * 100:.2f}%",
                    str(class_supp[i]),
                )

            table.add_section()
            table.add_row(
                "-",
                "PROMEDIO MACRO",
                f"{macro_prec * 100:.2f}%",
                f"{macro_rec * 100:.2f}%",
                f"{macro_f1 * 100:.2f}%",
                str(total),
                style="bold yellow",
            )
            table.add_row(
                "-",
                "PROMEDIO PONDERADO",
                f"{weighted_prec * 100:.2f}%",
                f"{weighted_rec * 100:.2f}%",
                f"{weighted_f1 * 100:.2f}%",
                str(total),
                style="bold green",
            )
            console.print(table)
        except ImportError:
            print(f"\n[*] Evaluated {total} samples in {data_path}")
            print(f"  • Accuracy:        {acc:.2f}%")
            print(f"  • Cross-Ent Loss:  {avg_loss:.4f}")
            print(f"  • Macro F1-Score:  {macro_f1 * 100:.2f}%")
            print(f"  • Macro Precision: {macro_prec * 100:.2f}%")
            print(f"  • Macro Recall:    {macro_rec * 100:.2f}%")

        # 2. Dual Confusion Matrix Plot (Raw Counts + Row-Normalized %)
        cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))

        sns.heatmap(
            cm_raw,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax1,
        )
        ax1.set_xlabel("Clase Predicha", fontweight="bold")
        ax1.set_ylabel("Clase Real (Ground Truth)", fontweight="bold")
        ax1.set_title(f"Conteos Absolutos (Acc: {acc:.2f}%)", fontweight="bold")
        ax1.tick_params(axis="x", rotation=45)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Greens",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax2,
        )
        ax2.set_xlabel("Clase Predicha", fontweight="bold")
        ax2.set_ylabel("Clase Real (Ground Truth)", fontweight="bold")
        ax2.set_title("Normalizada por Fila (% Recall por Clase)", fontweight="bold")
        ax2.tick_params(axis="x", rotation=45)

        plt.suptitle(
            f"Matriz de Confusión Dual - Evaluación ({data_path.name})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        cm_file = out_dir / "confusion_matrix.png"
        plt.savefig(cm_file, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[*] Confusion matrix plot saved: {cm_file}")

        # 3. Decision Threshold Calibration (Optimal F1 & Youden's J)
        calibration_results: dict[str, Any] | None = None
        if args.calibrate_threshold and len(class_names) == 2:
            calibration_results = calibrate_binary_thresholds(y_true, y_probs, positive_class_idx=1)
            nom = calibration_results["nominal"]
            cal_f1 = calibration_results["calibrated_f1"]
            cal_j = calibration_results["calibrated_youden"]

            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                cal_table = Table(
                    title="CALIBRACIÓN DE UMBRAL DE DECISIÓN (OPTIMIZACIÓN F1 & YOUDEN'S J)",
                    show_header=True,
                    header_style="bold magenta",
                )
                cal_table.add_column("Régimen de Decisión", style="bold")
                cal_table.add_column("Umbral (τ)", justify="right")
                cal_table.add_column("Accuracy", justify="right")
                cal_table.add_column("Precision", justify="right")
                cal_table.add_column("Recall", justify="right")
                cal_table.add_column("F1-Score", justify="right", style="bold green")

                cal_table.add_row(
                    "Nominal (Default Argmax)",
                    f"{nom['threshold']:.2f}",
                    f"{nom['accuracy'] * 100:.2f}%",
                    f"{nom['precision'] * 100:.2f}%",
                    f"{nom['recall'] * 100:.2f}%",
                    f"{nom['f1_score'] * 100:.2f}%",
                )
                cal_table.add_row(
                    "Calibrado (Óptimo F1)",
                    f"{cal_f1['optimal_threshold']:.2f}",
                    f"{cal_f1['accuracy'] * 100:.2f}%",
                    f"{cal_f1['precision'] * 100:.2f}%",
                    f"{cal_f1['recall'] * 100:.2f}%",
                    f"{cal_f1['f1_score'] * 100:.2f}%",
                    style="bold yellow",
                )
                cal_table.add_row(
                    "Calibrado (Óptimo Youden's J)",
                    f"{cal_j['optimal_threshold']:.2f}",
                    "-",
                    "-",
                    "-",
                    f"J = {cal_j['youden_j']:.3f}",
                    style="dim cyan",
                )
                console.print(cal_table)
            except ImportError:
                print(
                    f"[*] Calibrated Threshold: Nominal F1={nom['f1_score']*100:.2f}% (τ=0.50) -> "
                    f"Calibrated F1={cal_f1['f1_score']*100:.2f}% "
                    f"(τ={cal_f1['optimal_threshold']:.2f})"
                )

            # Generate Threshold Calibration Sweep Plot
            sweep = calibration_results["sweep"]
            th_vals = [s["threshold"] for s in sweep]
            f1_vals = [s["f1"] for s in sweep]
            p_vals = [s["precision"] for s in sweep]
            r_vals = [s["recall"] for s in sweep]
            j_vals = [s["youden_j"] for s in sweep]

            plt.figure(figsize=(10, 5.5))
            plt.plot(th_vals, f1_vals, label="F1-Score", color="green", linewidth=2.5)
            plt.plot(
                th_vals, p_vals, label="Precision", color="blue", linestyle="--", linewidth=1.5
            )
            plt.plot(
                th_vals,
                r_vals,
                label="Recall (Sensitivity)",
                color="orange",
                linestyle="--",
                linewidth=1.5,
            )
            plt.plot(
                th_vals,
                j_vals,
                label="Youden's J Index",
                color="purple",
                linestyle=":",
                linewidth=2,
            )
            plt.axvline(0.50, color="gray", linestyle="-.", label="Nominal (τ=0.50)")
            plt.axvline(
                cal_f1["optimal_threshold"],
                color="red",
                linestyle="-",
                label=f"Óptimo F1 (τ={cal_f1['optimal_threshold']:.2f})",
            )
            plt.xlabel("Umbral de Decisión (τ)", fontweight="bold")
            plt.ylabel("Métrica", fontweight="bold")
            plt.title(
                f"Curva de Calibración de Umbral de Decisión ({data_path.name})", fontweight="bold"
            )
            plt.grid(True, alpha=0.3)
            plt.legend(loc="lower left")
            plt.tight_layout()
            curve_file = out_dir / "threshold_calibration_curve.png"
            plt.savefig(curve_file, dpi=180, bbox_inches="tight")
            plt.close()
            print(f"[*] Threshold calibration curve saved: {curve_file}")

        elif args.calibrate_threshold and len(class_names) > 2:
            calibration_results = calibrate_multiclass_ovr_thresholds(y_true, y_probs, class_names)
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                cal_table = Table(
                    title="CALIBRACIÓN MULTICLASE ONE-VS-REST (ÓPTIMO F1 POR CLASE)",
                    show_header=True,
                    header_style="bold magenta",
                )
                cal_table.add_column("Clase", style="bold")
                cal_table.add_column("Nominal F1 (τ=0.50)", justify="right")
                cal_table.add_column("Umbral Óptimo (τ*)", justify="right", style="bold yellow")
                cal_table.add_column("Calibrated F1", justify="right", style="bold green")

                for c_name, c_res in calibration_results.items():
                    cal_table.add_row(
                        c_name,
                        f"{c_res['nominal_f1'] * 100:.2f}%",
                        f"{c_res['optimal_threshold']:.2f}",
                        f"{c_res['calibrated_f1'] * 100:.2f}%",
                    )
                console.print(cal_table)
            except ImportError:
                print(
                    f"[*] Multiclass OvR Threshold Calibration computed for "
                    f"{len(class_names)} classes."
                )

        if calibration_results is not None:
            if len(class_names) == 2:
                cal_f1_val = calibration_results.get("calibrated_f1", {})
                cal_j_val = calibration_results.get("calibrated_youden", {})
                tau_map = {
                    class_names[0]: 0.50,
                    class_names[1]: float(cal_f1_val.get("optimal_threshold", 0.50)),
                    "calibrated_f1": cal_f1_val,
                    "calibrated_youden": cal_j_val,
                }
            else:
                tau_map = {
                    c_name: float(c_res["optimal_threshold"])
                    for c_name, c_res in calibration_results.items()
                }

            bundle_meta["calibrated_thresholds"] = tau_map
            save_checkpoint_bundle(model_path, state, bundle_meta)
            print(f"[*] Injected calibrated thresholds into checkpoint bundle: {model_path}")
            if meta_path.is_file():
                try:
                    meta["calibrated_thresholds"] = tau_map
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    print(f"[*] Updated metadata mirror with calibrated thresholds: {meta_path}")
                except Exception as exc:
                    print(f"[!] Warning: Failed to update metadata mirror: {exc}", file=sys.stderr)

        # 4. Export Comprehensive JSON Evaluation Report
        report_data: dict[str, Any] = {
            "dataset_evaluated": str(data_path),
            "total_samples": total,
            "overall_accuracy": round(acc, 4),
            "cross_entropy_loss": round(avg_loss, 4),
            "macro_avg": {
                "precision": round(float(macro_prec), 4),
                "recall": round(float(macro_rec), 4),
                "f1_score": round(float(macro_f1), 4),
            },
            "weighted_avg": {
                "precision": round(float(weighted_prec), 4),
                "recall": round(float(weighted_rec), 4),
                "f1_score": round(float(weighted_f1), 4),
            },
            "per_class": {
                cname: {
                    "precision": round(float(class_prec[i]), 4),
                    "recall": round(float(class_rec[i]), 4),
                    "f1_score": round(float(class_f1[i]), 4),
                    "support": int(class_supp[i]),
                }
                for i, cname in enumerate(class_names)
            },
            "confusion_matrix_raw": cm_raw.tolist(),
            "calibration": calibration_results,
        }

        # 4. Canary Telemetry Watchdog & Incident Alarm Check
        canary_report_dict: dict[str, Any] | None = None
        critical_alarm_fired = False
        if args.canary_history:
            history_path = Path(args.canary_history)
            history = load_canary_history(history_path)
            round_id = args.round_id or f"round_{len(history)}"

            current_record = CanaryRoundRecord(
                round_id=round_id,
                accuracy=acc / 100.0,
                loss=avg_loss,
                macro_f1=float(macro_f1),
                per_class_f1={cname: float(class_f1[i]) for i, cname in enumerate(class_names)},
            )

            canary_report: CanaryAlarmReport = audit_canary_telemetry(
                current_record=current_record,
                history=history,
                warn_drop_threshold=args.warn_drop,
                critical_drop_threshold=args.critical_drop,
            )

            history.append(current_record)
            save_canary_history(history_path, history)
            canary_report_dict = canary_report.model_dump()
            report_data["canary_alarm_status"] = canary_report_dict

            try:
                from rich.console import Console
                from rich.panel import Panel

                c = Console()
                curr_p = canary_report.current_accuracy * 100.0
                peak_p = canary_report.peak_accuracy * 100.0
                drop_p = canary_report.drop_from_previous * 100.0

                if canary_report.level == CanaryAlarmLevel.CRITICAL:
                    critical_alarm_fired = True
                    alarm_text = "\n".join(f"• {a}" for a in canary_report.alarms)
                    c.print(
                        Panel(
                            f"[bold white]{alarm_text}[/bold white]\n\n"
                            f"[bold yellow]Acc Actual:[/bold yellow] {curr_p:.2f}% | "
                            f"[bold yellow]Pico Histórico:[/bold yellow] {peak_p:.2f}% | "
                            f"[bold yellow]Caída:[/bold yellow] -{drop_p:.2f}%",
                            title="[bold red]🚨 ALERTA CRÍTICA: REGRESIÓN EN CANARIO[/bold red]",
                            border_style="bold red",
                        )
                    )
                elif canary_report.level == CanaryAlarmLevel.WARNING:
                    alarm_text = "\n".join(f"• {a}" for a in canary_report.alarms)
                    c.print(
                        Panel(
                            f"[bold yellow]{alarm_text}[/bold yellow]\n\n"
                            f"[bold white]Acc Actual:[/bold white] {curr_p:.2f}% | "
                            f"[bold white]Pico Histórico:[/bold white] {peak_p:.2f}% | "
                            f"[bold white]Caída:[/bold white] -{drop_p:.2f}%",
                            title="[bold yellow]⚠️ AVISO: DEGRADACIÓN EN TEST CANARIO[/bold yellow]",
                            border_style="yellow",
                        )
                    )
                else:
                    c.print(
                        Panel(
                            f"[green]✓ Nominal ({current_record.round_id}).\n"
                            f"Acc: {curr_p:.2f}% (Pico: {peak_p:.2f}%)[/green]",
                            title="[bold green]CANARIO DE OBSERVABILIDAD: NOMINAL[/bold green]",
                            border_style="green",
                        )
                    )
            except ImportError:
                if canary_report.level != CanaryAlarmLevel.OK:
                    print(f"\n[!] CANARY ALARM TRIGGERED [{canary_report.level}]:")
                    for a in canary_report.alarms:
                        print(f"    - {a}")
                else:
                    print(
                        f"\n[*] Canary Nominal: {canary_report.current_accuracy * 100:.2f}%"
                    )

        report_file = out_dir / "evaluation_report.json"
        report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        print(f"[*] Full structured evaluation report saved: {report_file}")

        if args.fail_on_alarm and critical_alarm_fired:
            print("[!] Halting execution due to CRITICAL canary alarm regression.", file=sys.stderr)
            sys.exit(2)

    print("=" * 65)


if __name__ == "__main__":
    main()

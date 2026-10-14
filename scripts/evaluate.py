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
        "--arch",
        type=str,
        default=None,
        help="Model architecture (default: resolved from metadata.json or 'resnet18')",
    )
    return parser.parse_args()


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

    # 1. Resolve classes and architecture from metadata or dataset
    from agrifed_khaos.models import build_model, get_cam_target_layer

    classes: list[str] = []
    arch = args.arch
    meta_path = Path(args.metadata)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        classes = meta.get("classes", [])
        if not arch:
            arch = meta.get("architecture")

    if not classes and args.data_dir:
        classes = sorted([d.name for d in Path(args.data_dir).iterdir() if d.is_dir()])

    num_classes = len(classes) if classes else 9
    arch = arch or "resnet18"
    model = build_model(architecture=arch, num_classes=num_classes).to(device)
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    eval_tf = get_transforms()
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
        class_names = dataset.classes

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

        # 3. Export Comprehensive JSON Evaluation Report
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

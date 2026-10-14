#!/usr/bin/env python3
"""Hardware Profiler and Federated Model Recommendation CLI for AgriFed-KHAOS.

Analyzes the host machine's compute, memory, and network capabilities to automatically
recommend the optimal Computer Vision architecture and hyperparameters for Federated Learning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agrifed_khaos.hardware import HardwareProfile, ModelSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgriFed Hardware Profiler & Federated Model Selection Advisor"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "detection", "segmentation"],
        default="classification",
        help="Computer vision task (default: classification)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional path to local dataset to inspect class count and samples",
    )
    parser.add_argument(
        "--target-client-tier",
        type=str,
        choices=["cpu-low", "cpu-high", "gpu"],
        default="cpu-low",
        help="Target client hardware tier for federated constraints (default: cpu-low)",
    )
    parser.add_argument(
        "--network-speed-mbit",
        type=float,
        default=50.0,
        help="Estimated client uplink speed in Mbit/s (default: 50.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    # 1. Hardware Detection
    profile = HardwareProfile.detect()

    # 2. Dataset Inspection (if provided)
    num_classes = 0
    total_samples = 0
    class_names: list[str] = []
    if args.data_dir:
        data_p = Path(args.data_dir)
        if data_p.is_dir():
            class_dirs = [d for d in data_p.iterdir() if d.is_dir()]
            num_classes = len(class_dirs)
            class_names = sorted([d.name for d in class_dirs])
            for d in class_dirs:
                total_samples += len(list(d.glob("*.*")))

    # 3. Model Recommendation Engine
    task_literal: Literal["classification", "detection", "segmentation"] = args.task
    candidates = ModelSelector.get_recommendations(
        profile=profile,
        task=task_literal,
        network_speed_mbit=args.network_speed_mbit,
        target_client_tier=args.target_client_tier,
        dataset_sample_count=total_samples if total_samples > 0 else None,
        num_classes=num_classes if num_classes > 0 else None,
    )
    best_candidate = candidates[0]

    # --- RENDER DASHBOARD ---
    console.print()
    banner = (
        "[bold green]AGRIFED-KHAOS: ASESOR ARQUITECTÓNICO Y SELECCIÓN DE MODELOS FL[/bold green]\n"
        "[dim]Análisis de hardware, restricciones perimetrales y recomendación de visión[/dim]"
    )
    console.print(Panel.fit(banner, border_style="green"))

    # Table 1: Hardware & Dataset Profile
    hw_table = Table(title="PERFIL DE RECURSOS DEL SISTEMA", header_style="bold blue")
    hw_table.add_column("Recurso", style="cyan")
    hw_table.add_column("Valor Detectado", style="bold white")
    hw_table.add_column("Diagnóstico MLOps", style="dim")

    hw_table.add_row(
        "Dispositivo Principal",
        profile.device_type.upper(),
        f"Aceleración {profile.device_type.upper()} activa para PyTorch",
    )
    hw_table.add_row(
        "Nombre del Procesador",
        profile.device_name,
        f"{profile.cpu_count} núcleos lógicos detectados",
    )
    hw_table.add_row(
        "Memoria RAM del Sistema",
        f"{profile.memory_ram_gb:.1f} GB",
        "Capacidad adecuada para DataLoaders de visión",
    )
    if profile.device_type == "cuda":
        hw_table.add_row(
            "VRAM GPU Dedicada",
            f"{profile.gpu_memory_vram_gb:.1f} GB ({profile.gpu_count} GPU)",
            f"Compute {profile.cuda_compute_capability} (FP16 Mixed Precision habilitado)",
        )
    else:
        hw_table.add_row(
            "VRAM GPU Dedicada",
            "N/A (Modo CPU puro)",
            "Priorizar modelos compactos (ResNet-18, MobileNetV4, YOLOv8n) para turnos rápidos",
        )

    if args.data_dir:
        hw_table.add_section()
        hw_table.add_row(
            "Dataset Analizado",
            str(args.data_dir),
            f"{total_samples} imágenes en {num_classes} clases ({', '.join(class_names[:3])}...)",
        )

    console.print(hw_table)
    console.print()

    # Table 2: Model Candidates Ranked
    model_table = Table(
        title=f"MATRIZ DE MODELOS CANDIDATOS (TAREA: {args.task.upper()})",
        header_style="bold magenta",
    )
    model_table.add_column("Rank", justify="center", style="dim")
    model_table.add_column("Arquitectura", style="bold")
    model_table.add_column("Familia", style="dim")
    model_table.add_column("Parámetros", justify="right")
    model_table.add_column("Tensor FL", justify="right")
    model_table.add_column("Batch", justify="right")
    model_table.add_column("Score FL", justify="center", style="bold green")
    model_table.add_column("Justificación Arquitectónica")

    for idx, c in enumerate(candidates, start=1):
        score_color = "bold green" if c.federated_suitability_score >= 90 else "yellow"
        model_table.add_row(
            f"#{idx}",
            c.architecture,
            c.family,
            f"{c.param_count_millions:.1f}M",
            f"{c.checkpoint_size_mb:.1f} MB",
            str(c.recommended_batch_size),
            f"[{score_color}]{c.federated_suitability_score}/100[/{score_color}]",
            c.rationale,
        )

    console.print(model_table)
    console.print()

    # Recommendations Panel
    bs = best_candidate.recommended_batch_size
    rec_text = (
        f"[bold yellow]RECOMENDACIÓN FINAL:[/bold yellow]\n"
        f"• Arquitectura: [bold cyan]{best_candidate.architecture}[/bold cyan] "
        f"({best_candidate.family.upper()})\n\n"
        f"• [bold green]Ventajas:[/bold green] {'; '.join(best_candidate.pros)}\n"
        f"• [bold red]Compromisos:[/bold red] {'; '.join(best_candidate.cons)}\n\n"
        f"[bold white]COMANDO SUGERIDO PARA ENTRENAMIENTO BASE (AUTO-FL):[/bold white]\n"
        f"[dim]python scripts/train_base.py --data-dir <DIR> --arch auto "
        f"--target-client-tier {args.target_client_tier} --batch-size {bs} --epochs 3[/dim]"
    )
    console.print(Panel(rec_text, title="DECISIÓN ARQUITECTÓNICA", border_style="yellow"))
    console.print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Empirical Benchmark: Head-Only vs. Full Fine-Tuning across Data Regimes.

Part of AgriFed-KHAOS Multiplatform Implementation (Phase 0).
Demonstrates empirically on OliveFruit v1.1 why lightweight edge/mobile nodes
operate in `head_only` mode while heavy cooperative/silo nodes operate in `full` mode.

Methodological protocol:
- Base model (T0) trained on TRAIN_DIR.
- Simulated client nodes train exclusively on disjoint VAL_DIR (zero data contamination).
- All evaluations (T0 and T1) measured on disjoint TEST_DIR.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from agrifed_khaos.models import load_checkpoint_bundle
from agrifed_khaos.wrappers import BN_BUFFER_SUFFIXES, ParameterValidator


def resolve_dataset_dirs() -> tuple[Path, Path, Path]:
    """Locate OliveFruit v1.1 classification splits across container and host paths."""
    candidates = [
        Path("/data/agriKhaos/datasets/olivefruit_v1.1/classification"),
        Path("../datasets/olivefruit_v1.1/classification"),
        Path("data/olivefruit_v1.1/classification"),
    ]
    for root in candidates:
        train_p = root / "train"
        val_p = root / "val"
        test_p = root / "test"
        if train_p.is_dir() and val_p.is_dir() and test_p.is_dir():
            return train_p, val_p, test_p
    raise FileNotFoundError(
        "Could not find OliveFruit v1.1 classification dataset with train/val/test splits. "
        f"Checked: {[str(c) for c in candidates]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgriFed-KHAOS Phase 0 Benchmark: head_only vs full"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="models/benchmark_phase0_report.json",
        help="Path to save structured JSON report",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        help="Architecture for benchmark (default: resnet18)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.6,
        help="FedProx aggregation server step size (default: 0.6)",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
        help="FedProx proximal penalty parameter (default: 0.01)",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep temporary artifacts instead of cleaning up",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir, val_dir, test_dir = resolve_dataset_dirs()

    work_dir = Path(tempfile.mkdtemp(prefix="agrifed_bench_phase0_"))
    print("=" * 88)
    print("AGRIFED-KHAOS | FASE 0: BENCHMARK EMPIRICO HEAD_ONLY vs FULL")
    print("=" * 88)
    print(f"[*] Dataset Base (TRAIN):   {train_dir}")
    print(f"[*] Dataset Clientes (VAL):  {val_dir} (Disjoint from base training)")
    print(f"[*] Dataset Test (TEST):     {test_dir} (Disjoint from base & clients)")
    print(f"[*] Architecture:           {args.arch}")
    print(f"[*] Work directory:         {work_dir}")
    print("=" * 88)

    # -------------------------------------------------------------------------
    # PASO 1: Entrenar Modelo Base T0 sobre TRAIN_DIR
    # -------------------------------------------------------------------------
    base_pt = work_dir / "base_t0.pt"
    base_meta = work_dir / "base_t0.meta.json"
    print("\n[+] Entrenando Modelo Base T0 en TRAIN_DIR (resnet18, 2 epocas, 400 muestras base)...")
    t0_train_start = time.time()
    subprocess.run(
        [
            sys.executable,
            "scripts/train_base.py",
            "--data-dir",
            str(train_dir),
            "--output",
            str(base_pt),
            "--metadata",
            str(base_meta),
            "--arch",
            args.arch,
            "--epochs",
            "2",
            "--max-samples",
            "400",
            "--batch-size",
            "32",
        ],
        check=True,
    )
    t0_train_sec = time.time() - t0_train_start

    # Evaluar T0 en TEST_DIR
    t0_eval_dir = work_dir / "eval_t0"
    print("[+] Evaluando Modelo Base T0 en TEST_DIR (800 imagenes independientes)...")
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--model",
            str(base_pt),
            "--metadata",
            str(base_meta),
            "--data-dir",
            str(test_dir),
            "--out-dir",
            str(t0_eval_dir),
            "--no-calibrate-threshold",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    t0_report = json.loads((t0_eval_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    t0_acc = t0_report["overall_accuracy"]
    t0_loss = t0_report["cross_entropy_loss"]
    t0_f1 = t0_report["macro_avg"]["f1_score"] * 100
    t0_rec_def = t0_report["per_class"]["defectuosa"]["recall"] * 100
    t0_rec_san = t0_report["per_class"]["sana"]["recall"] * 100

    print("-" * 88)
    print(
        f"[RESULTADO T0 BASE] Accuracy: {t0_acc:.2f}% | Macro F1: {t0_f1:.2f}% | Loss: {t0_loss:.4f}"
    )
    print(
        f"                   Recall defectuosa: {t0_rec_def:.1f}% | Recall sana: {t0_rec_san:.1f}%"
    )
    print("-" * 88)

    base_state, _ = load_checkpoint_bundle(base_pt, device="cpu")

    # -------------------------------------------------------------------------
    # PASO 2: Matriz Experimental Clientes sobre VAL_DIR
    # -------------------------------------------------------------------------
    scenarios = [
        (40, "Móvil 20/clase (40 total)", 16),
        (100, "Móvil 50/clase (100 total)", 16),
        (600, "Cooperativa (600 total)", 32),
    ]
    modes = ["head_only", "full"]

    results_table: list[dict[str, Any]] = []

    for max_samples, profile_name, bs in scenarios:
        for scope in modes:
            tag = f"{max_samples}_{scope}"
            client_pt = work_dir / f"client_{tag}.pt"
            client_meta = work_dir / f"client_{tag}.meta.json"
            global_t1_pt = work_dir / f"global_t1_{tag}.pt"
            global_t1_meta = work_dir / f"global_t1_{tag}.meta.json"
            eval_t1_dir = work_dir / f"eval_t1_{tag}"

            print(f"\n---> Ejecutando: [{profile_name}] | Modo: {scope.upper()}")

            cmd_client = [
                sys.executable,
                "scripts/client_trainer.py",
                "--data-dir",
                str(val_dir),  # Disjoint dataset! Zero data contamination
                "--weights-in",
                str(base_pt),
                "--weights-out",
                str(client_pt),
                "--metadata",
                str(base_meta),
                "--epochs",
                "3",
                "--batch-size",
                str(bs),
                "--mu",
                str(args.mu),
                "--max-samples",
                str(max_samples),
                "--client-id",
                f"client_{tag}",
            ]
            if scope == "head_only":
                cmd_client.append("--head-only")

            t_c_start = time.time()
            proc = subprocess.run(cmd_client, capture_output=True, text=True)
            t_c_sec = time.time() - t_c_start
            if proc.returncode != 0:
                print(f"[!] Error en client_trainer: {proc.stderr}")
                sys.exit(1)

            # Auditar con ParameterValidator
            c_state, _ = load_checkpoint_bundle(client_pt, device="cpu")
            t_keys = [
                k for k in c_state.keys() if not any(k.endswith(s) for s in BN_BUFFER_SUFFIXES)
            ]
            cand_arrs = [c_state[k].numpy() for k in t_keys]
            ref_arrs = [base_state[k].numpy() for k in t_keys]

            validator = ParameterValidator(max_update_norm=150.0, max_relative_norm=0.50)
            val_audit = validator.validate(cand_arrs, ref_arrs)

            # Agregar en T1 con scripts/aggregate.py
            agg_cmd = [
                sys.executable,
                "scripts/aggregate.py",
                "--base-model",
                str(base_pt),
                "--client-weights",
                str(client_pt),
                "--output",
                str(global_t1_pt),
                "--metadata",
                str(base_meta),
                "--eta",
                str(args.eta),
                "--max-relative-norm",
                "0.50",
            ]
            proc_agg = subprocess.run(agg_cmd, capture_output=True, text=True)
            if proc_agg.returncode != 0:
                print(f"[!] Error en aggregate: {proc_agg.stderr}")
                sys.exit(1)

            # Evaluar T1 en TEST_DIR
            subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--model",
                    str(global_t1_pt),
                    "--metadata",
                    str(base_meta),
                    "--data-dir",
                    str(test_dir),
                    "--out-dir",
                    str(eval_t1_dir),
                    "--no-calibrate-threshold",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            t1_rep = json.loads(
                (eval_t1_dir / "evaluation_report.json").read_text(encoding="utf-8")
            )
            t1_acc = t1_rep["overall_accuracy"]
            t1_loss = t1_rep["cross_entropy_loss"]
            t1_f1 = t1_rep["macro_avg"]["f1_score"] * 100
            t1_rec_def = t1_rep["per_class"]["defectuosa"]["recall"] * 100
            t1_rec_san = t1_rep["per_class"]["sana"]["recall"] * 100
            payload_kb = client_pt.stat().st_size / 1024.0

            delta_acc = t1_acc - t0_acc
            delta_f1 = t1_f1 - t0_f1

            row = {
                "profile": profile_name,
                "samples": max_samples,
                "scope": scope,
                "t0_acc": round(t0_acc, 2),
                "t1_acc": round(t1_acc, 2),
                "delta_acc": round(delta_acc, 2),
                "t0_f1": round(t0_f1, 2),
                "t1_f1": round(t1_f1, 2),
                "delta_f1": round(delta_f1, 2),
                "recall_defectuosa": round(t1_rec_def, 2),
                "recall_sana": round(t1_rec_san, 2),
                "loss_t1": round(t1_loss, 4),
                "training_time_sec": round(t_c_sec, 2),
                "payload_kb": round(payload_kb, 1),
                "delta_norm": round(val_audit.delta_norm or 0.0, 4),
                "relative_norm": round(val_audit.relative_norm or 0.0, 4),
                "validator_passed": val_audit.valid,
            }
            results_table.append(row)
            print(
                f"    [OK] T1 Acc: {t1_acc:.2f}% (Δ {delta_acc:+.2f}%) | "
                f"F1: {t1_f1:.2f}% (Δ {delta_f1:+.2f}%) | "
                f"Time: {t_c_sec:.2f}s | Payload: {payload_kb:.1f} KB | "
                f"||ΔW||: {row['delta_norm']} (Rel: {row['relative_norm']})"
            )

    # -------------------------------------------------------------------------
    # PASO 3: Guardar Reporte Estructurado y Mostrar Tabla Comparativa
    # -------------------------------------------------------------------------
    out_report_p = Path(args.output_report)
    out_report_p.parent.mkdir(parents=True, exist_ok=True)
    full_report = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "OliveFruit_v1.1",
        "architecture": args.arch,
        "fedprox_mu": args.mu,
        "server_eta": args.eta,
        "t0_base_metrics": {
            "overall_accuracy": t0_acc,
            "macro_f1": t0_f1,
            "loss": t0_loss,
            "recall_defectuosa": t0_rec_def,
            "recall_sana": t0_rec_san,
            "train_samples": 400,
            "train_split": str(train_dir),
            "eval_split": str(test_dir),
        },
        "results": results_table,
    }
    out_report_p.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    print("\n" + "=" * 115)
    print("TABLA COMPARATIVA FINAL: FASE 0 BENCHMARK (OliveFruit v1.1 - TEST SET N=800)")
    print("=" * 115)
    header = (
        f"{'Escenario':<28} | {'Modo':<10} | {'T0 Acc':<7} | {'T1 Acc':<7} | {'Δ Acc':<7} | "
        f"{'T1 F1':<7} | {'Rec(def/san)':<13} | {'Payload':<10} | {'Tiempo':<8} | {'Norma Rel':<9}"
    )
    print(header)
    print("-" * 115)
    for r in results_table:
        rec_str = f"{r['recall_defectuosa']:.1f}%/{r['recall_sana']:.1f}%"
        payload_str = f"{r['payload_kb']:.1f} KB" if r['payload_kb'] < 1024 else f"{r['payload_kb']/1024:.2f} MB"
        print(
            f"{r['profile']:<28} | {r['scope']:<10} | {r['t0_acc']:>6.2f}% | {r['t1_acc']:>6.2f}% | "
            f"{r['delta_acc']:>+6.2f}% | {r['t1_f1']:>6.2f}% | {rec_str:<13} | {payload_str:<10} | "
            f"{r['training_time_sec']:>6.2f}s | {r['relative_norm']:>9.4f}"
        )
    print("=" * 115)
    print(f"\n[+] Reporte JSON persistido con éxito en: {out_report_p}")

    if not args.keep_workdir:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"[*] Artefactos temporales conservados en: {work_dir}")


if __name__ == "__main__":
    main()

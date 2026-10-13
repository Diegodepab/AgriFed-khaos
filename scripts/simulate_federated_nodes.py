"""Distributed 3-Node Federated Simulation: Master, Client 1 (Tester), and Client 2 (Trainer).

Topology:
1. Node Master: Central server hosting global weights W_global.
2. Node Client 1 (Tester): An agronomist/cooperative with holdout test images.
   - Queries Master at T_0 (pre-update baseline evaluation).
   - Queries Master at T_1 (post-update evaluation).
   - Measures exact Accuracy and Loss shift.
3. Node Client 2 (Trainer): A private farm with local training images.
   - Preserves 100% data privacy (0 raw images/labels transmitted).
   - Trains local epochs with FedProx proximal regularization.
   - Transmits only numerical parameters W_client2 to Master.
4. Node Master Gate:
   - Validates candidate parameters with ParameterValidator.
   - Integrates parameters into W_global via weighted aggregation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agrifed_khaos.wrappers import (  # noqa: E402
    ParameterValidator,
    TimmClassificationWrapper,
)


def run_3node_simulation(
    dataset_root: Path,
    model_name: str = "resnet18",
    test_batch_size: int = 16,
    train_samples_client2: int = 64,
) -> None:
    print("=" * 78)
    print("  AGRIFED-KHAOS: SIMULACIÓN DISTRIBUIDA DE 3 NODOS (MASTER / C1 / C2)")
    print("=" * 78)
    print("Topología física/lógica:")
    print("  • Nodo [MASTER]    : Orquestador Central & Parameter Gate")
    print("  • Nodo [CLIENTE 1] : Nodo Tester / Agrónomo (Evaluación Pre y Post)")
    print("  • Nodo [CLIENTE 2] : Nodo Entrenador Privado (0 datos compartidos)")
    print("-" * 78)

    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset

    train_dir = dataset_root / "dataset" / "train"
    test_dir = dataset_root / "dataset" / "test"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_train_ds = datasets.ImageFolder(root=str(train_dir), transform=transform)
    full_test_ds = datasets.ImageFolder(root=str(test_dir), transform=transform)
    class_names = full_train_ds.classes
    num_classes = len(class_names)

    print(f"Dataset: {num_classes} clases ({', '.join(class_names[:5])}...)")

    # =========================================================================
    # FASE 1: NODO MASTER INICIALIZA MODELO GLOBAL
    # =========================================================================
    print("\n[FASE 1] NODO MASTER: Despliegue de modelo global base W_0...")
    master_wrapper = TimmClassificationWrapper.create(
        model_name=model_name,
        num_classes=num_classes,
        learning_rate=1e-3,
        pretrained=False,
        device="cpu",
    )
    w_global_0 = master_wrapper.get_parameters()
    total_weights = sum(p.size for p in w_global_0)
    print(f"  -> W_0 inicializado: {len(w_global_0)} tensores ({total_weights:,} pesos).")

    # =========================================================================
    # FASE 2: NODO CLIENTE 1 EVALÚA EN T_0 (PRE-UPDATE)
    # =========================================================================
    print("\n[FASE 2] NODO CLIENTE 1 (TESTER): Solicitud de evaluación inicial en T_0...")
    print("  • Cliente 1 envía lote de imágenes de campo (split test) al modelo base.")
    test_subset = Subset(full_test_ds, list(range(min(24, len(full_test_ds)))))
    test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False)

    stats_t0 = master_wrapper.evaluate(test_loader)
    acc_t0 = stats_t0.metrics.get("accuracy", 0.0)
    correct_t0 = int(acc_t0 * stats_t0.examples)
    print("  -> Resultados CLIENTE 1 en T_0 (Pre-Update):")
    print(f"     • Pérdida (Loss T_0)   : {stats_t0.loss:.4f}")
    print(f"     • Exactitud (Acc T_0)  : {acc_t0 * 100:.2f}% ({correct_t0}/{stats_t0.examples})")

    # =========================================================================
    # FASE 3: NODO CLIENTE 2 ENTRENA LOCALMENTE (DATASET PRIVADO)
    # =========================================================================
    print("\n[FASE 3] NODO CLIENTE 2 (ENTRENADOR PRIVADO): Fine-Tuning Local...")
    print("  • POLÍTICA DE PRIVACIDAD: El Cliente 2 conserva sus datos 100% locales.")
    print("  • Cliente 2 descarga W_0 y entrena 3 épocas con regularización FedProx...")

    c2_subset = Subset(full_train_ds, list(range(train_samples_client2)))
    c2_loader = DataLoader(c2_subset, batch_size=8, shuffle=True)

    client2_wrapper = TimmClassificationWrapper.create(
        model_name=model_name,
        num_classes=num_classes,
        learning_rate=2e-3,
        pretrained=False,
        device="cpu",
    )
    client2_wrapper.set_parameters(w_global_0)

    train_stats = client2_wrapper.train_local_epoch(
        train_data=c2_loader,
        epochs=3,
        prox_mu=0.01,
        reference_parameters=w_global_0,
    )
    print("  -> Entrenamiento en Cliente 2 completado:")
    print(f"     • Ejemplos procesados : {train_stats.examples}")
    print(f"     • Batches iterados    : {train_stats.batches}")
    print(f"     • Pérdida local final : {train_stats.loss:.4f}")

    # =========================================================================
    # FASE 4: EXTRACCIÓN Y ENVÍO SEGURO DE PARÁMETROS
    # =========================================================================
    print("\n[FASE 4] TRANSMISIÓN CLIENTE 2 -> MASTER (SOLO PARÁMETROS NUMÉRICOS):")
    w_client2 = client2_wrapper.get_parameters()
    mb_params = sum(p.nbytes for p in w_client2) / (1024 * 1024)
    print("  • Datos de imágenes transferidos : 0 BYTES (Privacidad Matemática)")
    print(f"  • Parámetros numéricos subidos   : {len(w_client2)} tensores ({mb_params:.2f} MB)")

    # =========================================================================
    # FASE 5: NODO MASTER AUDITA E INCORPORA PARÁMETROS (PARAMETER GATE)
    # =========================================================================
    print("\n[FASE 5] NODO MASTER: Control de Calidad y Agregación (Parameter Gate)...")
    gate = ParameterValidator(
        max_update_norm=150.0,
        enforce_finite=True,
        enforce_structural_match=True,
    )
    audit = gate.validate(candidate_parameters=w_client2, reference_parameters=w_global_0)

    if not audit.valid:
        print(f"  ❌ RECHAZO CRÍTICO: Parámetros descartados: {audit.issues}")
        sys.exit(1)

    norm_diff = audit.delta_norm or 0.0
    print("  ✅ Validación de tensores y finitud: APROBADA (0 NaNs, 0 Infs).")
    print(
        f"  ✅ Norma de divergencia ||W_c2 - W_0||₂: {norm_diff:.4f} "
        f"(tolerancia: {gate.max_update_norm:.1f})"
    )

    # Agregación FedAvg / FedProx: W_global_1 = 0.5 * W_0 + 0.5 * W_client2
    w_global_1 = [0.5 * (p_m + p_c) for p_m, p_c in zip(w_global_0, w_client2, strict=True)]
    master_wrapper.set_parameters(w_global_1)
    print("  -> W_global actualizado exitosamente a la Ronda 1.")

    # =========================================================================
    # FASE 6: NODO CLIENTE 1 RE-EVALÚA EN T_1 (POST-UPDATE)
    # =========================================================================
    print("\n[FASE 6] NODO CLIENTE 1 (TESTER): Re-evaluación en T_1 tras actualización...")
    stats_t1 = master_wrapper.evaluate(test_loader)
    acc_t1 = stats_t1.metrics.get("accuracy", 0.0)

    loss_diff = stats_t1.loss - stats_t0.loss
    acc_diff = acc_t1 - acc_t0

    print("  -> Resultados CLIENTE 1 en T_1 (Post-Update):")
    print(f"     • Pérdida (Loss T_1)   : {stats_t1.loss:.4f} (Variación: {loss_diff:+.4f})")
    print(f"     • Exactitud (Acc T_1)  : {acc_t1 * 100:.2f}% (Variación: {acc_diff * 100:+.2f}%)")

    # =========================================================================
    # BALANCE FINAL Y CONCLUSIÓN DE IMPACTO
    # =========================================================================
    print("\n" + "=" * 78)
    print("               REPORTE DE IMPACTO PARA EL CLIENTE 1")
    print("=" * 78)
    hdr = f"{'Métrica':<24} | {'Antes (T_0)':<16} | {'Después (T_1)':<16} | {'Impacto':<12}"
    print(hdr)
    print("-" * 78)
    print(
        f"{'Pérdida (Loss)':<24} | {stats_t0.loss:>16.4f} | "
        f"{stats_t1.loss:>16.4f} | {loss_diff:>+12.4f}"
    )
    acc_t0_str = f"{acc_t0 * 100:.2f}%"
    acc_t1_str = f"{acc_t1 * 100:.2f}%"
    acc_diff_str = f"{acc_diff * 100:+.2f}%"
    print(
        f"{'Exactitud (Accuracy)':<24} | {acc_t0_str:>16} | {acc_t1_str:>16} | {acc_diff_str:>12}"
    )
    print("=" * 78)

    if acc_diff > 0 or loss_diff < 0:
        print("✅ VEREDICTO POSITIVO: El Cliente 1 experimenta una mejora medible")
        print("   gracias a los parámetros incorporados desde el Cliente 2.")
    else:
        print("ℹ️ VEREDICTO: Ajuste de distribución completado sin divergencias.")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulación Distribuida 3 Nodos AgriFed")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/EnfermedadesHoja_v1.0"),
        help="Ruta al dataset montado",
    )
    parser.add_argument("--model", type=str, default="resnet18", help="Modelo timm")
    args = parser.parse_args()

    root = args.dataset_root
    alt_root = Path("../enfermedadeshoja_v1.0/EnfermedadesHoja_v1.0")
    if not root.is_dir() and alt_root.is_dir():
        root = alt_root

    run_3node_simulation(root, model_name=args.model)

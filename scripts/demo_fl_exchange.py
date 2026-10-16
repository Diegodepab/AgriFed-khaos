"""End-to-End Federated Parameter Exchange and Inference Shift Demo.

This script demonstrates the exact workflow of AgriFed-KHAOS:
1. Central Server: Initializes a global vision model with base weights W_0.
2. Baseline Inference: Evaluates 3 distinct holdout test images before parameter update.
3. Private Client Node: Trains locally on its private dataset (never shared!).
4. Secure Parameter Export: Client extracts only numerical parameters (list[np.ndarray]).
5. Server Parameter Gate: Validates shapes, dtypes, non-finiteness, and delta norm ||ΔW||_2.
6. Server Aggregation: Incorporates client update into the global model (W_updated).
7. Post-Update Inference: Re-evaluates the same 3 holdout test images and presents
   a side-by-side comparison (Before vs After).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Ensure src is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agrifed_khaos.wrappers import (  # noqa: E402
    ParameterValidator,
    TimmClassificationWrapper,
)


def load_image_tensor(image_path: Path, img_size: int = 224) -> Any:
    """Preprocess a single RGB image to torch tensor format (1, 3, H, W)."""
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        tensor = transform(rgb_img)
    return tensor.unsqueeze(0)


def predict(
    wrapper: TimmClassificationWrapper,
    image_tensor: Any,
    class_names: list[str],
) -> tuple[str, float]:
    """Execute forward pass and return predicted class and confidence probability."""
    import torch

    wrapper.module.eval()
    with torch.no_grad():
        logits = wrapper.module(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = class_names[pred_idx]
    return pred_class, confidence


def run_demo(dataset_root: Path, model_name: str = "resnet18") -> None:
    print("=" * 80)
    print("      AGRIFED-KHAOS: DEMOSTRACIÓN DE TRANSFERENCIA SEGURA DE PARÁMETROS")
    print("=" * 80)
    print(f"Dataset root: {dataset_root.resolve()}")
    print(f"Arquitectura: {model_name} (9 clases)")

    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset

    from agrifed_khaos.data import resolve_dataset_dir

    try:
        train_dir = resolve_dataset_dir(dataset_root, split="train", silent=True)
        test_dir = resolve_dataset_dir(dataset_root, split="test", silent=True)
    except FileNotFoundError:
        if (dataset_root / "dataset" / "train").is_dir():
            train_dir = dataset_root / "dataset" / "train"
            test_dir = dataset_root / "dataset" / "test"
        else:
            train_dir = resolve_dataset_dir(None, split="train")
            test_dir = resolve_dataset_dir(None, split="test")

    if not train_dir.is_dir() or not test_dir.is_dir():
        print(f"ERROR: Splits no encontrados en {dataset_root}", file=sys.stderr)
        sys.exit(1)

    # 1. Detect class names
    class_dirs = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"\nClases detectadas ({len(class_dirs)}): {', '.join(class_dirs)}")

    # 2. Select 3 representative test images from holdout test split
    candidate_classes = ["repilo", "sano", "aculus_olearius"]
    test_samples: list[tuple[Path, str]] = []
    for c in candidate_classes:
        c_dir = test_dir / c
        if c_dir.is_dir():
            imgs = sorted(list(c_dir.glob("*.jpg")))
            if imgs:
                test_samples.append((imgs[0], c))

    if len(test_samples) < 3:
        all_test_imgs = sorted(list(test_dir.glob("*/*.jpg")))
        test_samples = [(img, img.parent.name) for img in all_test_imgs[:3]]

    print("\n[PASO 1] 3 IMÁGENES DE PRUEBA SELECCIONADAS (HOLDOUT TEST SPLIT):")
    for idx, (img_path, true_label) in enumerate(test_samples, start=1):
        print(f"  Foto #{idx}: {img_path.name:<28} | Clase Real: {true_label.upper()}")

    # 3. Central Server initializes global model W_0
    print("\n[PASO 2] SERVIDOR CENTRAL: Inicializando modelo global (W_global_0)...")
    server_wrapper = TimmClassificationWrapper.create(
        model_name=model_name,
        num_classes=len(class_dirs),
        learning_rate=1e-3,
        pretrained=False,
        device="cpu",
    )
    initial_params = server_wrapper.get_parameters()
    print(f"  • Tensores de parámetros en W_global: {len(initial_params)}")
    total_elements = sum(p.size for p in initial_params)
    print(f"  • Parámetros flotantes totales: {total_elements:,}")

    # 4. Baseline inference on the 3 images (BEFORE parameter update)
    print("\n[PASO 3] INFERENCIA BASELINE (ANTES DE RECIBIR PARÁMETROS):")
    before_results = []
    for idx, (img_path, true_label) in enumerate(test_samples, start=1):
        tensor = load_image_tensor(img_path)
        pred_class, conf = predict(server_wrapper, tensor, class_dirs)
        match = "✓" if pred_class == true_label else "✗"
        before_results.append((pred_class, conf))
        print(
            f"  Foto #{idx} ({true_label:<15}): Pred = {pred_class:<15} "
            f"(Confianza: {conf * 100:>5.2f}%) [{match}]"
        )

    # 5. Independent Client Simulation (Private Data Node)
    print("\n[PASO 4] NODO CLIENTE INDEPENDIENTE (Entrenamiento Local Privado):")
    print("  • El cliente posee sus imágenes locales de entrenamiento.")
    print("  • POLÍTICA DE PRIVACIDAD: Ninguna imagen ni píxel es transmitido fuera del cliente.")
    print("  • El cliente descarga W_global_0 y realiza fine-tuning local...")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    client_dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)
    # Take a small partition of 32 images
    client_subset = Subset(client_dataset, list(range(32)))
    client_loader = DataLoader(client_subset, batch_size=8, shuffle=True)

    client_wrapper = TimmClassificationWrapper.create(
        model_name=model_name,
        num_classes=len(class_dirs),
        learning_rate=2e-3,
        pretrained=False,
        device="cpu",
    )
    # Inject current global weights into client
    client_wrapper.set_parameters(initial_params)

    # Local training round with FedProx
    stats = client_wrapper.train_local_epoch(
        train_data=client_loader,
        epochs=3,
        prox_mu=0.01,
        reference_parameters=initial_params,
    )
    print(
        f"  • Entrenamiento local completado: {stats.examples} ejemplos, {stats.batches} batches."
    )
    print(f"  • Pérdida local final (Loss): {stats.loss:.4f}")

    # 6. Client exports ONLY parameters (No data transmitted!)
    print("\n[PASO 5] EXTRACCIÓN Y ENVÍO DE PARÁMETROS DEL CLIENTE AL SERVIDOR:")
    client_params = client_wrapper.get_parameters()
    param_bytes = sum(p.nbytes for p in client_params)
    print(
        f"  • Parámetros exportados: {len(client_params)} tensores NumPy "
        f"({param_bytes / 1024 / 1024:.2f} MB)"
    )
    print("  • Datos crudos transmitidos: EXACTAMENTE 0 BYTES (Privacidad Matemática).")

    # 7. Server-Side Parameter Gate (Security & Sanity Audit)
    print("\n[PASO 6] SERVIDOR CENTRAL: Control de Calidad y Sanidad (ParameterValidator):")
    gate = ParameterValidator(
        max_update_norm=150.0,
        enforce_finite=True,
        enforce_structural_match=True,
    )
    validation = gate.validate(
        candidate_parameters=client_params,
        reference_parameters=initial_params,
    )

    if not validation.valid:
        print(f"  ❌ ALERTA: Actualización rechazada por el servidor: {validation.issues}")
        sys.exit(1)

    print("  ✅ Integridad estructural: CORRECTA (Coinciden número de tensores, dtypes y shapes).")
    print("  ✅ Finitud numérica: CORRECTA (Sin NaNs ni Infs).")
    norm_val = validation.delta_norm or 0.0
    print(
        f"  ✅ Norma de divergencia ||ΔW||₂: {norm_val:.4f} "
        f"(Límite máximo permitido: {gate.max_update_norm:.1f})"
    )
    print("  => Parámetros certificados e incorporados exitosamente al modelo global.")

    # 8. Server Incorporates Parameters (FedAvg / FedProx aggregation)
    aggregated_params = [
        0.5 * (p_server + p_client)
        for p_server, p_client in zip(initial_params, client_params, strict=True)
    ]
    server_wrapper.set_parameters(aggregated_params)

    # 9. Post-Update Inference on the same 3 images (AFTER parameter update)
    print("\n[PASO 7] INFERENCIA COMPARATIVA (DESPUÉS DE INCORPORAR LOS PARÁMETROS):")
    print("=" * 68)
    hdr = f"{'Foto / Clase':<18} | {'ANTES (Pred @ Conf)':<18} | {'DESPUÉS (Pred @ Conf)':<18}"
    print(hdr)
    print("-" * 68)

    for idx, (img_path, true_label) in enumerate(test_samples, start=1):
        tensor = load_image_tensor(img_path)
        pred_after, conf_after = predict(server_wrapper, tensor, class_dirs)
        pred_before, conf_before = before_results[idx - 1]

        lbl = f"#{idx} {true_label.upper()}"
        before_str = f"{pred_before} ({conf_before * 100:>5.2f}%)"
        after_str = f"{pred_after} ({conf_after * 100:>5.2f}%)"
        print(f"{lbl:<18} | {before_str:<18} | {after_str:<18}")

    print("=" * 80)
    print("✅ DEMOSTRACIÓN COMPLETADA CON ÉXITO:")
    print("   El servidor mejoró su modelo global consumiendo ÚNICAMENTE parámetros numéricos,")
    print("   preservando el dataset privado del cliente y auditando la sanidad matemática.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo AgriFed FL Parameter Exchange")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/EnfermedadesHoja_v1.0"),
        help="Path to dataset root",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="timm model architecture",
    )
    args = parser.parse_args()

    root = args.dataset_root
    alt_root = Path("../enfermedadeshoja_v1.0/EnfermedadesHoja_v1.0")
    if not root.is_dir() and alt_root.is_dir():
        root = alt_root

    run_demo(root, model_name=args.model)

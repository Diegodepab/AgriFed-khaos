"""Hardware detection, resource profiling, and federated architecture recommendations.

Provides intelligent model selection for computer vision tasks in Federated Learning
environments, optimizing for compute (CPU vs GPU), memory, and communication bandwidth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


@dataclass(slots=True)
class HardwareProfile:
    """Snapshot of available compute and memory resources."""

    device_type: Literal["cuda", "cpu", "mps"]
    device_name: str
    cpu_count: int
    memory_ram_gb: float
    gpu_count: int = 0
    gpu_memory_vram_gb: float = 0.0
    cuda_compute_capability: str | None = None
    has_mixed_precision_fp16: bool = False

    @classmethod
    def detect(cls) -> HardwareProfile:
        """Inspect host environment and return detected hardware specifications."""
        cpu_count = os.cpu_count() or 4
        ram_gb = 8.0

        # Memory estimation from cgroups or psutil/system
        if hasattr(os, "sysconf"):
            try:
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                ram_gb = round((pages * page_size) / (1024**3), 2)
            except (ValueError, OSError):
                ram_gb = 8.0

        if _TORCH_AVAILABLE and torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_idx)
            gpu_mem = round(torch.cuda.get_device_properties(dev_idx).total_memory / (1024**3), 2)
            major, minor = torch.cuda.get_device_capability(dev_idx)
            return cls(
                device_type="cuda",
                device_name=dev_name,
                cpu_count=cpu_count,
                memory_ram_gb=ram_gb,
                gpu_count=torch.cuda.device_count(),
                gpu_memory_vram_gb=gpu_mem,
                cuda_compute_capability=f"{major}.{minor}",
                has_mixed_precision_fp16=True,
            )

        has_mps = (
            _TORCH_AVAILABLE
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if has_mps:
            return cls(
                device_type="mps",
                device_name="Apple Silicon Metal (MPS)",
                cpu_count=cpu_count,
                memory_ram_gb=ram_gb,
                gpu_count=1,
                has_mixed_precision_fp16=True,
            )

        # CPU Fallback
        cpu_model = "Generic x86_64 / ARM CPU"
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

        return cls(
            device_type="cpu",
            device_name=cpu_model,
            cpu_count=cpu_count,
            memory_ram_gb=ram_gb,
            gpu_count=0,
            gpu_memory_vram_gb=0.0,
            has_mixed_precision_fp16=False,
        )


@dataclass(slots=True)
class ModelCandidate:
    """Candidate vision architecture evaluated for federated suitability."""

    architecture: str
    family: Literal["torch", "timm", "ultralytics"]
    task: Literal["classification", "detection", "segmentation"]
    param_count_millions: float
    checkpoint_size_mb: float
    recommended_batch_size: int
    optimal_device: str
    federated_suitability_score: int  # 1-100
    rationale: str
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)


class ModelSelector:
    """Knowledge engine for selecting the optimal vision model for Federated Learning."""

    @staticmethod
    def get_recommendations(
        profile: HardwareProfile,
        task: Literal["classification", "detection", "segmentation"] = "classification",
        network_speed_mbit: float = 50.0,
        target_client_tier: Literal["cpu-low", "cpu-high", "gpu"] = "cpu-low",
        dataset_sample_count: int | None = None,
        num_classes: int | None = None,
        avg_resolution: tuple[int, int] | None = None,
    ) -> list[ModelCandidate]:
        """Rank and return architectures suited for given client tier and dataset constraints.

        In Federated Learning, model selection is constrained by the client nodes (rural farms,
        cooperatives on CPU) rather than server compute. Small datasets (<1,000 images) severely
        penalize Vision Transformers due to lack of inductive bias.
        """
        candidates: list[ModelCandidate] = []
        is_small_dataset = dataset_sample_count is not None and dataset_sample_count < 1000
        client_device = "cpu" if target_client_tier in {"cpu-low", "cpu-high"} else "cuda"

        if task == "classification":
            # --- Candidate 1: ResNet-18 (Gold standard FL benchmark & small-dataset champion) ---
            if target_client_tier == "cpu-low":
                score_resnet = 92
                rec_bs = 16
            elif target_client_tier == "cpu-high":
                score_resnet = 95
                rec_bs = 32
            else:  # gpu
                score_resnet = 96 if is_small_dataset else 93
                rec_bs = 64

            candidates.append(
                ModelCandidate(
                    architecture="resnet18",
                    family="torch",
                    task="classification",
                    param_count_millions=11.2,
                    checkpoint_size_mb=42.7,
                    recommended_batch_size=rec_bs,
                    optimal_device=client_device,
                    federated_suitability_score=score_resnet,
                    rationale=(
                        "Excepcional equilibrio entre capacidad representacional, bajo footprint "
                        "de memoria (42 MB) y rápida convergencia bajo FedProx. Su fuerte sesgo "
                        "inductivo convolucional previene sobreajuste incluso con pocas imágenes."
                    ),
                    pros=[
                        "Fuerte sesgo inductivo convolucional (muy estable ante datos No-IID)",
                        "Bajo ancho de banda por ronda (42 MB de ida y vuelta)",
                        "Entrenable en CPU en minutos sin requerir GPU obligatoria",
                    ],
                    cons=[
                        "Resolución espacial final reducida (7x7) para mapas Grad-CAM muy finos",
                    ],
                )
            )

            # --- Candidate 2: MobileNetV4 (Edge & Low-Bandwidth CPU Champion) ---
            if target_client_tier == "cpu-low":
                score_mob = 98  # Top pick for rural / low-resource edge
                rec_bs_mob = 16
            elif target_client_tier == "cpu-high":
                score_mob = 90
                rec_bs_mob = 32
            else:  # gpu
                score_mob = 86
                rec_bs_mob = 64

            candidates.append(
                ModelCandidate(
                    architecture="mobilenetv4_conv_small.e1200_r224_in1k",
                    family="timm",
                    task="classification",
                    param_count_millions=3.8,
                    checkpoint_size_mb=15.2,
                    recommended_batch_size=rec_bs_mob,
                    optimal_device=client_device,
                    federated_suitability_score=score_mob,
                    rationale=(
                        "Diseñado específicamente para edge y CPUs agrícolas de bajo consumo. "
                        "Huella de red mínima (~15 MB por ronda) ideal para redes rurales 3G/4G "
                        "e inferencia ultra-rápida (<15ms en CPU)."
                    ),
                    pros=[
                        "Huella de red mínima (~15 MB por ronda)",
                        "Inferencia ultra-rápida (<15ms en CPU)",
                        "Muy baja demanda de memoria RAM/VRAM en clientes",
                    ],
                    cons=[
                        "Menor capacidad para diferenciar sutiles manchas foliares complejas",
                    ],
                )
            )

            # --- Candidate 3: ConvNeXt-Tiny (GPU High Performance) ---
            if target_client_tier == "cpu-low":
                score_convnext = 45  # Prohibitive on low CPU
            elif target_client_tier == "cpu-high":
                score_convnext = 65
            else:  # gpu
                score_convnext = 89 if is_small_dataset else 96

            candidates.append(
                ModelCandidate(
                    architecture="convnext_tiny.fb_in22k_ft_in1k",
                    family="timm",
                    task="classification",
                    param_count_millions=28.6,
                    checkpoint_size_mb=109.5,
                    recommended_batch_size=16 if client_device == "cpu" else 64,
                    optimal_device="cuda",
                    federated_suitability_score=score_convnext,
                    rationale=(
                        "Arquitectura CNN de última generación con receptores 7x7. "
                        "Máxima precisión si los clientes tienen GPU CUDA; desaconsejado en "
                        "clientes cpu-low debido a su elevado peso de tensor (~110 MB)."
                    ),
                    pros=[
                        "Precisión de vanguardia (State-of-the-Art en clasificación visual)",
                        "Excelente gradiente y estabilidad de optimización",
                    ],
                    cons=[
                        "Lento en CPU pura; requiere GPU CUDA en clientes para rendimiento fluido",
                        "Mayor tamaño de tensor (~110 MB por ronda)",
                    ],
                )
            )

            # --- Candidate 4: Vision Transformer (ViT-Tiny) ---
            if is_small_dataset:
                # Severe penalty: lack of inductive bias overfits catastrophically on small datasets
                score_vit = 35 if client_device == "cpu" else 40
                vit_rationale = (
                    "[ALERTA DATASET PEQUEÑO] Dataset reducido (<1.000 muestras). ViT carece "
                    "de sesgo inductivo convolucional local (translation equivariance), provocando "
                    "sobreajuste catastrófico y divergencia en clientes federados."
                )
                vit_cons = [
                    "Sobreajuste catastrófico en datasets pequeños sin preentrenamiento masivo",
                    "Sin sesgo inductivo local; divergencia rápida con pocos datos por cliente",
                    "Complejidad de atención cuadrática pesada en CPUs modestas",
                ]
            else:
                if target_client_tier == "cpu-low":
                    score_vit = 38
                elif target_client_tier == "cpu-high":
                    score_vit = 60
                else:  # gpu
                    score_vit = 85

                vit_rationale = (
                    "Atención global para relaciones de largo alcance en la imagen. "
                    "Viable con aceleración GPU y volumen de datos amplio; sensible a datos No-IID."
                )
                vit_cons = [
                    "Sin sesgo inductivo local; divergencia rápida con pocos datos por cliente",
                    "Requiere preentrenamiento extenso en ImageNet",
                ]

            candidates.append(
                ModelCandidate(
                    architecture="vit_tiny_patch16_224",
                    family="timm",
                    task="classification",
                    param_count_millions=5.7,
                    checkpoint_size_mb=22.8,
                    recommended_batch_size=16 if client_device == "cpu" else 64,
                    optimal_device="cuda",
                    federated_suitability_score=score_vit,
                    rationale=vit_rationale,
                    pros=[
                        "Capacidad de modelar contexto global de la hoja/fruto",
                        "Peso compacto en variante tiny (~23 MB)",
                    ],
                    cons=vit_cons,
                )
            )

        elif task == "detection":
            score_yolo_n = 95 if client_device == "cpu" else 90
            score_yolo_s = 75 if client_device == "cpu" else 95

            candidates.append(
                ModelCandidate(
                    architecture="yolov8n.pt",
                    family="ultralytics",
                    task="detection",
                    param_count_millions=3.2,
                    checkpoint_size_mb=6.5,
                    recommended_batch_size=8 if client_device == "cpu" else 32,
                    optimal_device=client_device,
                    federated_suitability_score=score_yolo_n,
                    rationale=(
                        "Detección de objetos ultra-ligera en tiempo real. "
                        "El modelo más adecuado para localizar y contar frutos u hojas en CPU/Edge."
                    ),
                    pros=[
                        "Peso de checkpoint minúsculo (~6.5 MB)",
                        "Predicción de cajas delimitadoras en tiempo real",
                    ],
                    cons=[
                        "Menor recall en frutos diminutos u ocluidos",
                    ],
                )
            )
            candidates.append(
                ModelCandidate(
                    architecture="yolov8s.pt",
                    family="ultralytics",
                    task="detection",
                    param_count_millions=11.2,
                    checkpoint_size_mb=22.5,
                    recommended_batch_size=8 if client_device == "cpu" else 32,
                    optimal_device="cuda",
                    federated_suitability_score=score_yolo_s,
                    rationale=(
                        "Mayor precisión en detección de lesiones complejas o frutos en racimos."
                    ),
                    pros=["Excelente mAP50-95", "Gran estabilidad"],
                    cons=["Exige GPU para entrenamiento federado ágil"],
                )
            )

        elif task == "segmentation":
            candidates.append(
                ModelCandidate(
                    architecture="yolov8n-seg.pt",
                    family="ultralytics",
                    task="segmentation",
                    param_count_millions=3.4,
                    checkpoint_size_mb=7.1,
                    recommended_batch_size=8 if client_device == "cpu" else 16,
                    optimal_device=client_device,
                    federated_suitability_score=95,
                    rationale=(
                        "Segmentación de instancias a nivel de píxel. Genera polígonos exactos "
                        "del contorno sin depender de mapas difusos de Grad-CAM."
                    ),
                    pros=[
                        "Máscaras vectoriales exactas a nivel de píxel",
                        "Menos de 8 MB de checkpoint federado",
                    ],
                    cons=[
                        "Requiere datasets anotados con polígonos de segmentación",
                    ],
                )
            )

        # Ordenar por idoneidad federada decreciente
        candidates.sort(key=lambda c: c.federated_suitability_score, reverse=True)
        return candidates

    @classmethod
    def choose_best(
        cls,
        task: Literal["classification", "detection", "segmentation"] = "classification",
        target_client_tier: Literal["cpu-low", "cpu-high", "gpu"] = "cpu-low",
        dataset_sample_count: int | None = None,
        num_classes: int | None = None,
        avg_resolution: tuple[int, int] | None = None,
        profile: HardwareProfile | None = None,
    ) -> tuple[HardwareProfile, ModelCandidate]:
        """Auto-detect hardware and return the top recommended candidate."""
        if profile is None:
            profile = HardwareProfile.detect()
        candidates = cls.get_recommendations(
            profile=profile,
            task=task,
            target_client_tier=target_client_tier,
            dataset_sample_count=dataset_sample_count,
            num_classes=num_classes,
            avg_resolution=avg_resolution,
        )
        return profile, candidates[0]

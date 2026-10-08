"""Configuration models for AgriFed-KHAOS.

All runtime input enters through these Pydantic models and typed settings.
Business logic must strictly consume these validated models rather than raw
dictionaries or unchecked YAML inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetConfig(BaseModel):
    """Dataset paths, split layout, and client partitioning."""

    root: Path = Field(..., description="Root folder for the macro-dataset")
    annotations_csv: Path | None = Field(
        default=None,
        description="Path to annotations/labels.csv",
    )
    train_dir: str = Field(default="dataset/train")
    val_dir: str = Field(default="dataset/val")
    test_dir: str = Field(default="dataset/test")
    allow_test_as_client: bool = Field(
        default=False,
        description="Explicit opt-in for unsafe simulations consuming test as client data",
    )
    num_clients: int = Field(default=2, ge=1, description="Number of simulated federated clients")
    client_split_seed: int = Field(default=42, description="Seed for deterministic client sharding")

    @model_validator(mode="after")
    def _resolve_paths(self) -> DatasetConfig:
        if self.annotations_csv is None:
            self.annotations_csv = self.root / "annotations" / "labels.csv"
        if not self.annotations_csv.is_absolute():
            self.annotations_csv = (self.root / self.annotations_csv).resolve()
        return self

    @property
    def train_path(self) -> Path:
        return (self.root / self.train_dir).resolve()

    @property
    def val_path(self) -> Path:
        return (self.root / self.val_dir).resolve()

    @property
    def test_path(self) -> Path:
        return (self.root / self.test_dir).resolve()


class ModelConfig(BaseModel):
    """Computer vision model architecture and device configuration."""

    family: Literal["timm", "ultralytics", "torch"] = Field(
        default="timm",
        description="Model family: timm (ViT/CNN classification), ultralytics (YOLO), or torch",
    )
    name: str = Field(
        default="vit_tiny_patch16_224",
        description="Model identifier (e.g. timm architecture or YOLO model config/weights)",
    )
    num_classes: int = Field(default=9, ge=1, description="Number of target classes")
    pretrained: bool = Field(default=False, description="Initialize from pretrained weights")
    device: str = Field(default="cpu", description="Compute device: cpu, cuda, or specific GPU id")
    img_size: int = Field(default=224, ge=32, description="Target square image resolution")


class ParameterGateConfig(BaseModel):
    """Client parameter validation and anomaly gate thresholds."""

    max_update_norm: float = Field(
        default=100.0,
        gt=0.0,
        description="Maximum allowed L2 norm of parameter delta ||W_client - W_global||_2",
    )
    enforce_finite: bool = Field(
        default=True,
        description="Reject updates containing NaN or infinite values",
    )
    enforce_structural_match: bool = Field(
        default=True,
        description="Verify tensor counts, shapes, and dtypes match global model",
    )
    reject_anomalies: bool = Field(
        default=True,
        description="Drop client updates exceeding max_update_norm rather than clamping",
    )


class FederationConfig(BaseModel):
    """Federated learning hyperparameters and FedProx settings."""

    rounds: int = Field(default=10, ge=1, description="Number of federated communication rounds")
    min_clients: int = Field(default=2, ge=1, description="Total expected clients")
    min_fit_clients: int = Field(
        default=2, ge=1, description="Minimum clients sampled for training"
    )
    min_evaluate_clients: int = Field(
        default=2, ge=1, description="Minimum clients sampled for evaluation"
    )
    local_epochs: int = Field(default=1, ge=1, description="Local training epochs per client round")
    batch_size: int = Field(default=16, ge=1, description="Local training batch size")
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Local optimizer learning rate")
    prox_mu: float = Field(
        default=0.01,
        ge=0.0,
        description="FedProx proximal parameter mu constraining local drift",
    )
    seed: int = Field(default=42, description="Global random seed for reproducibility")


class QualityConfig(BaseModel):
    """Pre-FL quality baseline and validation criteria."""

    min_baseline_accuracy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable baseline accuracy on validation set",
    )
    max_baseline_loss: float = Field(
        default=100.0,
        gt=0.0,
        description="Maximum acceptable baseline cross-entropy loss",
    )
    run_pre_fl_baseline: bool = Field(
        default=True,
        description="Evaluate global model on validation set before starting Round 1",
    )
    evaluate_global_every_round: bool = Field(
        default=True,
        description="Evaluate aggregated model on central validation set after every round",
    )


class AppConfig(BaseModel):
    """Top-level configuration bundle."""

    dataset: DatasetConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    parameter_gate: ParameterGateConfig = Field(default_factory=ParameterGateConfig)
    federation: FederationConfig = Field(default_factory=FederationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)


class RuntimeSettings(BaseSettings):
    """Environment-backed settings to locate configuration file."""

    model_config = SettingsConfigDict(env_prefix="AGRIFED_", env_nested_delimiter="__")

    config_path: Path = Field(default=Path("configs/federated.yaml"))

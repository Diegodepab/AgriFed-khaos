"""Explicit Pydantic Schemas and Data Transfer Objects (DTOs) for AgriFed-KHAOS."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AutoFLRequest(BaseModel):
    """Payload schema for Zero-to-Model Auto-FL training request."""

    title: str
    slug: str | None = None
    domain: str = "General"
    crop: str = ""
    target_element: str = ""
    purpose: str = ""
    thumbnail_url: str | None = None
    description: str = ""
    input_guide: str = "Fotografía clara de la muestra con iluminación adecuada."
    dataset_path: str
    task: str = "auto"
    epochs: int = 3
    architecture: str = "auto"
    batch_size: int = 32
    lr: float = 1e-3
    target_client_tier: str = "cpu-low"
    transfer_learning: str = "partial"
    loss_function: str = "ce"
    augmentation: str = "basic"
    train_split: float | None = 0.8
    val_split: float | None = 0.1
    test_split: float | None = 0.1


class DatasetInspectRequest(BaseModel):
    """Payload schema for inspecting dataset directories."""

    dataset_path: str
    source_name: str | None = None


class DatasetDeduplicateRequest(BaseModel):
    """Payload schema for deduplicating and cleaning a dataset directory."""

    dataset_path: str
    remove_duplicates: bool = True
    remove_corrupted: bool = False


class DatasetFetchURLRequest(BaseModel):
    """Payload schema for fetching a dataset archive from a remote URL."""

    url: str


class DatasetConvertCropsRequest(BaseModel):
    """Payload schema for extracting object crops from a COCO dataset into ImageFolder format."""

    dataset_path: str
    target_categories: list[str] | None = None
    min_box_size: int = 24
    padding_pct: float = 0.05
    max_crops_per_class: int | None = 2000
    output_name: str | None = None


class DatasetConvertYOLORequest(BaseModel):
    """Payload schema for converting a COCO dataset to Ultralytics YOLO format."""

    dataset_path: str
    output_name: str | None = None


class PredictResponse(BaseModel):
    """Response schema for single image inference prediction."""

    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    raw_probabilities: dict[str, float] | None = None
    model_name: str
    threshold_applied: str = "nominal"
    ood: dict[str, Any] | None = None
    latency_ms: float = 0.0
    task: str = "classification"
    total_detections: int | None = None
    detection_counts: dict[str, int] | None = None
    detections: list[dict[str, Any]] | None = None


class BatchPredictItem(BaseModel):
    """Prediction result for a single sample within a batch."""

    filename: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    threshold_applied: str = "nominal"
    ood: dict[str, Any] | None = None
    task: str = "classification"
    total_detections: int | None = None
    detection_counts: dict[str, int] | None = None
    detections: list[dict[str, Any]] | None = None


class BatchPredictResponse(BaseModel):
    """Response schema for batch image inference."""

    model_name: str
    total_samples: int
    predictions: list[BatchPredictItem]
    latency_ms: float = 0.0


class JobActionResponse(BaseModel):
    """Response schema for asynchronous job runner trigger."""

    status: str
    job_id: str
    action: str
    queue_position: int
    message: str


class HealthResponse(BaseModel):
    """Service health response schema."""

    status: str = "ok"
    version: str = "1.0.0"
    active_model: str
    architecture: str
    num_classes: int
    classes: list[str]
    device: str


class LightClientHeadUpdateRequest(BaseModel):
    """Payload schema for edge / PWA client head-only federated update."""

    client_id: str
    model_slug: str
    base_model_sha256: str
    training_scope: str = "head_only"
    num_samples: int
    classes: list[str]
    missing_classes: list[int] = []
    head_tensors: dict[str, Any]
    training_stats: dict[str, Any] = {}
    payload_sha256: str | None = None
    hmac_signature: str | None = None


class ModelBundleResponse(BaseModel):
    """Manifest response schema for PWA edge model bundle."""

    schema_version: str = "1.0"
    model_slug: str
    base_model_sha256: str
    architecture: str
    num_classes: int
    classes: list[str]
    embedding_dim: int
    head_tensor_keys: dict[str, str]
    preprocessing: dict[str, Any]
    head_weights: dict[str, Any]
    onnx_download_url: str
    cached: bool = False


class ExpertSaveVersionRequest(BaseModel):
    """Payload schema for an expert saving a frozen model version snapshot."""

    custom_name: str
    description: str = ""
    version_tag: str | None = None
    base_model_sha256: str
    num_samples: int = 0
    accuracy: float = 0.0
    final_loss: float = 0.0
    delta_norm: float = 0.0
    classes: list[str] = []
    head_weights: dict[str, Any]
    is_pinned: bool = False


class ExpertPinVersionRequest(BaseModel):
    """Payload schema for toggling the pinned state of an expert version."""

    is_pinned: bool




"""Dataset inventory and split-aware partitioning helpers."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class DatasetRecord(BaseModel):
    """One row in annotations/labels.csv."""

    filename: str
    split: str
    disease: str
    condition: str
    task: str
    source_dataset: str
    source_url: str


class SplitSummary(BaseModel):
    """Counts for a single split."""

    split: str
    count: int = Field(default=0, ge=0)
    missing_files: int = Field(default=0, ge=0)


class DatasetInventory(BaseModel):
    """Validated dataset inventory."""

    root: Path
    annotations_csv: Path
    total_records: int = Field(default=0, ge=0)
    split_summaries: list[SplitSummary] = Field(default_factory=list)
    class_counts: dict[str, int] = Field(default_factory=dict)
    missing_files: list[str] = Field(default_factory=list)
    unknown_splits: list[str] = Field(default_factory=list)


def load_records(annotations_csv: Path) -> list[DatasetRecord]:
    """Load annotations with strict header validation."""
    with annotations_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "filename",
            "split",
            "disease",
            "condition",
            "task",
            "source_dataset",
            "source_url",
        }
        if reader.fieldnames is None or required_columns.difference(reader.fieldnames):
            missing = sorted(required_columns.difference(reader.fieldnames or []))
            raise ValueError(f"annotations/labels.csv is missing required columns: {missing}")

        return [DatasetRecord.model_validate(row) for row in reader]


def sample_path(root: Path, record: DatasetRecord) -> Path:
    """Resolve the image path for a single record."""
    classification_path = (
        root / "dataset" / record.split / record.disease / record.filename
    ).resolve()
    if classification_path.is_file():
        return classification_path

    detection_path = (root / "detection" / record.split / "images" / record.filename).resolve()
    if detection_path.is_file():
        return detection_path

    return classification_path


def load_inventory(root: Path, annotations_csv: Path | None = None) -> DatasetInventory:
    """Build a split-aware inventory from the on-disk dataset."""
    resolved_annotations = annotations_csv or root / "annotations" / "labels.csv"
    records = load_records(resolved_annotations)

    split_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    missing_files: list[str] = []
    unknown_splits: list[str] = []
    known_splits = {"train", "val", "test"}

    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        class_counts[record.disease] = class_counts.get(record.disease, 0) + 1
        if record.split not in known_splits:
            unknown_splits.append(record.split)
        if not sample_path(root, record).is_file():
            missing_files.append(record.filename)

    summaries = [
        SplitSummary(
            split=split,
            count=count,
            missing_files=sum(
                1
                for record in records
                if record.split == split and not sample_path(root, record).is_file()
            ),
        )
        for split, count in sorted(split_counts.items())
    ]

    return DatasetInventory(
        root=root.resolve(),
        annotations_csv=resolved_annotations.resolve(),
        total_records=len(records),
        split_summaries=summaries,
        class_counts=dict(sorted(class_counts.items())),
        missing_files=sorted(set(missing_files)),
        unknown_splits=sorted(set(unknown_splits)),
    )


def partition_train_records(
    records: list[DatasetRecord],
    client_index: int,
    client_count: int,
) -> list[DatasetRecord]:
    """Deterministically shard train records across private clients without data leakage."""
    if client_count < 1:
        raise ValueError("client_count must be positive")
    if not 0 <= client_index < client_count:
        raise ValueError("client_index must be within client_count")

    selected: list[DatasetRecord] = []
    for record in records:
        key = hashlib.sha256(record.filename.encode("utf-8")).hexdigest()
        shard = int(key[:8], 16) % client_count
        if shard == client_index:
            selected.append(record)
    return selected


def create_torch_dataloader(
    root_dir: Path,
    batch_size: int = 16,
    img_size: int = 224,
    shuffle: bool = True,
) -> Any:
    """Create a PyTorch DataLoader using standard torchvision ImageFolder if available."""
    try:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torch and torchvision are required for data loaders") from exc

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(root=str(root_dir), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

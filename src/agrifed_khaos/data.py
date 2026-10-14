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
    alpha: float | None = None,
    seed: int = 42,
) -> list[DatasetRecord]:
    """Deterministically partition train records across private clients without data leakage.

    Supports:
    1. Uniform deterministic hash partitioning (alpha=None, default):
       Uniform distribution across clients via SHA-256(filename).
    2. Non-IID Dirichlet distribution (alpha > 0.0):
       Partitions label distributions across clients according to Dir(alpha),
       accurately modeling severe real-world agricultural label skew.
    """
    if client_count < 1:
        raise ValueError("client_count must be positive")
    if not 0 <= client_index < client_count:
        raise ValueError("client_index must be within client_count")

    if alpha is None:
        selected: list[DatasetRecord] = []
        for record in records:
            key = hashlib.sha256(record.filename.encode("utf-8")).hexdigest()
            shard = int(key[:8], 16) % client_count
            if shard == client_index:
                selected.append(record)
        return selected

    if alpha <= 0.0:
        raise ValueError("Dirichlet concentration alpha must be positive (e.g. 0.1 or 0.5)")

    import numpy as np

    # Group records deterministically by class/disease
    class_groups: dict[str, list[DatasetRecord]] = {}
    for record in records:
        class_groups.setdefault(record.disease, []).append(record)

    client_records: list[DatasetRecord] = []
    rng = np.random.default_rng(seed)

    for _disease_name, group in sorted(class_groups.items()):
        sorted_group = sorted(group, key=lambda r: r.filename)
        n_samples = len(sorted_group)

        # Draw proportions from Dirichlet(alpha * 1_K)
        proportions = rng.dirichlet(np.repeat(alpha, client_count))

        # Convert proportions to counts ensuring exact sum
        counts = np.floor(proportions * n_samples).astype(int)
        remainder = n_samples - int(np.sum(counts))
        fractional = (proportions * n_samples) - counts
        for top_idx in np.argsort(-fractional)[:remainder]:
            counts[top_idx] += 1

        cum_counts = np.cumsum(counts)
        start_idx = 0 if client_index == 0 else int(cum_counts[client_index - 1])
        end_idx = int(cum_counts[client_index])
        client_records.extend(sorted_group[start_idx:end_idx])

    return client_records


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


def get_dataset_targets(dataset: Any) -> list[int]:
    """Extract integer target labels from an ImageFolder, Subset, or generic dataset."""
    if hasattr(dataset, "targets") and isinstance(dataset.targets, (list, tuple)):
        return [int(t) for t in dataset.targets]
    if hasattr(dataset, "targets") and hasattr(dataset.targets, "tolist"):
        return [int(t) for t in dataset.targets.tolist()]
    if hasattr(dataset, "samples") and isinstance(dataset.samples, (list, tuple)):
        return [int(s[1]) for s in dataset.samples]
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent_targets = get_dataset_targets(dataset.dataset)
        return [parent_targets[idx] for idx in dataset.indices]
    raise ValueError(
        f"Unable to extract class targets from dataset of type {type(dataset).__name__}: "
        "missing 'targets' or 'samples' attribute."
    )


def create_stratified_subset(
    dataset: Any,
    max_samples: int,
    seed: int = 42,
) -> Any:
    """Create a stratified subset of dataset preserving class proportions.

    Uses the Largest Remainder (Hamilton-Hare) apportionment method to assign
    quotas per class, guaranteeing that all classes present receive at least 1 sample
    (when max_samples >= number of distinct classes) and that empirical class ratios
    are preserved without distortion.

    Args:
        dataset: PyTorch Dataset (e.g. ImageFolder or Subset) containing targets.
        max_samples: Maximum number of samples to select. If max_samples >= len(dataset),
            the original dataset is returned.
        seed: Random seed for deterministic intra-class index shuffling.

    Returns:
        torch.utils.data.Subset wrapping dataset with the stratified indices.
    """
    import numpy as np
    from torch.utils.data import Subset

    n_total = len(dataset)
    if max_samples >= n_total:
        return dataset

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    targets = get_dataset_targets(dataset)
    class_indices: dict[int, list[int]] = {}
    for idx, target in enumerate(targets):
        class_indices.setdefault(target, []).append(idx)

    num_classes = len(class_indices)
    rng = np.random.default_rng(seed)

    sorted_classes = sorted(class_indices.keys())
    # Exact Hare quota on total max_samples
    exact_shares = {c: (len(class_indices[c]) / n_total) * max_samples for c in sorted_classes}
    allocated = {c: int(np.floor(exact_shares[c])) for c in sorted_classes}

    # If max_samples >= num_classes, guarantee min 1 sample per present class
    if max_samples >= num_classes:
        for c in sorted_classes:
            if allocated[c] == 0:
                allocated[c] = 1

    current_total = sum(allocated.values())
    if current_total < max_samples:
        remainder = max_samples - current_total
        fractionals = [(exact_shares[c] - np.floor(exact_shares[c]), c) for c in sorted_classes]
        fractionals.sort(key=lambda item: (-item[0], item[1]))
        for i in range(remainder):
            c = fractionals[i][1]
            allocated[c] += 1
    elif current_total > max_samples:
        excess = current_total - max_samples
        fractionals = [(exact_shares[c] - np.floor(exact_shares[c]), c) for c in sorted_classes]
        fractionals.sort(key=lambda item: (item[0], item[1]))
        for _, c in fractionals:
            if excess <= 0:
                break
            if allocated[c] > 1:
                allocated[c] -= 1
                excess -= 1

    selected_indices: list[int] = []
    for c in sorted_classes:
        quota = min(allocated[c], len(class_indices[c]))
        class_idx_list = class_indices[c].copy()
        rng.shuffle(class_idx_list)
        selected_indices.extend(class_idx_list[:quota])

    selected_indices.sort()
    return Subset(dataset, selected_indices)


def compute_class_weights(
    dataset_or_targets: Any,
    num_classes: int | None = None,
    smoothing: float = 0.0,
) -> Any:
    """Compute balanced class weights using inverse class frequency.

    Formula:
        w_c = N / (C_present * N_c)
    Normalized such that sum(w_c) == C (mean weight = 1.0).
    Classes with zero samples are assigned weight 1.0.

    Args:
        dataset_or_targets: Either a list/tuple/Tensor of integer class labels,
            or a PyTorch dataset from which targets can be extracted.
        num_classes: Total number of classes. If None, inferred as max(targets) + 1.
        smoothing: Additive label smoothing factor (default: 0.0).

    Returns:
        torch.FloatTensor of shape (num_classes,) containing balanced weights.
    """
    import numpy as np
    import torch

    if isinstance(dataset_or_targets, (list, tuple)):
        targets = [int(t) for t in dataset_or_targets]
    elif hasattr(dataset_or_targets, "tolist") and not hasattr(dataset_or_targets, "dataset"):
        targets = [int(t) for t in dataset_or_targets.tolist()]
    else:
        targets = get_dataset_targets(dataset_or_targets)

    if not targets:
        if num_classes is not None and num_classes > 0:
            return torch.ones(num_classes, dtype=torch.float32)
        raise ValueError("Cannot compute class weights on empty dataset.")

    inferred_num_classes = max(targets) + 1 if targets else 0
    c_count = max(num_classes or inferred_num_classes, inferred_num_classes, 1)

    counts = np.zeros(c_count, dtype=np.float64)
    for t in targets:
        if 0 <= t < c_count:
            counts[t] += 1.0

    n_samples = float(len(targets))
    weights = np.ones(c_count, dtype=np.float32)

    active_mask = counts > 0
    n_active = float(np.sum(active_mask))

    if n_active > 0:
        if smoothing > 0.0:
            smoothed_counts = counts + smoothing
            smoothed_n = n_samples + smoothing * c_count
            weights = (smoothed_n / (c_count * smoothed_counts)).astype(np.float32)
        else:
            for c in range(c_count):
                if counts[c] > 0:
                    weights[c] = float(n_samples / (n_active * counts[c]))
                else:
                    weights[c] = 1.0

        active_weights = weights[active_mask]
        if len(active_weights) > 0 and float(np.mean(active_weights)) > 0:
            weights[active_mask] = weights[active_mask] / float(np.mean(active_weights))

    return torch.tensor(weights, dtype=torch.float32)


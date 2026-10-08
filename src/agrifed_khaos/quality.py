"""Pre-FL baseline quality assessment and split integrity verification.

In the AgriFed-khaos architecture, data curation (filtering blur, low-res, corrupt
images) belongs to the upstream companion repo (agrivision-khaos).
AgriFed-khaos focuses on:
1. Zero-Round / Pre-FL Baseline Quality: Auditing the initial global model on validation
   data before federated communication rounds commence.
2. Dataset Partition & Zero-Leakage Integrity: Ensuring train, val, and test splits
   are disjoint and well-formed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import QualityConfig
from .wrappers import BaseFLWrapper, EvaluationStats


class BaselineQualityReport(BaseModel):
    """Quality and performance metrics of the model prior to federated training."""

    round_index: int = 0
    examples: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    pass_gate: bool = True
    issues: list[str] = Field(default_factory=list)


class SplitIntegrityReport(BaseModel):
    """Verification of partition isolation and zero data leakage."""

    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    overlap_detected: bool = False
    valid: bool = True
    issues: list[str] = Field(default_factory=list)


def evaluate_pre_fl_baseline(
    wrapper: BaseFLWrapper,
    val_loader: Any,
    config: QualityConfig,
) -> BaselineQualityReport:
    """Evaluate Round 0 performance on validation data to establish reference quality."""
    stats: EvaluationStats = wrapper.evaluate(val_loader)
    accuracy = stats.metrics.get("accuracy", 0.0)

    issues: list[str] = []
    if accuracy < config.min_baseline_accuracy:
        issues.append(
            f"Baseline accuracy ({accuracy:.4f}) is below minimum threshold "
            f"({config.min_baseline_accuracy:.4f})"
        )
    if stats.loss > config.max_baseline_loss:
        issues.append(
            f"Baseline loss ({stats.loss:.4f}) exceeds maximum threshold "
            f"({config.max_baseline_loss:.4f})"
        )

    return BaselineQualityReport(
        round_index=0,
        examples=stats.examples,
        loss=stats.loss,
        accuracy=accuracy,
        pass_gate=len(issues) == 0,
        issues=issues,
    )


def audit_split_integrity(
    train_paths: list[Path],
    val_paths: list[Path],
    test_paths: list[Path],
) -> SplitIntegrityReport:
    """Ensure zero-leakage partition isolation across train, val, and test splits."""
    issues: list[str] = []

    train_names = {p.name for p in train_paths}
    val_names = {p.name for p in val_paths}
    test_names = {p.name for p in test_paths}

    overlap_train_val = train_names.intersection(val_names)
    overlap_train_test = train_names.intersection(test_names)
    overlap_val_test = val_names.intersection(test_names)

    has_overlap = bool(overlap_train_val or overlap_train_test or overlap_val_test)
    if overlap_train_val:
        issues.append(
            f"Data leakage detected between train and val: {len(overlap_train_val)} files"
        )
    if overlap_train_test:
        issues.append(
            f"Data leakage detected between train and test: {len(overlap_train_test)} files"
        )
    if overlap_val_test:
        issues.append(f"Data leakage detected between val and test: {len(overlap_val_test)} files")

    return SplitIntegrityReport(
        train_count=len(train_paths),
        val_count=len(val_paths),
        test_count=len(test_paths),
        overlap_detected=has_overlap,
        valid=not has_overlap,
        issues=issues,
    )

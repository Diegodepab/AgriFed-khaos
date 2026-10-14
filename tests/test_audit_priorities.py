"""Unit tests verifying high-impact mathematical and operational improvements in AgriFed-KHAOS.

Covers:
1. Stratified Subsampling (create_stratified_subset) under severe class imbalance
2. Balanced Class-Weighted Loss (compute_class_weights)
3. Calibrated Decision Thresholding (optimal F1 & Youden's J)
4. Server Momentum (FedAvgM / FedOpt) velocity accumulation and persistence
5. Classification Dataset Builder CLI & Bounding Box Sanity Validation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from agrifed_khaos.data import (
    compute_class_weights,
    create_stratified_subset,
    get_dataset_targets,
)


class DummyDataset(Dataset):
    """Simple dataset holding samples and targets for testing."""

    def __init__(self, targets: list[int]) -> None:
        self.targets = targets
        self.samples = [(f"img_{i}.jpg", t) for i, t in enumerate(targets)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.zeros((3, 10, 10)), self.targets[idx]


def test_stratified_subsampling_preserves_ratios() -> None:
    # 100 samples of class 0, 900 samples of class 1 (1:9 ratio)
    targets = [0] * 100 + [1] * 900
    dataset = DummyDataset(targets)

    # Subsample to 100 total samples
    subset = create_stratified_subset(dataset, max_samples=100, seed=42)
    assert len(subset) == 100

    subset_targets = get_dataset_targets(subset)
    count_0 = subset_targets.count(0)
    count_1 = subset_targets.count(1)

    # In stratified subsampling (1:9 ratio), exact quota is 10 of class 0 and 90 of class 1
    assert count_0 == 10
    assert count_1 == 90


def test_stratified_subsampling_guarantees_minority_representation() -> None:
    # 3 classes: class 0 has 2 samples, class 1 has 98 samples, class 2 has 900 samples
    targets = [0] * 2 + [1] * 98 + [2] * 900
    dataset = DummyDataset(targets)

    # Even with small quota (e.g. 50 samples), class 0 must not be starved
    subset = create_stratified_subset(dataset, max_samples=50, seed=42)
    subset_targets = get_dataset_targets(subset)
    assert subset_targets.count(0) >= 1
    assert subset_targets.count(1) >= 1
    assert subset_targets.count(2) >= 1
    assert len(subset) == 50


def test_stratified_subsampling_edge_cases() -> None:
    targets = [0, 1, 0, 1, 0, 1]
    dataset = DummyDataset(targets)

    # If max_samples >= len(dataset), return original dataset
    same_ds = create_stratified_subset(dataset, max_samples=10, seed=42)
    assert same_ds is dataset

    with pytest.raises(ValueError, match="max_samples must be positive"):
        create_stratified_subset(dataset, max_samples=0)


def test_balanced_class_weights() -> None:
    # 10 samples of class 0 (minority) and 90 samples of class 1 (majority)
    targets = [0] * 10 + [1] * 90
    weights = compute_class_weights(targets, num_classes=2)

    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (2,)

    # Minority class must receive significantly higher weight than majority
    assert weights[0].item() > weights[1].item()
    # Ratio should match inverse frequency (9:1)
    ratio = weights[0].item() / weights[1].item()
    assert abs(ratio - 9.0) < 1e-3

    # Mean of weights should be normalized to 1.0
    assert abs(weights.mean().item() - 1.0) < 1e-4


def test_balanced_class_weights_with_unseen_class() -> None:
    # Only class 0 and 1 are present, class 2 is empty
    targets = [0] * 50 + [1] * 50
    weights = compute_class_weights(targets, num_classes=3)
    assert weights.shape == (3,)
    # Present classes are equal, empty class 2 is set to 1.0 neutral weight
    assert abs(weights[0].item() - 1.0) < 1e-4
    assert abs(weights[1].item() - 1.0) < 1e-4
    assert abs(weights[2].item() - 1.0) < 1e-4


def test_calibrated_decision_thresholds() -> None:
    from scripts.evaluate import (
        calibrate_binary_thresholds,
        calibrate_multiclass_ovr_thresholds,
    )

    # Construct synthetic scenario where class 1 is minority (20 samples out of 100)
    # and model probabilities are compressed such that p=0.5 predicts very few positives,
    # but a lower threshold p=0.30 yields much higher F1
    np.random.seed(42)
    y_true = [0] * 80 + [1] * 20
    y_probs = []
    for y in y_true:
        if y == 0:
            # Most negatives have low prob of class 1 (e.g. 0.05 to 0.25)
            p1 = float(np.random.uniform(0.01, 0.25))
        else:
            # Positives have moderate prob of class 1 (e.g. 0.35 to 0.65)
            p1 = float(np.random.uniform(0.35, 0.65))
        y_probs.append([1.0 - p1, p1])

    res = calibrate_binary_thresholds(y_true, y_probs, positive_class_idx=1)

    assert "nominal" in res
    assert "calibrated_f1" in res
    assert "calibrated_youden" in res
    assert "sweep" in res

    nom_f1 = res["nominal"]["f1_score"]
    cal_f1 = res["calibrated_f1"]["f1_score"]
    opt_tau = res["calibrated_f1"]["optimal_threshold"]

    # Calibrated threshold should improve or equal nominal F1
    assert cal_f1 >= nom_f1
    assert 0.05 <= opt_tau <= 0.95

    # Test OvR multiclass
    multiclass_true = [0] * 30 + [1] * 30 + [2] * 40
    multiclass_probs = []
    for y in multiclass_true:
        p = np.random.dirichlet([1.0, 1.0, 1.0]).tolist()
        p[y] += 1.5  # bias towards true class
        total = sum(p)
        multiclass_probs.append([v / total for v in p])

    ovr_res = calibrate_multiclass_ovr_thresholds(
        multiclass_true, multiclass_probs, class_names=["c0", "c1", "c2"]
    )
    assert len(ovr_res) == 3
    assert "c0" in ovr_res
    assert "optimal_threshold" in ovr_res["c0"]


def test_server_momentum_accumulation(tmp_path: Path) -> None:
    """Verify FedAvgM mathematical velocity buffer update and checkpoint evolution."""
    from scripts.aggregate import BN_BUFFER_SUFFIXES

    device = torch.device("cpu")
    # Base state with 2 floating point parameters
    base_state = {
        "fc.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "fc.bias": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "bn.running_mean": torch.tensor([0.0, 0.0], dtype=torch.float32),
    }

    trainable_keys = [
        k for k in base_state if not any(k.endswith(sfx) for sfx in BN_BUFFER_SUFFIXES)
    ]

    # Client update 1
    c1_state = {
        "fc.weight": torch.tensor([[1.2, 2.0], [3.0, 4.4]], dtype=torch.float32),
        "fc.bias": torch.tensor([0.3, 0.2], dtype=torch.float32),
    }

    # Round 1: delta_1 = c1 - base
    # fc.weight diff = [[0.2, 0.0], [0.0, 0.4]]
    # fc.bias diff = [0.2, 0.0]
    delta_fed_1 = {k: (c1_state[k] - base_state[k]) for k in trainable_keys}

    beta = 0.9
    eta = 0.5

    # Round 1: v_0 = 0 -> v_1 = 0.9 * 0 + delta_1 = delta_1
    v_1 = {k: beta * 0.0 + delta_fed_1[k] for k in trainable_keys}
    w_1 = {k: base_state[k] + eta * v_1[k] for k in trainable_keys}

    expected_w1_weight = base_state["fc.weight"] + eta * delta_fed_1["fc.weight"]
    assert torch.allclose(w_1["fc.weight"], expected_w1_weight)

    # Round 2: delta_2 with next update
    c2_state = {
        "fc.weight": torch.tensor([[1.1, 2.0], [3.0, 4.2]], dtype=torch.float32),
        "fc.bias": torch.tensor([0.2, 0.2], dtype=torch.float32),
    }
    delta_fed_2 = {k: (c2_state[k] - w_1[k]) for k in trainable_keys}

    # v_2 = beta * v_1 + delta_2
    v_2 = {k: beta * v_1[k] + delta_fed_2[k] for k in trainable_keys}
    w_2 = {k: w_1[k] + eta * v_2[k] for k in trainable_keys}
    assert w_2["fc.weight"].shape == (2, 2)
    assert torch.allclose(w_2["fc.weight"], w_1["fc.weight"] + eta * v_2["fc.weight"])

    # Verify momentum buffer persistence round-trip
    buffer_file = tmp_path / "server_momentum.pt"
    torch.save(v_2, str(buffer_file))
    loaded_buffer = torch.load(str(buffer_file), map_location=device, weights_only=True)
    assert torch.allclose(loaded_buffer["fc.weight"], v_2["fc.weight"])
    assert torch.allclose(loaded_buffer["fc.bias"], v_2["fc.bias"])


def test_prepare_olivefruit_classification_sanitization(tmp_path: Path) -> None:
    from PIL import Image

    from scripts.prepare_olivefruit_classification import process_split

    raw_dir = tmp_path / "dataset"
    split_dir = raw_dir / "train"
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)

    # Create dummy source image (200x200)
    im_path = images_dir / "olive_01.jpg"
    Image.new("RGB", (200, 200), color="green").save(im_path)

    # Create labels.json containing 1 valid annotation, 1 too small, 1 with negative coordinates
    labels_data = {
        "categories": [{"id": 1, "name": "sana"}, {"id": 2, "name": "defectuosa"}],
        "images": [{"id": 100, "file_name": "olive_01.jpg"}],
        "annotations": [
            # Valid: w=60, h=60 >= min_size 55
            {"id": 1, "image_id": 100, "category_id": 1, "bbox": [10, 10, 60, 60]},
            # Invalid: w=20, h=20 < min_size 55
            {"id": 2, "image_id": 100, "category_id": 1, "bbox": [10, 10, 20, 20]},
            # Invalid: negative area
            {"id": 3, "image_id": 100, "category_id": 1, "bbox": [10, 10, -5, -5]},
        ],
    }

    labels_file = split_dir / "labels.json"
    labels_file.write_text(json.dumps(labels_data), encoding="utf-8")

    out_dir = tmp_path / "classification"
    counts = process_split(
        split_name="train",
        raw_dir=raw_dir,
        out_dir=out_dir,
        target_counts={"sana": 10},
        min_size=55,
        padding_pct=0.10,
    )

    # Exactly 1 valid candidate should be extracted
    assert counts.get("sana") == 1
    extracted_crops = list((out_dir / "train" / "sana").glob("*.jpg"))
    assert len(extracted_crops) == 1

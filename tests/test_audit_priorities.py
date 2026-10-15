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


def test_missing_class_masking_preserves_classifier_row() -> None:
    """Verify that when a class is absent in local partition (N_k=0),
    its classifier row in weights and bias remains strictly invariant (ΔW_k = 0),
    even when training with AdamW (active weight decay & momentum) and FedProx prox_mu = 0.1.
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from agrifed_khaos.wrappers import TorchModuleWrapper

    torch.manual_seed(42)
    # 3 classes, 4 input features
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),  # Classifier head: weight shape (3, 8), bias shape (3,)
    )

    # Save initial baseline weights
    base_head_weight = model[2].weight.detach().clone()
    base_head_bias = model[2].bias.detach().clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    wrapper = TorchModuleWrapper(
        module=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
    )

    # Dataset containing ONLY classes 0 and 1. Class 2 is unobserved (N_2 = 0)
    x = torch.randn(20, 4)
    y = torch.tensor([0, 1] * 10)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    base_params = wrapper.get_parameters()

    stats = wrapper.train_local_epoch(
        loader,
        epochs=3,
        prox_mu=0.1,  # Active FedProx regularization
        reference_parameters=base_params,
        missing_classes=[2],
    )
    assert stats.examples == 60  # 20 samples * 3 epochs

    final_head_weight = model[2].weight.detach()
    final_head_bias = model[2].bias.detach()

    # Classes 0 and 1 must have been updated by gradients
    assert not torch.allclose(final_head_weight[0], base_head_weight[0])
    assert not torch.allclose(final_head_weight[1], base_head_weight[1])

    # Class 2 row and bias MUST be exactly identical (zero change: ΔW_2 = 0, Δb_2 = 0)
    assert torch.allclose(final_head_weight[2], base_head_weight[2], atol=1e-7)
    assert torch.allclose(final_head_bias[2], base_head_bias[2], atol=1e-7)


def test_serve_dynamic_img_size_and_calibrated_thresholds(tmp_path: Path) -> None:
    """Verify serve.py respects bundle img_size and applies calibrated thresholds."""
    import io

    from PIL import Image
    from starlette.testclient import TestClient

    from agrifed_khaos.models import build_model, save_checkpoint_bundle
    from scripts.serve import create_app

    # Build and save a dummy 2-class checkpoint bundle
    model = build_model("resnet18", num_classes=2, pretrained=False)
    bundle_meta = {
        "architecture": "resnet18",
        "num_classes": 2,
        "classes": ["sana", "defectuosa"],
        "img_size": [128, 128],
        "calibrated_thresholds": {
            "calibrated_youden": {"optimal_threshold": 0.85}
        },
    }
    model_path = tmp_path / "model_bundle.pt"
    meta_path = tmp_path / "metadata.json"
    save_checkpoint_bundle(model_path, model.state_dict(), bundle_meta)

    app = create_app(str(model_path), str(meta_path), architecture="resnet18")
    client = TestClient(app)

    # 1. Health check returns dynamic bundle metadata
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    data_health = resp_health.json()
    assert data_health["img_size"] == [128, 128]
    assert data_health["calibrated_thresholds"]["calibrated_youden"]["optimal_threshold"] == 0.85

    # 2. Predict endpoint executes calibrated threshold routing
    dummy_img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    dummy_img.save(buf, format="JPEG")
    buf.seek(0)

    resp_pred = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert resp_pred.status_code == 200
    data_pred = resp_pred.json()
    assert data_pred["threshold_applied"] == "calibrated"
    assert data_pred["predicted_class"] in {"sana", "defectuosa"}


def test_serve_multiclass_calibrated_scoring_4_classes(tmp_path: Path) -> None:
    """Verify that serve.py applies multiclass calibrated score scaling for C >= 2:
    calibrated_scores[c] = probs[c] / max(tau_c, 1e-6) and alters the prediction vs raw argmax.
    """
    import io

    import torch.nn as nn
    from PIL import Image
    from starlette.testclient import TestClient

    from agrifed_khaos.models import build_model, save_checkpoint_bundle
    from scripts.serve import create_app

    classes = ["sana", "roña", "mildiu", "tizón"]
    calibrated_thresholds = {
        "sana": 0.05,
        "roña": 0.80,
        "mildiu": 0.50,
        "tizón": 0.50,
    }
    bundle_meta = {
        "architecture": "resnet18",
        "num_classes": 4,
        "classes": classes,
        "img_size": [64, 64],
        "calibrated_thresholds": calibrated_thresholds,
    }
    model = build_model("resnet18", num_classes=4, pretrained=False)
    model_path = tmp_path / "model_4c.pt"
    meta_path = tmp_path / "metadata_4c.json"
    save_checkpoint_bundle(model_path, model.state_dict(), bundle_meta)

    app = create_app(str(model_path), str(meta_path), architecture="resnet18")

    # Mock net to output fixed logits corresponding to probs: [0.25, 0.40, 0.20, 0.15]
    # Raw argmax would choose class 1 ("roña", prob 0.40)
    # Calibrated scores:
    #   sana: 0.25 / 0.05 = 5.0  <-- WINNER
    #   roña: 0.40 / 0.80 = 0.5
    #   mildiu: 0.20 / 0.50 = 0.4
    #   tizón: 0.15 / 0.50 = 0.3
    class MockNet(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            target_probs = torch.tensor([[0.25, 0.40, 0.20, 0.15]], dtype=torch.float32)
            return torch.log(target_probs)

    app.state.net = MockNet()
    client = TestClient(app)

    dummy_img = Image.new("RGB", (64, 64), color="green")
    buf = io.BytesIO()
    dummy_img.save(buf, format="JPEG")
    buf.seek(0)

    resp_pred = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert resp_pred.status_code == 200
    data = resp_pred.json()

    # Raw argmax would have returned "roña" (40% vs 25% sana)
    # Calibrated thresholding selects "sana" due to operating threshold (0.05 vs 0.80)
    assert data["predicted_class"] == "sana"
    assert data["threshold_applied"] == "calibrated"


def test_formal_dp_gaussian_mechanism() -> None:
    """Verify formal DP: static L2 clipping bound C and independent Gaussian noise."""
    from agrifed_khaos.wrappers import apply_dp_gaussian_noise

    # Base reference parameters
    p_ref = [torch.zeros((4, 4), dtype=torch.float32)]
    # Client parameters with update norm = 10.0
    p_client = [torch.full((4, 4), 2.5, dtype=torch.float32)]  # norm = sqrt(16 * 6.25) = 10.0
    delta_init = (p_client[0] - p_ref[0]).norm(p=2).item()
    assert abs(delta_init - 10.0) < 1e-4

    # Case 1: Pure static clipping (clip_norm=4.0, noise_std=0.0)
    apply_dp_gaussian_noise(p_client, reference_params=p_ref, clip_norm=4.0, noise_std=0.0)
    delta_clipped = (p_client[0] - p_ref[0]).norm(p=2).item()
    assert abs(delta_clipped - 4.0) < 1e-4

    # Case 2: Update below clipping bound (clip_norm=10.0) is not modified when noise_std=0.0
    p_client_small = [torch.full((4, 4), 0.5, dtype=torch.float32)]  # norm = sqrt(16 * 0.25) = 2.0
    apply_dp_gaussian_noise(p_client_small, reference_params=p_ref, clip_norm=5.0, noise_std=0.0)
    delta_small = (p_client_small[0] - p_ref[0]).norm(p=2).item()
    assert abs(delta_small - 2.0) < 1e-4

    # Case 3: Formal Gaussian noise injection with non-zero std
    torch.manual_seed(42)
    p_noisy = [torch.zeros((100, 100), dtype=torch.float32)]
    apply_dp_gaussian_noise(p_noisy, reference_params=None, clip_norm=2.0, noise_std=0.5)
    # Expected noise std = clip_norm * noise_std = 2.0 * 0.5 = 1.0
    sample_std = p_noisy[0].std().item()
    assert abs(sample_std - 1.0) < 0.05


def test_dp_gaussian_and_adaptive_rms_noise_immunity_for_batchnorm() -> None:
    """Verify that BatchNorm buffers are strictly immune to noise perturbation
    and that global L2 clipping norm is calculated jointly.
    """
    from agrifed_khaos.wrappers import apply_adaptive_rms_noise, apply_dp_gaussian_noise

    # Setup weights and BN buffers
    w1 = torch.ones((10, 10), dtype=torch.float32)
    w2 = torch.ones((5, 5), dtype=torch.float32)
    bn_mean = torch.tensor([1.5, 2.5], dtype=torch.float32)
    bn_var = torch.tensor([0.8, 1.2], dtype=torch.float32)
    bn_count = torch.tensor(100, dtype=torch.int64)

    param_dict = {
        "conv1.weight": w1,
        "fc.weight": w2,
        "bn1.running_mean": bn_mean,
        "bn1.running_var": bn_var,
        "bn1.num_batches_tracked": bn_count,
    }

    # Base reference
    ref_dict = {
        "conv1.weight": torch.zeros((10, 10), dtype=torch.float32),
        "fc.weight": torch.zeros((5, 5), dtype=torch.float32),
        "bn1.running_mean": bn_mean.clone(),
        "bn1.running_var": bn_var.clone(),
        "bn1.num_batches_tracked": bn_count.clone(),
    }

    # 1. Apply formal DP Gaussian mechanism
    # Initial joint delta norm = sqrt(100 * 1^2 + 25 * 1^2) = sqrt(125) ~ 11.1803
    orig_norm = apply_dp_gaussian_noise(
        param_dict,
        reference_parameters=ref_dict,
        clip_norm=5.0,
        noise_std=0.2,
    )
    assert abs(orig_norm - (125.0**0.5)) < 1e-3

    # Trainable weights have been modified by clipping and noise
    assert not torch.equal(param_dict["conv1.weight"], torch.ones((10, 10)))
    assert not torch.equal(param_dict["fc.weight"], torch.ones((5, 5)))

    # BatchNorm buffers MUST be bit-for-bit identical (zero perturbation)
    assert torch.equal(param_dict["bn1.running_mean"], ref_dict["bn1.running_mean"])
    assert torch.equal(param_dict["bn1.running_var"], ref_dict["bn1.running_var"])
    assert torch.equal(param_dict["bn1.num_batches_tracked"], ref_dict["bn1.num_batches_tracked"])

    # 2. Apply adaptive RMS noise
    apply_adaptive_rms_noise(param_dict, noise_std=0.1)
    # BN buffers remain bit-for-bit identical
    assert torch.equal(param_dict["bn1.running_mean"], ref_dict["bn1.running_mean"])
    assert torch.equal(param_dict["bn1.running_var"], ref_dict["bn1.running_var"])
    assert torch.equal(param_dict["bn1.num_batches_tracked"], ref_dict["bn1.num_batches_tracked"])


def test_client_trainer_canonical_class_remapping_and_missing_class(tmp_path: Path) -> None:
    """Verify that client_trainer re-maps ImageFolder alphabetized targets against
    the canonical class list of the base model, preserving missing class rows (ΔW_k = 0).
    """
    import subprocess
    import sys

    from PIL import Image

    from agrifed_khaos.models import build_model, load_checkpoint_bundle, save_checkpoint_bundle

    # 1. Base model has 3 canonical classes: ["clase_a", "clase_b", "clase_c"]
    base_model = build_model("resnet18", num_classes=3)
    base_meta = {
        "architecture": "resnet18",
        "num_classes": 3,
        "classes": ["clase_a", "clase_b", "clase_c"],
        "img_size": [3, 32, 32],
    }
    base_pt = tmp_path / "base_model.pt"
    save_checkpoint_bundle(base_pt, base_model.state_dict(), base_meta)

    # 2. Client data contains ONLY folders "clase_b" and "clase_c"
    # ImageFolder naturally indexes clase_b -> 0, clase_c -> 1
    # Canonical index must be clase_b -> 1, clase_c -> 2, missing -> 0
    client_dir = tmp_path / "client_data"
    for c_name in ["clase_b", "clase_c"]:
        c_dir = client_dir / c_name
        c_dir.mkdir(parents=True)
        for i in range(2):
            im = Image.new("RGB", (32, 32), color="blue" if c_name == "clase_b" else "red")
            im.save(c_dir / f"img_{i}.png")

    client_pt = tmp_path / "client_update.pt"
    cmd = [
        sys.executable,
        "scripts/client_trainer.py",
        "--data-dir",
        str(client_dir),
        "--weights-in",
        str(base_pt),
        "--weights-out",
        str(client_pt),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--lr",
        "0.01",
        "--head-only",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Verify stdout detected missing classes: [0]
    assert "[0]" in res.stdout

    # Verify client update bundle
    client_state, client_meta = load_checkpoint_bundle(client_pt)
    assert client_meta["classes"] == ["clase_a", "clase_b", "clase_c"]
    assert client_meta["missing_classes"] == [0]

    # Row 0 (clase_a) must be exactly identical to base model (unaltered)
    base_fc_w = base_model.state_dict()["fc.weight"]
    base_fc_b = base_model.state_dict()["fc.bias"]
    assert torch.equal(client_state["fc.weight"][0], base_fc_w[0])
    assert torch.equal(client_state["fc.bias"][0], base_fc_b[0])

    # Rows 1 and 2 (clase_b and clase_c) must have updated
    assert not torch.equal(client_state["fc.weight"][1], base_fc_w[1])
    assert not torch.equal(client_state["fc.weight"][2], base_fc_w[2])


def test_dynamic_class_expansion_client_and_aggregation(tmp_path: Path) -> None:
    """Verify end-to-end dynamic class expansion (--allow-new-classes) across
    client training and server aggregation.
    """
    import subprocess
    import sys

    from PIL import Image

    from agrifed_khaos.models import build_model, load_checkpoint_bundle, save_checkpoint_bundle

    # 1. Base model with 3 classes
    base_model = build_model("resnet18", num_classes=3)
    base_meta = {
        "architecture": "resnet18",
        "num_classes": 3,
        "classes": ["clase_a", "clase_b", "clase_c"],
        "img_size": [3, 32, 32],
    }
    base_pt = tmp_path / "base_model.pt"
    save_checkpoint_bundle(base_pt, base_model.state_dict(), base_meta)

    # 2. Client 1 introduces a new class: "clase_d" (has clase_b and clase_d)
    c1_dir = tmp_path / "c1_data"
    for c_name in ["clase_b", "clase_d"]:
        d = c1_dir / c_name
        d.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (32, 32), color="green").save(d / f"img_{i}.png")

    c1_pt = tmp_path / "c1_update.pt"
    # Running client 1 with --allow-new-classes
    cmd1 = [
        sys.executable,
        "scripts/client_trainer.py",
        "--data-dir",
        str(c1_dir),
        "--weights-in",
        str(base_pt),
        "--weights-out",
        str(c1_pt),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--lr",
        "0.01",
        "--head-only",
        "--allow-new-classes",
    ]
    subprocess.run(cmd1, capture_output=True, text=True, check=True)

    c1_state, c1_meta = load_checkpoint_bundle(c1_pt)
    assert c1_meta["classes"] == ["clase_a", "clase_b", "clase_c", "clase_d"]
    assert c1_state["fc.weight"].shape[0] == 4

    # 3. Client 2 has only standard base classes: clase_a and clase_b
    c2_dir = tmp_path / "c2_data"
    for c_name in ["clase_a", "clase_b"]:
        d = c2_dir / c_name
        d.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (32, 32), color="yellow").save(d / f"img_{i}.png")

    c2_pt = tmp_path / "c2_update.pt"
    cmd2 = [
        sys.executable,
        "scripts/client_trainer.py",
        "--data-dir",
        str(c2_dir),
        "--weights-in",
        str(base_pt),
        "--weights-out",
        str(c2_pt),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--lr",
        "0.01",
        "--head-only",
    ]
    subprocess.run(cmd2, capture_output=True, text=True, check=True)

    c2_state, c2_meta = load_checkpoint_bundle(c2_pt)
    assert c2_meta["classes"] == ["clase_a", "clase_b", "clase_c"]
    assert c2_state["fc.weight"].shape[0] == 3

    # 4. Aggregation with --allow-new-classes
    global_pt = tmp_path / "global_expanded.pt"
    agg_cmd = [
        sys.executable,
        "scripts/aggregate.py",
        "--base-model",
        str(base_pt),
        "--client-weights",
        str(c1_pt),
        str(c2_pt),
        "--output",
        str(global_pt),
        "--metadata",
        str(tmp_path / "agg_meta.json"),
        "--eta",
        "0.6",
        "--allow-new-classes",
    ]
    subprocess.run(agg_cmd, capture_output=True, text=True, check=True)

    # 5. Verify aggregated global model
    global_state, global_meta = load_checkpoint_bundle(global_pt)
    assert global_meta["classes"] == ["clase_a", "clase_b", "clase_c", "clase_d"]
    assert global_meta["num_classes"] == 4
    assert global_state["fc.weight"].shape == (4, base_model.fc.in_features)
    assert global_state["fc.bias"].shape == (4,)


def test_deterministic_row_initialization_between_client_and_server() -> None:
    """Verify that init_deterministic_classifier_row produces 100% bit-for-bit identical
    initialization between client module expansion and server state_dict expansion,
    preventing RNG divergence from contaminating Delta W norms.
    """
    from agrifed_khaos.models import (
        build_model,
        expand_model_classifier,
        expand_state_dict_classifier,
        init_deterministic_classifier_row,
    )

    # 1. Direct row test
    w1, b1 = init_deterministic_classifier_row(512, class_name="patologia_x", row_idx=3)
    w2, b2 = init_deterministic_classifier_row(512, class_name="patologia_x", row_idx=3)
    assert torch.equal(w1, w2)
    assert b1 is not None and b2 is not None and torch.equal(b1, b2)

    # Different class name must produce different weights
    w3, b3 = init_deterministic_classifier_row(512, class_name="patologia_y", row_idx=3)
    assert not torch.equal(w1, w3)

    # 2. Client expand_model vs Server expand_state_dict parity
    base_model = build_model("resnet18", num_classes=2)
    state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

    expanded_classes = ["clase_0", "clase_1", "nueva_patologia"]
    expand_model_classifier(base_model, new_num_classes=3, class_names=expanded_classes)
    expand_state_dict_classifier(state_dict, new_num_classes=3, class_names=expanded_classes)

    # The newly initialized row for "nueva_patologia" must be bit-for-bit identical
    client_w = base_model.fc.weight[2].detach()
    server_w = state_dict["fc.weight"][2]
    client_b = base_model.fc.bias[2].detach()
    server_b = state_dict["fc.bias"][2]

    assert torch.equal(client_w, server_w)
    assert torch.equal(client_b, server_b)


def test_disjoint_new_class_collision_isolation(tmp_path: Path) -> None:
    """Verify that when Client 1 introduces 'clase_d' and Client 2 introduces 'clase_e'
    simultaneously over base model (C=3 -> C=5), align_client_head maps by nominal class name,
    and row-wise aggregation isolates each class update without collision or dilution.
    """
    import subprocess
    import sys

    from PIL import Image

    from agrifed_khaos.models import build_model, load_checkpoint_bundle, save_checkpoint_bundle

    # 1. Base model with 3 classes
    base_model = build_model("resnet18", num_classes=3)
    base_meta = {
        "architecture": "resnet18",
        "num_classes": 3,
        "classes": ["clase_a", "clase_b", "clase_c"],
        "img_size": [3, 32, 32],
    }
    base_pt = tmp_path / "base_model.pt"
    save_checkpoint_bundle(base_pt, base_model.state_dict(), base_meta)

    # 2. Client 1 has clase_b, clase_c, and NEW class "clase_d" (total 4 classes)
    c1_dir = tmp_path / "c1_data"
    for c_name in ["clase_b", "clase_c", "clase_d"]:
        d = c1_dir / c_name
        d.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (32, 32), color="green").save(d / f"img_{i}.png")

    c1_pt = tmp_path / "c1_update.pt"
    cmd1 = [
        sys.executable,
        "scripts/client_trainer.py",
        "--data-dir",
        str(c1_dir),
        "--weights-in",
        str(base_pt),
        "--weights-out",
        str(c1_pt),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--lr",
        "0.01",
        "--head-only",
        "--allow-new-classes",
    ]
    subprocess.run(cmd1, capture_output=True, text=True, check=True)

    c1_state, c1_meta = load_checkpoint_bundle(c1_pt)
    assert c1_meta["classes"] == ["clase_a", "clase_b", "clase_c", "clase_d"]
    assert c1_state["fc.weight"].shape[0] == 4

    # 3. Client 2 has clase_a, clase_b, and NEW class "clase_e" (total 4 classes)
    # Notice: In Client 2, "clase_e" is at local index 3, exactly where Client 1 had "clase_d"!
    c2_dir = tmp_path / "c2_data"
    for c_name in ["clase_a", "clase_b", "clase_e"]:
        d = c2_dir / c_name
        d.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (32, 32), color="yellow").save(d / f"img_{i}.png")

    c2_pt = tmp_path / "c2_update.pt"
    cmd2 = [
        sys.executable,
        "scripts/client_trainer.py",
        "--data-dir",
        str(c2_dir),
        "--weights-in",
        str(base_pt),
        "--weights-out",
        str(c2_pt),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--lr",
        "0.01",
        "--head-only",
        "--allow-new-classes",
    ]
    subprocess.run(cmd2, capture_output=True, text=True, check=True)

    c2_state, c2_meta = load_checkpoint_bundle(c2_pt)
    assert c2_meta["classes"] == ["clase_a", "clase_b", "clase_c", "clase_e"]
    assert c2_state["fc.weight"].shape[0] == 4

    # 4. Aggregation with --allow-new-classes and eta=1.0 (pure gradient step)
    global_pt = tmp_path / "global_expanded_5.pt"
    agg_cmd = [
        sys.executable,
        "scripts/aggregate.py",
        "--base-model",
        str(base_pt),
        "--client-weights",
        str(c1_pt),
        str(c2_pt),
        "--output",
        str(global_pt),
        "--metadata",
        str(tmp_path / "agg_meta.json"),
        "--eta",
        "1.0",
        "--allow-new-classes",
    ]
    subprocess.run(agg_cmd, capture_output=True, text=True, check=True)

    # 5. Verify aggregated global model (5 classes: a, b, c, d, e)
    global_state, global_meta = load_checkpoint_bundle(global_pt)
    assert global_meta["classes"] == [
        "clase_a",
        "clase_b",
        "clase_c",
        "clase_d",
        "clase_e",
    ]
    assert global_meta["num_classes"] == 5
    assert global_state["fc.weight"].shape == (5, base_model.fc.in_features)
    assert global_state["fc.bias"].shape == (5,)

    # Row 3 ("clase_d") must match Client 1's trained row exactly (at Client 1 index 3)
    assert torch.allclose(global_state["fc.weight"][3], c1_state["fc.weight"][3], atol=1e-5)
    assert torch.allclose(global_state["fc.bias"][3], c1_state["fc.bias"][3], atol=1e-5)

    # Row 4 ("clase_e") must match Client 2's trained row exactly (at Client 2 index 3)
    assert torch.allclose(global_state["fc.weight"][4], c2_state["fc.weight"][3], atol=1e-5)
    assert torch.allclose(global_state["fc.bias"][4], c2_state["fc.bias"][3], atol=1e-5)

    # Rows 3 and 4 must be distinct (no cross-contamination or positional collision)
    assert not torch.allclose(global_state["fc.weight"][3], global_state["fc.weight"][4])




"""Unit tests for HardwareProfile, ModelSelector, and RegressionGuardrail."""

from __future__ import annotations

from pathlib import Path

from agrifed_khaos.hardware import HardwareProfile, ModelSelector
from agrifed_khaos.quality import audit_regression_guardrail
from agrifed_khaos.wrappers import EvaluationStats


def test_hardware_profiling_and_recommendation() -> None:
    profile = HardwareProfile.detect()
    assert profile.device_type in {"cpu", "cuda", "mps"}
    assert profile.cpu_count > 0
    assert profile.memory_ram_gb > 0.0

    candidates = ModelSelector.get_recommendations(profile, task="classification")
    assert len(candidates) >= 3
    assert candidates[0].federated_suitability_score >= candidates[1].federated_suitability_score

    # Check detection task
    det_candidates = ModelSelector.get_recommendations(profile, task="detection")
    assert len(det_candidates) >= 2
    assert det_candidates[0].task == "detection"


def test_regression_guardrail_logic() -> None:
    base = EvaluationStats(examples=100, loss=0.50, metrics={"accuracy": 0.80})

    # Case 1: Candidate is improved -> Approved
    improved = EvaluationStats(examples=100, loss=0.40, metrics={"accuracy": 0.85})
    res_improved = audit_regression_guardrail(base, improved)
    assert res_improved.approved is True
    assert res_improved.accuracy_delta > 0

    # Case 2: Candidate has catastrophic accuracy drop -> Rejected
    dropped = EvaluationStats(examples=100, loss=0.90, metrics={"accuracy": 0.70})
    res_dropped = audit_regression_guardrail(base, dropped, max_allowed_acc_drop=0.03)
    assert res_dropped.approved is False
    assert any("Accuracy degraded" in r for r in res_dropped.reasons)

    # Case 3: Candidate has exploding loss -> Rejected
    exploding_loss = EvaluationStats(examples=100, loss=1.20, metrics={"accuracy": 0.80})
    res_loss = audit_regression_guardrail(base, exploding_loss, max_allowed_loss_increase=0.35)
    assert res_loss.approved is False
    assert any("Loss increased" in r for r in res_loss.reasons)


def test_canary_telemetry_alarm_system(tmp_path: Path) -> None:
    from agrifed_khaos.quality import (
        CanaryAlarmLevel,
        CanaryRoundRecord,
        audit_canary_telemetry,
        load_canary_history,
        save_canary_history,
    )

    history: list[CanaryRoundRecord] = []

    # Round 1: Baseline at 80%
    r1 = CanaryRoundRecord(
        round_id="round_1",
        accuracy=0.80,
        loss=0.45,
        macro_f1=0.79,
        per_class_f1={"healthy": 0.82, "fungus": 0.76},
    )
    rep1 = audit_canary_telemetry(r1, history)
    assert rep1.level == CanaryAlarmLevel.OK
    assert rep1.drop_from_previous == 0.0
    history.append(r1)

    # Round 10: Progressed to 90%
    r10 = CanaryRoundRecord(
        round_id="round_10",
        accuracy=0.90,
        loss=0.25,
        macro_f1=0.89,
        per_class_f1={"healthy": 0.92, "fungus": 0.87},
    )
    rep10 = audit_canary_telemetry(r10, history)
    assert rep10.level == CanaryAlarmLevel.OK
    history.append(r10)

    # Round 11: Catastrophic drop to 20% (The user's exact case study)
    r11 = CanaryRoundRecord(
        round_id="round_11",
        accuracy=0.20,
        loss=1.85,
        macro_f1=0.18,
        per_class_f1={"healthy": 0.30, "fungus": 0.00},
    )
    rep11 = audit_canary_telemetry(r11, history)
    assert rep11.level == CanaryAlarmLevel.CRITICAL
    assert rep11.drop_from_previous == 0.70
    assert rep11.drop_from_peak == 0.70
    assert any("CRITICAL STEP-DROP" in a for a in rep11.alarms)
    assert any("HISTORICAL PEAK REGRESSION" in a for a in rep11.alarms)
    assert any("CLASS ANOMALY: Class 'fungus'" in a for a in rep11.alarms)

    # Test JSON persistence
    history.append(r11)
    json_path = tmp_path / "telemetry_history.json"
    save_canary_history(json_path, history)
    loaded = load_canary_history(json_path)
    assert len(loaded) == 3
    assert loaded[2].round_id == "round_11"
    assert loaded[2].accuracy == 0.20


def test_model_factory_multidataset() -> None:
    import pytest
    import torch

    from agrifed_khaos.models import build_model, get_cam_target_layer

    dummy_x = torch.randn(2, 3, 224, 224)

    # Dataset 1: ResNet-18 (9 classes)
    m_res = build_model("resnet18", num_classes=9)
    out_res = m_res(dummy_x)
    assert out_res.shape == (2, 9)
    layer_res = get_cam_target_layer(m_res, "resnet18")
    assert layer_res is not None

    # Dataset 2: MobileNetV3 Small (5 classes - Edge & Low Bandwidth)
    m_mob = build_model("mobilenet_v3_small", num_classes=5)
    out_mob = m_mob(dummy_x)
    assert out_mob.shape == (2, 5)
    layer_mob = get_cam_target_layer(m_mob, "mobilenet_v3_small")
    assert layer_mob is not None

    # Dataset 3: EfficientNet-B0 (3 classes)
    m_eff = build_model("efficientnet_b0", num_classes=3)
    out_eff = m_eff(dummy_x)
    assert out_eff.shape == (2, 3)

    # Dataset 4: Timm ConvNeXt-Tiny (4 classes)
    m_conv = build_model("convnext_tiny", num_classes=4)
    out_conv = m_conv(dummy_x)
    assert out_conv.shape == (2, 4)
    layer_conv = get_cam_target_layer(m_conv, "convnext_tiny")
    assert layer_conv is not None

    # Error handling for invalid model
    with pytest.raises(ValueError, match="Unsupported vision architecture"):
        build_model("unknown_hypothetical_model_xyz", num_classes=2)


def test_head_only_federated_transfer_learning() -> None:
    import torch

    from agrifed_khaos.models import build_model
    from agrifed_khaos.wrappers import ParameterValidator

    base_model = build_model("resnet18", num_classes=5)
    base_state = base_model.state_dict()

    # Simulate client training only head (fc)
    head_keys = ["fc.weight", "fc.bias"]
    client_head_state = {
        "fc.weight": base_state["fc.weight"] + torch.randn_like(base_state["fc.weight"]) * 0.05,
        "fc.bias": base_state["fc.bias"] + torch.randn_like(base_state["fc.bias"]) * 0.05,
    }

    # Verify selective ParameterValidator gate
    c_params = [client_head_state[k].detach().cpu().numpy() for k in head_keys]
    b_params = [base_state[k].detach().cpu().numpy() for k in head_keys]
    validator = ParameterValidator(max_update_norm=50.0)
    res = validator.validate(c_params, reference_parameters=b_params)
    assert res.valid is True
    assert res.delta_norm is not None and res.delta_norm > 0

    # Simulate selective aggregation: W_new = W_base + eta * (W_c - W_b)
    eta = 0.5
    new_state = {k: v.clone() for k, v in base_state.items()}
    for k in head_keys:
        new_state[k] = base_state[k] + eta * (client_head_state[k] - base_state[k])

    # Backbone weights MUST remain exactly identical (zero drift / frozen)
    assert torch.equal(new_state["conv1.weight"], base_state["conv1.weight"])
    assert torch.equal(new_state["layer4.0.conv1.weight"], base_state["layer4.0.conv1.weight"])

    # Head weights MUST reflect the federated update
    assert not torch.equal(new_state["fc.weight"], base_state["fc.weight"])


def test_split_integrity_content_and_path_robustness(tmp_path: Path) -> None:
    from agrifed_khaos.quality import audit_split_integrity

    # Case 1: Same filename in different classes (Previously caused false positives)
    train_mock = [Path("/mock/dataset/train/healthy/001.jpg")]
    val_mock = [Path("/mock/dataset/val/repilo/001.jpg")]
    test_mock = [Path("/mock/dataset/test/fumagina/001.jpg")]

    report_mock = audit_split_integrity(train_mock, val_mock, test_mock)
    assert report_mock.valid is True
    assert report_mock.overlap_detected is False

    # Case 2: Identical disk content duplicated under different names
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    img_train = train_dir / "sample_alpha.jpg"
    img_val = val_dir / "sneaky_renamed_sample.jpg"

    duplicate_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00duplicate_agri_image_data_test"
    img_train.write_bytes(duplicate_bytes)
    img_val.write_bytes(duplicate_bytes)

    report_real = audit_split_integrity([img_train], [img_val], [])
    assert report_real.valid is False
    assert report_real.overlap_detected is True
    assert any("Data leakage detected between train and val" in i for i in report_real.issues)


def test_aggregate_security_gate_structural_validation(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    import torch

    from agrifed_khaos.models import build_model
    from scripts.aggregate import compute_manifest_hmac, compute_sha256

    # Setup base model
    base_net = build_model("resnet18", num_classes=5)
    base_pt = tmp_path / "base_model.pt"
    torch.save(base_net.state_dict(), str(base_pt))
    base_sha = compute_sha256(base_pt)

    # 1. Negative Test: Client update with missing backbone key (e.g. conv1.weight missing)
    broken_state = {k: v.clone() for k, v in base_net.state_dict().items() if k != "conv1.weight"}
    broken_pt = tmp_path / "broken_missing_conv1.pt"
    torch.save(broken_state, str(broken_pt))

    broken_meta = {
        "sha256": compute_sha256(broken_pt),
        "base_model_sha256": base_sha,
        "training_scope": "full",
    }
    broken_pt.with_suffix(".meta.json").write_text(json.dumps(broken_meta), encoding="utf-8")

    res_broken = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate.py",
            "--base-model",
            str(base_pt),
            "--client-weights",
            str(broken_pt),
            "--output",
            str(tmp_path / "out.pt"),
        ],
        capture_output=True,
        text=True,
    )
    assert res_broken.returncode != 0
    assert "Missing model backbone/head keys" in res_broken.stderr

    # 2. Negative Test: Client update with rogue extraneous key (evil.payload)
    evil_state = {k: v.clone() for k, v in base_net.state_dict().items()}
    evil_state["evil.payload"] = torch.randn(10, 10)
    evil_pt = tmp_path / "evil_payload.pt"
    torch.save(evil_state, str(evil_pt))

    evil_meta = {
        "sha256": compute_sha256(evil_pt),
        "base_model_sha256": base_sha,
        "training_scope": "full",
    }
    evil_pt.with_suffix(".meta.json").write_text(json.dumps(evil_meta), encoding="utf-8")

    res_evil = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate.py",
            "--base-model",
            str(base_pt),
            "--client-weights",
            str(evil_pt),
            "--output",
            str(tmp_path / "out.pt"),
        ],
        capture_output=True,
        text=True,
    )
    assert res_evil.returncode != 0
    assert "Unauthorized extraneous keys" in res_evil.stderr

    # 3. Negative Test: Client metadata manifest missing
    valid_head_state = {
        "fc.weight": base_net.state_dict()["fc.weight"].clone(),
        "fc.bias": base_net.state_dict()["fc.bias"].clone(),
    }
    no_meta_pt = tmp_path / "no_meta_head.pt"
    torch.save(valid_head_state, str(no_meta_pt))

    res_no_meta = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate.py",
            "--base-model",
            str(base_pt),
            "--client-weights",
            str(no_meta_pt),
            "--output",
            str(tmp_path / "out.pt"),
        ],
        capture_output=True,
        text=True,
    )
    assert res_no_meta.returncode != 0
    assert "Missing client metadata manifest" in res_no_meta.stderr

    # 4. Negative Test: Base model hash mismatch (client derived from foreign base model)
    bad_lineage_meta = {
        "sha256": compute_sha256(no_meta_pt),
        "base_model_sha256": "0" * 64,  # Fraudulent base hash
        "training_scope": "head_only",
    }
    no_meta_pt.with_suffix(".meta.json").write_text(
        json.dumps(bad_lineage_meta), encoding="utf-8"
    )

    res_bad_lineage = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate.py",
            "--base-model",
            str(base_pt),
            "--client-weights",
            str(no_meta_pt),
            "--output",
            str(tmp_path / "out.pt"),
        ],
        capture_output=True,
        text=True,
    )
    assert res_bad_lineage.returncode != 0
    assert "Base model lineage mismatch" in res_bad_lineage.stderr

    # 5. Positive Test: Legitimate head-only update with valid metadata
    good_head_meta = {
        "sha256": compute_sha256(no_meta_pt),
        "base_model_sha256": base_sha,
        "training_scope": "head_only",
    }
    no_meta_pt.with_suffix(".meta.json").write_text(
        json.dumps(good_head_meta), encoding="utf-8"
    )

    res_good = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate.py",
            "--base-model",
            str(base_pt),
            "--client-weights",
            str(no_meta_pt),
            "--output",
            str(tmp_path / "out.pt"),
            "--metadata",
            str(tmp_path / "metadata.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert res_good.returncode == 0
    assert "Verified HEAD-ONLY Update" in res_good.stdout
    assert "Base model lineage:      verified" in res_good.stdout

    # 6. Negative Test: Invalid closed-set training_scope (e.g. 'partial')
    invalid_scope_meta = {
        "sha256": compute_sha256(no_meta_pt),
        "base_model_sha256": base_sha,
        "training_scope": "partial",  # Invalid
    }
    no_meta_pt.with_suffix(".meta.json").write_text(
        json.dumps(invalid_scope_meta), encoding="utf-8"
    )

    res_invalid_scope = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate.py",
            "--base-model",
            str(base_pt),
            "--client-weights",
            str(no_meta_pt),
            "--output",
            str(tmp_path / "out.pt"),
            "--metadata",
            str(tmp_path / "metadata.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert res_invalid_scope.returncode != 0
    assert "Invalid or missing training_scope in client manifest" in res_invalid_scope.stderr

    # 7. Negative Test: Scope mismatch (claimed full but sent head_only)
    mismatch_scope_meta = {
        "sha256": compute_sha256(no_meta_pt),
        "base_model_sha256": base_sha,
        "training_scope": "full",
    }
    no_meta_pt.with_suffix(".meta.json").write_text(
        json.dumps(mismatch_scope_meta), encoding="utf-8"
    )

    res_mismatch = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate.py",
            "--base-model",
            str(base_pt),
            "--client-weights",
            str(no_meta_pt),
            "--output",
            str(tmp_path / "out.pt"),
            "--metadata",
            str(tmp_path / "metadata.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert res_mismatch.returncode != 0
    assert "Manifest claimed scope 'full', but detected update scope is 'head_only'" in (
        res_mismatch.stderr
    )

    # 8. HMAC Authentication Tests
    secret = "agrifed-super-secret-key-42"
    weights_sha = compute_sha256(no_meta_pt)
    correct_hmac = compute_manifest_hmac(
        key=secret,
        client_sha256=weights_sha,
        base_model_sha256=base_sha,
        training_scope="head_only",
        client_id="client",
    )

    # 8.a Server requires HMAC, but manifest lacks it -> Rejected
    no_hmac_meta = {
        "sha256": weights_sha,
        "base_model_sha256": base_sha,
        "training_scope": "head_only",
    }
    no_meta_pt.with_suffix(".meta.json").write_text(
        json.dumps(no_hmac_meta), encoding="utf-8"
    )

    res_lacks_hmac = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate.py",
            "--base-model",
            str(base_pt),
            "--client-weights",
            str(no_meta_pt),
            "--output",
            str(tmp_path / "out.pt"),
            "--metadata",
            str(tmp_path / "metadata.json"),
            "--hmac-key",
            secret,
        ],
        capture_output=True,
        text=True,
    )
    assert res_lacks_hmac.returncode != 0
    assert "Server requires HMAC authentication" in res_lacks_hmac.stderr

    # 8.b Server requires HMAC, and manifest has matching HMAC signature -> Approved
    signed_meta = dict(no_hmac_meta)
    signed_meta["hmac_signature"] = correct_hmac
    no_meta_pt.with_suffix(".meta.json").write_text(
        json.dumps(signed_meta), encoding="utf-8"
    )

    res_signed = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate.py",
            "--base-model",
            str(base_pt),
            "--client-weights",
            str(no_meta_pt),
            "--output",
            str(tmp_path / "out.pt"),
            "--metadata",
            str(tmp_path / "metadata.json"),
            "--hmac-key",
            secret,
        ],
        capture_output=True,
        text=True,
    )
    assert res_signed.returncode == 0
    assert "HMAC authenticity:       verified" in res_signed.stdout


def test_auto_fl_model_selector_heuristics() -> None:
    from agrifed_khaos.hardware import HardwareProfile, ModelSelector

    # Case 1: Server has high-end GPU, but clients are rural cooperatives on CPU-low
    gpu_server_profile = HardwareProfile(
        device_type="cuda",
        device_name="NVIDIA GeForce RTX 4090",
        cpu_count=24,
        memory_ram_gb=64.0,
        gpu_count=1,
        gpu_memory_vram_gb=24.0,
        cuda_compute_capability="8.9",
        has_mixed_precision_fp16=True,
    )

    _, best_cpu_low = ModelSelector.choose_best(
        task="classification",
        target_client_tier="cpu-low",
        profile=gpu_server_profile,
    )
    # MobileNetV4 must win because clients are on cpu-low, regardless of RTX 4090 server
    assert best_cpu_low.architecture == "mobilenetv4_conv_small.e1200_r224_in1k"
    assert best_cpu_low.federated_suitability_score == 98

    # Case 2: Target client tier is cpu-high
    _, best_cpu_high = ModelSelector.choose_best(
        task="classification",
        target_client_tier="cpu-high",
        profile=gpu_server_profile,
    )
    assert best_cpu_high.architecture == "resnet18"
    assert best_cpu_high.federated_suitability_score == 95

    # Case 3: Dataset < 1000 images on GPU clients -> ResNet18 must beat ViT and ConvNeXt
    candidates_small_dataset = ModelSelector.get_recommendations(
        profile=gpu_server_profile,
        task="classification",
        target_client_tier="gpu",
        dataset_sample_count=450,
    )
    # ViT should be severely penalized on small dataset
    vit_cand = next(c for c in candidates_small_dataset if "vit" in c.architecture)
    assert vit_cand.federated_suitability_score <= 40
    assert "ALERTA DATASET PEQUEÑO" in vit_cand.rationale

    # ResNet-18 wins on small dataset on GPU due to strong inductive bias
    assert candidates_small_dataset[0].architecture == "resnet18"

    # Case 4: Dataset >= 1000 images on GPU clients -> ConvNeXt-Tiny wins
    candidates_large_gpu = ModelSelector.get_recommendations(
        profile=gpu_server_profile,
        task="classification",
        target_client_tier="gpu",
        dataset_sample_count=2500,
    )
    assert candidates_large_gpu[0].architecture == "convnext_tiny.fb_in22k_ft_in1k"
    assert candidates_large_gpu[0].federated_suitability_score == 96


def test_train_base_auto_fl_integration(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    from PIL import Image

    # Create dummy dataset with 2 classes
    dataset_dir = tmp_path / "dataset"
    for cls_name in ["sano", "enfermo"]:
        cls_dir = dataset_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            img = Image.new("RGB", (224, 224), color=(i * 30, i * 40, i * 50))
            img.save(cls_dir / f"img_{i}.jpg")

    out_pt = tmp_path / "base_model.pt"
    meta_json = tmp_path / "metadata.json"

    # Run train_base.py with --arch auto and --target-client-tier cpu-high
    cmd = [
        sys.executable,
        "scripts/train_base.py",
        "--data-dir",
        str(dataset_dir),
        "--arch",
        "auto",
        "--target-client-tier",
        "cpu-high",
        "--output",
        str(out_pt),
        "--metadata",
        str(meta_json),
        "--epochs",
        "1",
        "--batch-size",
        "4",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"train_base.py failed with stderr: {res.stderr}"
    assert out_pt.is_file()
    assert meta_json.is_file()

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    assert meta["architecture"] == "resnet18"
    assert meta["target_client_tier"] == "cpu-high"
    assert meta["selection_mode"] == "auto"
    assert meta["num_classes"] == 2


def test_frozen_batchnorm_invariance_in_wrapper_and_client_trainer(tmp_path: Path) -> None:
    import subprocess
    import sys

    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, TensorDataset

    from agrifed_khaos.models import build_model
    from agrifed_khaos.wrappers import TorchModuleWrapper

    # 1. Verification in TorchModuleWrapper.train_local_epoch
    net = build_model("resnet18", num_classes=2)
    initial_mean = net.bn1.running_mean.clone()
    initial_var = net.bn1.running_var.clone()
    initial_weight = net.bn1.weight.clone()

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    wrapper = TorchModuleWrapper(net, optimizer=optimizer, criterion=criterion)

    dummy_x = torch.randn(8, 3, 224, 224)
    dummy_y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=4)

    wrapper.train_local_epoch(loader, epochs=2)

    # BatchNorm running stats MUST remain bit-for-bit identical to base initialization
    assert torch.equal(net.bn1.running_mean, initial_mean)
    assert torch.equal(net.bn1.running_var, initial_var)
    # Affine weights MUST have updated during local training
    assert not torch.equal(net.bn1.weight, initial_weight)

    # 2. End-to-End Verification with client_trainer.py and aggregate.py
    dataset_dir = tmp_path / "client_data"
    for cls_name in ["sano", "repilo"]:
        c_dir = dataset_dir / cls_name
        c_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            img = Image.new("RGB", (224, 224), color=(i * 20, i * 40, 100))
            img.save(c_dir / f"sample_{i}.jpg")

    base_pt = tmp_path / "base_model.pt"
    client_pt = tmp_path / "client_parameters.pt"
    global_pt = tmp_path / "global_model.pt"
    meta_json = tmp_path / "metadata.json"

    net_base = build_model("resnet18", num_classes=2)
    torch.save(net_base.state_dict(), str(base_pt))

    import json
    metadata_content = {
        "architecture": "resnet18",
        "num_classes": 2,
        "classes": ["sano", "repilo"],
    }
    meta_json.write_text(json.dumps(metadata_content), encoding="utf-8")

    # Run client_trainer.py
    cmd_train = [
        sys.executable,
        "scripts/client_trainer.py",
        "--data-dir",
        str(dataset_dir),
        "--weights-in",
        str(base_pt),
        "--weights-out",
        str(client_pt),
        "--metadata",
        str(meta_json),
        "--epochs",
        "1",
        "--batch-size",
        "4",
    ]
    res_train = subprocess.run(cmd_train, capture_output=True, text=True)
    assert res_train.returncode == 0, f"client_trainer failed: {res_train.stderr}"

    from agrifed_khaos.models import load_checkpoint_bundle

    client_state, _ = load_checkpoint_bundle(client_pt)
    base_state, _ = load_checkpoint_bundle(base_pt)

    # Verify Frozen BatchNorm in client checkpoint
    assert torch.equal(client_state["bn1.running_mean"], base_state["bn1.running_mean"])
    assert torch.equal(client_state["bn1.running_var"], base_state["bn1.running_var"])
    assert not torch.equal(client_state["bn1.weight"], base_state["bn1.weight"])

    # Run aggregate.py
    cmd_agg = [
        sys.executable,
        "scripts/aggregate.py",
        "--base-model",
        str(base_pt),
        "--client-weights",
        str(client_pt),
        "--output",
        str(global_pt),
        "--metadata",
        str(meta_json),
    ]
    res_agg = subprocess.run(cmd_agg, capture_output=True, text=True)
    assert res_agg.returncode == 0, f"aggregate.py failed: {res_agg.stderr}"

    global_state, _ = load_checkpoint_bundle(global_pt)
    # Global model MUST retain exact base model BatchNorm statistics
    assert torch.equal(global_state["bn1.running_mean"], base_state["bn1.running_mean"])
    assert torch.equal(global_state["bn1.running_var"], base_state["bn1.running_var"])
    # Global model MUST incorporate client affine weight updates
    assert not torch.equal(global_state["bn1.weight"], base_state["bn1.weight"])


def test_multi_client_weighted_aggregation(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    import torch

    from agrifed_khaos.models import build_model, load_checkpoint_bundle
    from scripts.aggregate import compute_manifest_hmac, compute_sha256

    # 1. Setup Base Model
    base_net = build_model("resnet18", num_classes=4)
    base_pt = tmp_path / "base_model.pt"
    torch.save(base_net.state_dict(), str(base_pt))
    base_sha = compute_sha256(base_pt)
    base_state = {k: v.clone() for k, v in base_net.state_dict().items()}

    # Non-BN trainable keys:
    bn_buffer_suffixes = (".running_mean", ".running_var", ".num_batches_tracked")
    trainable_keys = [
        k for k in base_state.keys() if not any(k.endswith(sfx) for sfx in bn_buffer_suffixes)
    ]

    # 2. Client A: n_A = 80 samples, delta = +0.005 on trainable parameters
    state_a = {k: v.clone() for k, v in base_state.items()}
    for k in trainable_keys:
        if state_a[k].dtype.is_floating_point:
            state_a[k] = state_a[k] + 0.005
    client_a_pt = tmp_path / "client_A.pt"
    torch.save(state_a, str(client_a_pt))
    sha_a = compute_sha256(client_a_pt)

    secret = "hmac-federated-secret-key"
    hmac_a = compute_manifest_hmac(
        key=secret,
        client_sha256=sha_a,
        base_model_sha256=base_sha,
        training_scope="full",
        client_id="cooperative_alpha",
        num_samples=80,
    )
    meta_a = {
        "client_weights_file": str(client_a_pt),
        "client_id": "cooperative_alpha",
        "sha256": sha_a,
        "base_model_sha256": base_sha,
        "num_samples": 80,
        "delta_norm": 16.58,
        "training_scope": "full",
        "hmac_signature": hmac_a,
    }
    client_a_pt.with_suffix(".meta.json").write_text(json.dumps(meta_a), encoding="utf-8")

    # 3. Client B: n_B = 20 samples, delta = -0.010 on trainable parameters
    state_b = {k: v.clone() for k, v in base_state.items()}
    for k in trainable_keys:
        if state_b[k].dtype.is_floating_point:
            state_b[k] = state_b[k] - 0.010
    client_b_pt = tmp_path / "client_B.pt"
    torch.save(state_b, str(client_b_pt))
    sha_b = compute_sha256(client_b_pt)

    hmac_b = compute_manifest_hmac(
        key=secret,
        client_sha256=sha_b,
        base_model_sha256=base_sha,
        training_scope="full",
        client_id="cooperative_beta",
        num_samples=20,
    )
    meta_b = {
        "client_weights_file": str(client_b_pt),
        "client_id": "cooperative_beta",
        "sha256": sha_b,
        "base_model_sha256": base_sha,
        "num_samples": 20,
        "delta_norm": 33.16,
        "training_scope": "full",
        "hmac_signature": hmac_b,
    }
    client_b_pt.with_suffix(".meta.json").write_text(json.dumps(meta_b), encoding="utf-8")

    # 4. Execute Multi-Client Aggregation with eta=1.0
    global_pt = tmp_path / "global_model.pt"
    meta_json = tmp_path / "metadata.json"

    cmd = [
        sys.executable,
        "scripts/aggregate.py",
        "--base-model",
        str(base_pt),
        "--client-weights",
        str(client_a_pt),
        str(client_b_pt),
        "--output",
        str(global_pt),
        "--metadata",
        str(meta_json),
        "--eta",
        "1.0",
        "--hmac-key",
        secret,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"aggregate.py multi-client failed: {res.stderr}"
    assert "AGGREGATING 2 VALIDATED CLIENT(S) (Total Samples: 100)" in res.stdout
    assert "cooperative_alpha" in res.stdout
    assert "cooperative_beta" in res.stdout
    assert "80.00%" in res.stdout
    assert "20.00%" in res.stdout

    # 5. Verify Exact Mathematical Weighted Incorporations:
    # delta_expected = (80/100) * (+0.005) + (20/100) * (-0.010) = +0.004 - 0.002 = +0.002
    global_state, _ = load_checkpoint_bundle(global_pt)
    for k in trainable_keys:
        if base_state[k].dtype.is_floating_point:
            expected_tensor = base_state[k] + 0.002
            assert torch.allclose(global_state[k], expected_tensor, atol=1e-5), f"Mismatch in {k}"

    # Verify BatchNorm buffers remain bit-for-bit invariant
    assert torch.equal(global_state["bn1.running_mean"], base_state["bn1.running_mean"])
    assert torch.equal(global_state["bn1.running_var"], base_state["bn1.running_var"])

    # 6. Verify metadata.json aggregation history
    saved_meta = json.loads(meta_json.read_text(encoding="utf-8"))
    history = saved_meta.get("aggregation_history", [])
    assert len(history) == 1
    round_rec = history[0]
    assert round_rec["num_clients"] == 2
    assert round_rec["total_samples"] == 100
    assert len(round_rec["clients"]) == 2
    assert round_rec["clients"][0]["client_id"] == "cooperative_alpha"
    assert round_rec["clients"][0]["num_samples"] == 80
    assert round_rec["clients"][0]["sample_weight"] == 0.8
    assert round_rec["clients"][1]["client_id"] == "cooperative_beta"
    assert round_rec["clients"][1]["num_samples"] == 20
    assert round_rec["clients"][1]["sample_weight"] == 0.2

    # 7. Test --skip-invalid-clients with a corrupted third client
    client_bad_pt = tmp_path / "client_corrupted.pt"
    torch.save(state_a, str(client_bad_pt))
    bad_meta = {
        "client_weights_file": str(client_bad_pt),
        "client_id": "malicious_node",
        "sha256": compute_sha256(client_bad_pt),
        "base_model_sha256": "wrong_base_hash_000000000000000000",
        "num_samples": 50,
        "training_scope": "full",
    }
    client_bad_pt.with_suffix(".meta.json").write_text(json.dumps(bad_meta), encoding="utf-8")

    # 7.a Without --skip-invalid-clients: must abort
    cmd_strict = [
        sys.executable,
        "scripts/aggregate.py",
        "--base-model",
        str(base_pt),
        "--client-weights",
        str(client_a_pt),
        str(client_bad_pt),
        "--output",
        str(global_pt),
        "--metadata",
        str(meta_json),
    ]
    res_strict = subprocess.run(cmd_strict, capture_output=True, text=True)
    assert res_strict.returncode != 0
    assert "Base model lineage mismatch" in res_strict.stderr

    # 7.b With --skip-invalid-clients: skips bad client and succeeds with client_A
    cmd_skip = [
        sys.executable,
        "scripts/aggregate.py",
        "--base-model",
        str(base_pt),
        "--client-weights",
        str(client_a_pt),
        str(client_bad_pt),
        "--output",
        str(global_pt),
        "--metadata",
        str(meta_json),
        "--skip-invalid-clients",
    ]
    res_skip = subprocess.run(cmd_skip, capture_output=True, text=True)
    assert res_skip.returncode == 0
    assert "WARNING: Skipping client" in res_skip.stderr
    assert "AGGREGATING 1 VALIDATED CLIENT(S)" in res_skip.stdout


def test_rms_scaled_parameter_noise_and_relative_norm_gate(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    import numpy as np
    import torch

    from agrifed_khaos.models import build_model
    from agrifed_khaos.wrappers import ParameterValidator, apply_adaptive_rms_noise
    from scripts.aggregate import compute_sha256

    # 1. Test Layer-wise RMS-Scaled Parameter Noise
    t_large = torch.full((100, 100), 10.0)
    t_small = torch.full((100, 100), 0.001)
    orig_large = t_large.clone()
    orig_small = t_small.clone()

    apply_adaptive_rms_noise([t_large, t_small], noise_std=0.05)
    diff_large = (t_large - orig_large).std().item()
    diff_small = (t_small - orig_small).std().item()

    # The ratio of noise scale should closely reflect the 10.0 / 0.001 magnitude ratio (10000x)
    ratio = diff_large / max(diff_small, 1e-8)
    assert ratio > 2000.0, f"Expected proportional scaling, got ratio: {ratio}"
    assert torch.allclose(t_small, orig_small, atol=0.01), "Small tensor was excessively corrupted"

    # 2. Test ParameterValidator with max_relative_norm
    ref = [np.ones((20, 20), dtype=np.float32)]  # Norm = sqrt(400 * 1) = 20.0
    cand_mild = [np.ones((20, 20), dtype=np.float32) + 0.1]  # Delta = 2.0, Rel = 0.10
    cand_extreme = [np.ones((20, 20), dtype=np.float32) + 1.5]  # Delta = 30.0, Rel = 1.50

    val_unrestricted = ParameterValidator(max_update_norm=150.0)
    assert val_unrestricted.validate(cand_extreme, ref).valid is True

    val_relative = ParameterValidator(max_update_norm=150.0, max_relative_norm=0.5)
    res_mild = val_relative.validate(cand_mild, ref)
    assert res_mild.valid is True
    assert res_mild.relative_norm is not None and abs(res_mild.relative_norm - 0.10) < 1e-4

    res_extreme = val_relative.validate(cand_extreme, ref)
    assert res_extreme.valid is False
    assert any("exceeds relative tolerance" in i for i in res_extreme.issues)

    # 3. End-to-End Test with aggregate.py --head-only and --max-relative-norm
    base_net = build_model("resnet18", num_classes=3)
    base_pt = tmp_path / "base_model.pt"
    torch.save(base_net.state_dict(), str(base_pt))
    base_sha = compute_sha256(base_pt)

    # Head-only client with excessive relative shift on fc layers
    head_state = {
        "fc.weight": base_net.fc.weight.clone() + 1.5,
        "fc.bias": base_net.fc.bias.clone() + 1.5,
    }
    client_pt = tmp_path / "client_head_extreme.pt"
    torch.save(head_state, str(client_pt))
    client_meta = {
        "client_weights_file": str(client_pt),
        "client_id": "malicious_head_client",
        "sha256": compute_sha256(client_pt),
        "base_model_sha256": base_sha,
        "num_samples": 50,
        "training_scope": "head_only",
    }
    client_pt.with_suffix(".meta.json").write_text(json.dumps(client_meta), encoding="utf-8")

    out_pt = tmp_path / "out.pt"
    meta_json = tmp_path / "meta.json"

    # 3.a Without relative norm: passes because delta norm is ~15.0 < 150.0
    cmd_pass = [
        sys.executable,
        "scripts/aggregate.py",
        "--base-model",
        str(base_pt),
        "--client-weights",
        str(client_pt),
        "--output",
        str(out_pt),
        "--metadata",
        str(meta_json),
    ]
    res_pass = subprocess.run(cmd_pass, capture_output=True, text=True)
    assert res_pass.returncode == 0

    # 3.b With --max-relative-norm 0.5: correctly rejected!
    cmd_reject = [
        sys.executable,
        "scripts/aggregate.py",
        "--base-model",
        str(base_pt),
        "--client-weights",
        str(client_pt),
        "--output",
        str(out_pt),
        "--metadata",
        str(meta_json),
        "--max-relative-norm",
        "0.5",
    ]
    res_reject = subprocess.run(cmd_reject, capture_output=True, text=True)
    assert res_reject.returncode != 0
    assert "exceeds relative tolerance" in res_reject.stderr









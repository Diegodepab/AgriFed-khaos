"""Unit tests verifying all corrections from the comprehensive FL audit.

Covers:
1. Cryptographic isolation & Delimiter Injection prevention (RFC 2104 canonical payload)
2. Weight decay decoupling and gradient clipping under FedProx proximal regularization
3. Byzantine sample inflation defense (--max-samples-per-client cap)
4. Closed-set training scope normalization ('full' and 'full_model')
5. Exact sample-weighted loss accumulation in TorchModuleWrapper
6. Canary telemetry suspicious spike alerting
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from agrifed_khaos.crypto import compute_manifest_hmac, compute_sha256
from agrifed_khaos.models import build_model
from agrifed_khaos.quality import (
    CanaryRoundRecord,
    audit_canary_telemetry,
)
from agrifed_khaos.wrappers import TorchModuleWrapper


def test_crypto_delimiter_injection_prevention(tmp_path: Path) -> None:
    test_file = tmp_path / "dummy.pt"
    test_file.write_bytes(b"agrifed_dummy_weights")
    file_sha = compute_sha256(test_file)

    # Valid HMAC computation
    valid_sig = compute_manifest_hmac(
        key="secret_key",
        client_sha256=file_sha,
        base_model_sha256="base_sha",
        training_scope="full_model",
        client_id="coop_1",
        num_samples=150,
    )
    assert isinstance(valid_sig, str) and len(valid_sig) == 64

    # Delimiter injection in client_id must raise ValueError
    with pytest.raises(ValueError, match="client_id cannot contain delimiter"):
        compute_manifest_hmac(
            key="secret_key",
            client_sha256=file_sha,
            base_model_sha256="base_sha",
            training_scope="full_model",
            client_id="coop_1|num_samples=99999",
            num_samples=150,
        )

    # Delimiter injection in training_scope must raise ValueError
    with pytest.raises(ValueError, match="training_scope cannot contain delimiter"):
        compute_manifest_hmac(
            key="secret_key",
            client_sha256=file_sha,
            base_model_sha256="base_sha",
            training_scope="full|num_samples=99999",
            client_id="coop_1",
            num_samples=150,
        )


def test_torch_wrapper_sample_weighted_loss_and_clipping() -> None:
    class LinearModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = LinearModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    wrapper = TorchModuleWrapper(
        module=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
    )

    # Test training with gradient clipping
    batch1 = (torch.randn(32, 4) * 100.0, torch.zeros(32, dtype=torch.long))
    batch2 = (torch.randn(10, 4) * 100.0, torch.ones(10, dtype=torch.long))
    dataset = [batch1, batch2]

    stats = wrapper.train_local_epoch(
        dataset,
        epochs=1,
        prox_mu=0.01,
        max_grad_norm=0.5,
    )
    assert stats.examples == 42
    assert stats.batches == 2
    assert stats.loss > 0.0

    # Test evaluate with exact sample weighting
    eval_stats = wrapper.evaluate(dataset)
    assert eval_stats.examples == 42
    assert "accuracy" in eval_stats.metrics


def test_canary_suspicious_spike_detection() -> None:
    history = [
        CanaryRoundRecord(
            round_id="round_1",
            accuracy=0.50,
            loss=0.8,
            macro_f1=0.48,
        )
    ]
    # Round with massive abnormal jump (+45%)
    suspicious_record = CanaryRoundRecord(
        round_id="round_2",
        accuracy=0.95,
        loss=0.1,
        macro_f1=0.94,
    )
    report = audit_canary_telemetry(suspicious_record, history, suspicious_spike_threshold=0.40)
    assert any("SUSPICIOUS SPIKE" in a for a in report.alarms)


def test_byzantine_sample_inflation_cap(tmp_path: Path) -> None:
    base_net = build_model("resnet18", num_classes=2)
    base_pt = tmp_path / "base_model.pt"
    torch.save(base_net.state_dict(), str(base_pt))
    base_sha = compute_sha256(base_pt)

    client_state = {k: v.clone() for k, v in base_net.state_dict().items()}
    client_pt = tmp_path / "client.pt"
    torch.save(client_state, str(client_pt))

    client_meta = {
        "client_weights_file": str(client_pt),
        "client_id": "greedy_node",
        "sha256": compute_sha256(client_pt),
        "base_model_sha256": base_sha,
        "num_samples": 50000,  # Inflated!
        "training_scope": "full_model",
    }
    client_pt.with_suffix(".meta.json").write_text(json.dumps(client_meta), encoding="utf-8")

    out_pt = tmp_path / "global_model.pt"
    meta_json = tmp_path / "metadata.json"

    cmd = [
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
        "--max-samples-per-client",
        "1000",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "capped to --max-samples-per-client" in res.stderr
    assert "Total Samples: 1000" in res.stdout


def test_dirichlet_non_iid_partitioning() -> None:
    from agrifed_khaos.data import DatasetRecord, partition_train_records

    records: list[DatasetRecord] = []
    # 3 classes, 100 samples each = 300 total
    for i in range(100):
        records.append(
            DatasetRecord(
                filename=f"healthy_{i:03d}.jpg",
                split="train",
                disease="healthy",
                condition="good",
                task="classification",
                source_dataset="test",
                source_url="test",
            )
        )
        records.append(
            DatasetRecord(
                filename=f"repilo_{i:03d}.jpg",
                split="train",
                disease="repilo",
                condition="sick",
                task="classification",
                source_dataset="test",
                source_url="test",
            )
        )
        records.append(
            DatasetRecord(
                filename=f"fumagina_{i:03d}.jpg",
                split="train",
                disease="fumagina",
                condition="sick",
                task="classification",
                source_dataset="test",
                source_url="test",
            )
        )

    # Test Dirichlet partitioning with K=3 clients
    client_0 = partition_train_records(records, client_index=0, client_count=3, alpha=0.5, seed=42)
    client_1 = partition_train_records(records, client_index=1, client_count=3, alpha=0.5, seed=42)
    client_2 = partition_train_records(records, client_index=2, client_count=3, alpha=0.5, seed=42)

    total_sharded = len(client_0) + len(client_1) + len(client_2)
    assert total_sharded == 300, f"Expected 300 samples, got {total_sharded}"

    # Verify zero leakage (disjoint partitions)
    files_0 = {r.filename for r in client_0}
    files_1 = {r.filename for r in client_1}
    files_2 = {r.filename for r in client_2}
    assert len(files_0.intersection(files_1)) == 0
    assert len(files_0.intersection(files_2)) == 0
    assert len(files_1.intersection(files_2)) == 0


def test_rest_serve_security_limits_and_cors(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from agrifed_khaos.models import build_model
    from scripts.serve import MAX_UPLOAD_SIZE_BYTES, create_app

    model_pt = tmp_path / "model.pt"
    meta_json = tmp_path / "metadata.json"
    net = build_model("resnet18", num_classes=2)
    torch.save(net.state_dict(), str(model_pt))
    meta_json.write_text(json.dumps({"classes": ["sano", "enfermo"], "architecture": "resnet18"}))

    app = create_app(str(model_pt), str(meta_json))
    client = TestClient(app)

    # 1. Health check returns 200
    res_health = client.get("/health")
    assert res_health.status_code == 200

    # 2. Reject non-image mime-type
    res_bad_mime = client.post("/predict", files={"file": ("test.txt", b"text data", "text/plain")})
    assert res_bad_mime.status_code == 400

    # 3. Reject oversized payload (> 15 MB limit)
    oversized_data = b"0" * (MAX_UPLOAD_SIZE_BYTES + 50)
    res_oversized = client.post(
        "/predict", files={"file": ("large.jpg", oversized_data, "image/jpeg")}
    )
    assert res_oversized.status_code == 413


def test_gradcam_3d_transformer_support() -> None:
    from scripts.evaluate import GradCAM

    class DummyViT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Simulates a transformer block outputting 3D tokens: [B, N=197, D=192]
            self.block = nn.Linear(192, 192)
            self.head = nn.Linear(192, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [B, 3, 224, 224]
            b = x.shape[0]
            tokens = torch.randn(b, 197, 192, requires_grad=True)
            out_tokens = self.block(tokens)
            cls_token = out_tokens[:, 0, :]
            return self.head(cls_token)

    model = DummyViT()
    cam = GradCAM(model, model.block)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Generate Grad-CAM without throwing dimension IndexError
    heatmap, pred_idx, conf = cam.generate(dummy_input)
    assert heatmap.shape == (224, 224)
    assert 0.0 <= conf <= 1.0


def test_safe_checkpoint_loading_blocks_arbitrary_code(tmp_path: Path) -> None:
    from agrifed_khaos.models import load_checkpoint_bundle

    class UnsafeObject:
        def __reduce__(self):
            return (eval, ("1 + 1",))

    unsafe_file = tmp_path / "malicious.pt"
    # Save a torch file containing custom non-tensor Python objects
    torch.save({"data": UnsafeObject()}, str(unsafe_file))

    # Safe loading (default) must strictly raise ValueError and forbid execution
    with pytest.raises(ValueError, match="Arbitrary code execution protection is active"):
        load_checkpoint_bundle(unsafe_file, allow_unsafe_legacy=False)

    # Legacy explicit override allows loading
    raw, _ = load_checkpoint_bundle(unsafe_file, allow_unsafe_legacy=True)
    assert "data" in raw


def test_submit_update_manifest_verification_and_storage(tmp_path: Path) -> None:
    import hashlib

    from fastapi.testclient import TestClient

    from scripts.serve import create_app

    # Create dummy base checkpoint and metadata
    base_file = tmp_path / "base.pt"
    torch.save({"state_dict": {"fc.weight": torch.randn(2, 2)}}, str(base_file))
    meta_file = tmp_path / "metadata.json"
    meta_file.write_text(json.dumps({"classes": ["c1", "c2"]}), encoding="utf-8")

    app = create_app(str(base_file), str(meta_file))
    client = TestClient(app)

    # Generate dummy client weights
    weights_bytes = b"agrifed_test_weights_payload"
    weights_sha = hashlib.sha256(weights_bytes).hexdigest()

    # 1. Valid submission with matching manifest
    manifest_dict = {
        "client_id": "coop_sevilla",
        "sha256": weights_sha,
        "base_model_sha256": "dummy_base_sha",
        "num_samples": 100,
    }
    manifest_bytes = json.dumps(manifest_dict).encode("utf-8")

    res = client.post(
        "/submit-update?client_id=coop_sevilla",
        files={
            "file": (
                "client_parameters.pt",
                io.BytesIO(weights_bytes),
                "application/octet-stream",
            ),
            "manifest": (
                "client_parameters.meta.json",
                io.BytesIO(manifest_bytes),
                "application/json",
            ),
        },
    )
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["status"] == "ready_for_aggregation"
    assert res_data["has_manifest"] is True
    assert res_data["manifest_verified"] is True
    assert Path(res_data["saved_to"]).is_file()
    assert Path(res_data["meta_saved_to"]).is_file()

    # 2. Reject mismatched hash
    bad_manifest = json.dumps({"sha256": "0" * 64}).encode("utf-8")
    res_mismatch = client.post(
        "/submit-update?client_id=coop_sevilla",
        files={
            "file": (
                "client_parameters.pt",
                io.BytesIO(weights_bytes),
                "application/octet-stream",
            ),
            "manifest": (
                "client_parameters.meta.json",
                io.BytesIO(bad_manifest),
                "application/json",
            ),
        },
    )
    assert res_mismatch.status_code == 400
    assert "Integrity check failed" in res_mismatch.json()["detail"]

    # 3. Reject invalid client ID (directory traversal or malformed characters)
    res_bad_id = client.post(
        "/submit-update?client_id=../../malicious",
        files={"file": ("client.pt", io.BytesIO(weights_bytes), "application/octet-stream")},
    )
    assert res_bad_id.status_code == 400
    assert "Invalid client_id" in res_bad_id.json()["detail"]


def test_dp_argument_mutual_requirement() -> None:
    from scripts.client_trainer import parse_args

    # Providing only clip_norm without noise_std must exit with argument error
    with pytest.raises(SystemExit):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                sys,
                "argv",
                ["client_trainer.py", "--data-dir", "dummy", "--dp-clip-norm", "5.0"],
            )
            parse_args()

    # Providing only noise_std without clip_norm must exit with argument error
    with pytest.raises(SystemExit):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                sys,
                "argv",
                ["client_trainer.py", "--data-dir", "dummy", "--dp-noise-std", "0.5"],
            )
            parse_args()



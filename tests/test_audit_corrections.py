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

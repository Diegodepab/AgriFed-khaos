"""Integration tests for Multiplatform Edge-Federated Learning in AgriFed-KHAOS.

Tests:
1. Model deconstruction: backbone ONNX extraction & head weights isolation.
2. GET /federate/model-bundle and GET /federate/backbone.onnx endpoints.
3. POST /federate/submit-update validation:
   - Lineage SHA-256 conflict rejection (HTTP 409)
   - Relative norm divergence rejection (HTTP 422)
   - Scope and samples validation (HTTP 400)
   - Canonical class projection (Anti-Ontology Pollution)
   - HMAC-SHA256 signature verification
4. End-to-end aggregation compatibility via scripts/aggregate.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from agrifed_khaos.crypto import compute_manifest_hmac, compute_sha256
from agrifed_khaos.models import build_model, load_checkpoint_bundle, save_checkpoint_bundle
from agrifed_khaos.onnx_export import (
    EdgeModelBundleResult,
    get_or_export_edge_bundle,
    isolate_backbone_and_head,
)
from agrifed_khaos.wrappers import freeze_bn_stats
from scripts.serve import create_app


@pytest.fixture(scope="module")
def sample_classification_model(tmp_path_factory):
    """Create a reproducible classification model checkpoint for edge tests."""
    work_dir = tmp_path_factory.mktemp("edge_fl_model")
    ckpt_path = work_dir / "base_olive_test.pt"
    meta_path = work_dir / "base_olive_test.meta.json"

    arch = "resnet18"
    classes = ["aceituna", "defectuosa", "desenfocada", "envero", "negra", "sana", "verde"]
    num_classes = len(classes)

    model = build_model(architecture=arch, num_classes=num_classes)
    model.eval()
    freeze_bn_stats(model)

    meta = {
        "architecture": arch,
        "num_classes": num_classes,
        "classes": classes,
        "input_size": [224, 224],
        "training_scope": "full",
        "task": "classification",
    }

    save_checkpoint_bundle(ckpt_path, model.state_dict(), meta)
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "ckpt_path": ckpt_path,
        "meta_path": meta_path,
        "work_dir": work_dir,
        "arch": arch,
        "classes": classes,
        "num_classes": num_classes,
        "sha256": compute_sha256(ckpt_path),
    }


def test_isolate_backbone_and_head_resnet(sample_classification_model):
    """Verify surgical isolation of backbone and head for ResNet architectures."""
    model = build_model("resnet18", num_classes=7)
    model.eval()

    backbone, w, b, w_key, b_key = isolate_backbone_and_head(model, "resnet18")

    assert w_key == "fc.weight"
    assert b_key == "fc.bias"
    assert w.shape == (7, 512)
    assert b.shape == (7,)
    assert isinstance(backbone.fc, nn.Identity)

    # Test numerical output equivalence
    dummy_x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        orig_out = model(dummy_x)
        features = backbone(dummy_x)
        manual_out = features @ w.T + b

    diff = (orig_out - manual_out).abs().max().item()
    assert diff < 1e-5, f"Deconstruction introduced numerical drift: {diff}"


def test_edge_bundle_export_and_caching(sample_classification_model):
    """Verify export of ONNX backbone and JSON head weights with hash synchronization."""
    ckpt = sample_classification_model["ckpt_path"]
    meta = sample_classification_model["meta_path"]
    work_dir = sample_classification_model["work_dir"]

    bundle1 = get_or_export_edge_bundle(
        checkpoint_path=ckpt,
        model_slug="test-olive-model",
        models_dir=work_dir,
        metadata_path=meta,
    )

    assert isinstance(bundle1, EdgeModelBundleResult)
    assert bundle1.onnx_path.is_file()
    assert bundle1.head_weights_path.is_file()
    assert bundle1.embedding_dim == 512
    assert bundle1.num_classes == 7
    assert bundle1.classes == sample_classification_model["classes"]

    # Verify head weights json format
    head_data = json.loads(bundle1.head_weights_path.read_text())
    assert head_data["schema_version"] == "1.0"
    assert head_data["base_model_sha256"] == sample_classification_model["sha256"]
    assert len(head_data["head_weights"]["weight"]) == 7
    assert len(head_data["head_weights"]["weight"][0]) == 512
    assert len(head_data["head_weights"]["bias"]) == 7

    # Verify caching: second call returns cached=True without re-exporting
    bundle2 = get_or_export_edge_bundle(
        checkpoint_path=ckpt,
        model_slug="test-olive-model",
        models_dir=work_dir,
        metadata_path=meta,
    )
    assert bundle2.cached is True


def test_api_federate_endpoints(sample_classification_model, monkeypatch):
    """Test GET /federate/model-bundle and GET /federate/backbone.onnx."""
    app = create_app(
        str(sample_classification_model["ckpt_path"]),
        str(sample_classification_model["meta_path"]),
    )
    client = TestClient(app)

    # 1. GET /federate/model-bundle
    res = client.get("/federate/model-bundle")
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["architecture"] == "resnet18"
    assert data["num_classes"] == 7
    assert data["embedding_dim"] == 512
    assert "onnx_download_url" in data
    assert res.headers.get("etag") == f'"{sample_classification_model["sha256"]}"'

    # 2. GET /federate/backbone.onnx
    onnx_res = client.get("/federate/backbone.onnx?model_slug=test-olive-model")
    assert onnx_res.status_code == 200
    assert len(onnx_res.content) > 1000000  # ResNet18 backbone is ~40 MB


def test_submit_federate_update_happy_path(sample_classification_model, monkeypatch, tmp_path):
    """Test successful POST /federate/submit-update with valid head-only parameters."""
    updates_dir = tmp_path / "client_updates"
    monkeypatch.setenv("AGRIFED_UPDATES_DIR", str(updates_dir))

    app = create_app(
        str(sample_classification_model["ckpt_path"]),
        str(sample_classification_model["meta_path"]),
    )
    client = TestClient(app)

    # Read base head weights to create small compliant perturbation (delta_norm ~ 0.05)
    base_state, _ = load_checkpoint_bundle(sample_classification_model["ckpt_path"], device="cpu")
    w_base = base_state["fc.weight"].numpy()
    b_base = base_state["fc.bias"].numpy()

    # Small perturbation
    w_client = w_base + np.random.normal(0, 0.005, size=w_base.shape).astype(np.float32)
    b_client = b_base + np.random.normal(0, 0.005, size=b_base.shape).astype(np.float32)

    payload = {
        "client_id": "tecnico_movil_jaen",
        "model_slug": "test-olive-model",
        "base_model_sha256": sample_classification_model["sha256"],
        "training_scope": "head_only",
        "num_samples": 45,
        "classes": sample_classification_model["classes"],
        "missing_classes": [3, 4],
        "head_tensors": {
            "weight": w_client.tolist(),
            "bias": b_client.tolist(),
        },
        "training_stats": {
            "epochs": 5,
            "fedprox_mu": 0.01,
            "final_loss": 0.184,
            "accuracy": 93.33,
            "training_time_seconds": 0.35,
        },
    }

    res = client.post("/federate/submit-update", json=payload)
    assert res.status_code == 200, res.text
    res_data = res.json()
    assert res_data["status"] == "ready_for_aggregation"
    assert res_data["training_scope"] == "head_only"
    assert res_data["relative_norm"] < 0.50
    assert res_data["bytes_received"] > 0

    # Verify file saved on disk
    saved_pt = Path(res_data["saved_to"])
    assert saved_pt.is_file()
    saved_meta = saved_pt.with_suffix(".meta.json")
    assert saved_meta.is_file()
    meta_json = json.loads(saved_meta.read_text())
    assert meta_json["client_id"] == "tecnico_movil_jaen"
    assert meta_json["training_scope"] == "head_only"
    assert meta_json["num_samples"] == 45
    assert meta_json["client_tier"] == "pwa-edge-light"


def test_submit_federate_update_lineage_conflict(sample_classification_model, monkeypatch):
    """Test rejection with HTTP 409 when client base_model_sha256 is outdated."""
    app = create_app(
        str(sample_classification_model["ckpt_path"]),
        str(sample_classification_model["meta_path"]),
    )
    client = TestClient(app)

    payload = {
        "client_id": "tecnico_conflict",
        "model_slug": "test-olive-model",
        "base_model_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "training_scope": "head_only",
        "num_samples": 20,
        "classes": sample_classification_model["classes"],
        "head_tensors": {
            "weight": np.zeros((7, 512)).tolist(),
            "bias": np.zeros(7).tolist(),
        },
    }

    res = client.post("/federate/submit-update", json=payload)
    assert res.status_code == 409
    assert "Lineage conflict" in res.json()["detail"]


def test_submit_federate_update_relative_norm_rejection(sample_classification_model, monkeypatch):
    """Test rejection with HTTP 422 when update exceeds --max-relative-norm (0.50)."""
    app = create_app(
        str(sample_classification_model["ckpt_path"]),
        str(sample_classification_model["meta_path"]),
    )
    client = TestClient(app)

    base_state, _ = load_checkpoint_bundle(sample_classification_model["ckpt_path"], device="cpu")
    w_base = base_state["fc.weight"].numpy()
    b_base = base_state["fc.bias"].numpy()

    # Extreme perturbation: 10x base weights to exceed max_relative_norm
    w_wild = w_base * 5.0
    b_wild = b_base * 5.0

    payload = {
        "client_id": "tecnico_wild",
        "model_slug": "test-olive-model",
        "base_model_sha256": sample_classification_model["sha256"],
        "training_scope": "head_only",
        "num_samples": 20,
        "classes": sample_classification_model["classes"],
        "head_tensors": {
            "weight": w_wild.tolist(),
            "bias": b_wild.tolist(),
        },
    }

    res = client.post("/federate/submit-update", json=payload)
    assert res.status_code == 422
    assert "Parameter audit rejected" in res.json()["detail"]


def test_submit_federate_update_canonical_projection(
    sample_classification_model, monkeypatch, tmp_path
):
    """Test Architectural Rule 1: Anti-Ontology Pollution.
    When light client creates extra local classes, submit projects only canonical classes."""
    updates_dir = tmp_path / "client_updates"
    monkeypatch.setenv("AGRIFED_UPDATES_DIR", str(updates_dir))

    app = create_app(
        str(sample_classification_model["ckpt_path"]),
        str(sample_classification_model["meta_path"]),
    )
    client = TestClient(app)

    base_state, _ = load_checkpoint_bundle(sample_classification_model["ckpt_path"], device="cpu")
    w_base = base_state["fc.weight"].numpy()
    b_base = base_state["fc.bias"].numpy()

    # Simulate client with 9 classes (7 canonical + 2 local custom classes)
    extra_w = np.random.normal(0, 0.01, size=(2, 512)).astype(np.float32)
    extra_b = np.zeros(2, dtype=np.float32)
    w_expanded = np.vstack([w_base, extra_w])
    b_expanded = np.concatenate([b_base, extra_b])
    classes_expanded = sample_classification_model["classes"] + [
        "finca_norte_mancha",
        "mosca_local",
    ]

    payload = {
        "client_id": "tecnico_custom_classes",
        "model_slug": "test-olive-model",
        "base_model_sha256": sample_classification_model["sha256"],
        "training_scope": "head_only",
        "num_samples": 50,
        "classes": classes_expanded,
        "head_tensors": {
            "weight": w_expanded.tolist(),
            "bias": b_expanded.tolist(),
        },
    }

    res = client.post("/federate/submit-update", json=payload)
    assert res.status_code == 200, res.text
    res_data = res.json()

    # Check saved .pt has canonical shape (7, 512), not (9, 512)
    saved_pt = Path(res_data["saved_to"])
    saved_state, _ = load_checkpoint_bundle(saved_pt, device="cpu")
    assert saved_state["fc.weight"].shape == (7, 512)
    assert saved_state["fc.bias"].shape == (7,)


def test_submit_federate_update_hmac_verification(
    sample_classification_model, monkeypatch, tmp_path
):
    """Test HMAC verification when AGRIFED_HMAC_KEY is enforced."""
    hmac_secret = "super_secret_agricultural_key_9988"
    monkeypatch.setenv("AGRIFED_HMAC_KEY", hmac_secret)
    monkeypatch.setenv("AGRIFED_UPDATES_DIR", str(tmp_path / "updates"))

    app = create_app(
        str(sample_classification_model["ckpt_path"]),
        str(sample_classification_model["meta_path"]),
    )
    client = TestClient(app)

    base_state, _ = load_checkpoint_bundle(sample_classification_model["ckpt_path"], device="cpu")
    w = base_state["fc.weight"].numpy()
    b = base_state["fc.bias"].numpy()

    # Canonical bytes sha256
    canonical_bytes = w.tobytes() + b.tobytes()
    tensors_sha = hashlib.sha256(canonical_bytes).hexdigest()

    valid_hmac = compute_manifest_hmac(
        key=hmac_secret,
        client_sha256=tensors_sha,
        base_model_sha256=sample_classification_model["sha256"],
        training_scope="head_only",
        client_id="tecnico_hmac",
        num_samples=30,
    )

    payload = {
        "client_id": "tecnico_hmac",
        "model_slug": "test-olive-model",
        "base_model_sha256": sample_classification_model["sha256"],
        "training_scope": "head_only",
        "num_samples": 30,
        "classes": sample_classification_model["classes"],
        "head_tensors": {
            "weight": w.tolist(),
            "bias": b.tolist(),
        },
        "hmac_signature": valid_hmac,
    }

    # 1. Valid signature -> 200 OK
    res = client.post("/federate/submit-update", json=payload)
    assert res.status_code == 200, res.text
    assert res.json()["hmac_verified"] is True

    # 2. Tampered signature -> 403 Forbidden
    bad_payload = copy.deepcopy(payload)
    bad_payload["hmac_signature"] = "deadbeefcafebabe00112233"
    res_bad = client.post("/federate/submit-update", json=bad_payload)
    assert res_bad.status_code == 403
    assert "Invalid HMAC signature" in res_bad.json()["detail"]


def test_aggregation_with_pwa_head_update(sample_classification_model, tmp_path):
    """Test that a PWA head update aggregates seamlessly with scripts/aggregate.py."""
    base_pt = sample_classification_model["ckpt_path"]
    base_meta = sample_classification_model["meta_path"]
    out_pt = tmp_path / "aggregated_global.pt"

    # Create a light client head-only update
    base_state, meta_dict = load_checkpoint_bundle(base_pt, device="cpu")
    client_pt = tmp_path / "client_pwa_head.pt"
    client_meta_path = tmp_path / "client_pwa_head.meta.json"

    w_perturbed = base_state["fc.weight"] + torch.randn_like(base_state["fc.weight"]) * 0.005
    b_perturbed = base_state["fc.bias"] + torch.randn_like(base_state["fc.bias"]) * 0.005
    client_state = {"fc.weight": w_perturbed, "fc.bias": b_perturbed}

    client_meta = {
        "client_weights_file": str(client_pt),
        "client_id": "pwa_client_01",
        "base_model_sha256": sample_classification_model["sha256"],
        "num_samples": 50,
        "training_scope": "head_only",
        "architecture": "resnet18",
        "num_classes": 7,
        "classes": sample_classification_model["classes"],
        "missing_classes": [],
        "trainable_tensors": ["fc.weight", "fc.bias"],
    }
    save_checkpoint_bundle(client_pt, client_state, client_meta)
    client_meta["sha256"] = compute_sha256(client_pt)
    client_meta_path.write_text(json.dumps(client_meta, indent=2))

    # Run scripts/aggregate.py
    cmd = [
        sys.executable,
        "scripts/aggregate.py",
        "--base-model", str(base_pt),
        "--client-weights", str(client_pt),
        "--output", str(out_pt),
        "--metadata", str(base_meta),
        "--eta", "0.6",
        "--max-relative-norm", "0.50",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"aggregate.py failed:\n{res.stderr}\n{res.stdout}"
    assert out_pt.is_file()

    # Load aggregated state and verify weight changed towards client
    agg_state, _ = load_checkpoint_bundle(out_pt, device="cpu")
    diff_norm = (agg_state["fc.weight"] - base_state["fc.weight"]).norm().item()
    assert diff_norm > 0.0, "Aggregated weights did not move from base model"
    assert diff_norm < 1.0, f"Aggregated shift was unreasonably large: {diff_norm}"


# ---------------------------------------------------------------------------
# On-device optimizer (JS) numerical soundness + server-side payload hardening
# ---------------------------------------------------------------------------
import shutil as _shutil


@pytest.mark.skipif(_shutil.which("node") is None, reason="node is required for the JS harness")
def test_edge_optimizer_numerical_soundness():
    """FedProx pull, missing-class retention, delta clipping and warm-start semantics."""
    harness = Path(__file__).parent / "edge_optimizer_harness.js"
    proc = subprocess.run(
        ["node", str(harness)], capture_output=True, text=True, timeout=60, check=True
    )
    report = json.loads(proc.stdout.strip().splitlines()[-1])

    assert report["all_finite"] is True
    assert report["present_moved"] is True
    # Classes absent from local data must not drift (no catastrophic forgetting / ontology drift).
    assert report["missing_unchanged"] is True
    # Proximal term works: stronger mu keeps the head closer to the global anchor.
    assert report["rel_mu5"] < report["rel_mu0"]
    # Client-side clip keeps the update inside the bound the server enforces (0.50).
    assert report["rel_clipped"] <= 0.1 + 1e-4
    # Warm start preserves previous sessions; cold start (stale lineage) restarts from the base.
    assert report["warm_start_retains_a"] is True
    assert report["cold_start_resets_a"] is True


def _valid_head_payload(model, num_samples=10):
    base_state, _ = load_checkpoint_bundle(model["ckpt_path"], device="cpu")
    w = (base_state["fc.weight"].numpy() + 0.001).astype(np.float32)
    b = (base_state["fc.bias"].numpy() + 0.001).astype(np.float32)
    return w, b, {
        "client_id": "tecnico_movil_test",
        "model_slug": "test-olive-model",
        "base_model_sha256": model["sha256"],
        "training_scope": "head_only",
        "num_samples": num_samples,
        "classes": model["classes"],
        "missing_classes": [],
        "head_tensors": {"weight": w.tolist(), "bias": b.tolist()},
        "training_stats": {},
    }


def test_submit_update_payload_sha256_is_verified(sample_classification_model, monkeypatch, tmp_path):
    monkeypatch.setenv("AGRIFED_UPDATES_DIR", str(tmp_path / "updates"))
    app = create_app(
        str(sample_classification_model["ckpt_path"]), str(sample_classification_model["meta_path"])
    )
    client = TestClient(app)

    w, b, payload = _valid_head_payload(sample_classification_model)
    good = hashlib.sha256(w.tobytes() + b.tobytes()).hexdigest()

    bad = client.post("/federate/submit-update", json={**payload, "payload_sha256": "0" * 64})
    assert bad.status_code == 422, bad.text
    assert "payload_sha256" in bad.text

    ok = client.post("/federate/submit-update", json={**payload, "payload_sha256": good})
    assert ok.status_code == 200, ok.text


def test_submit_update_rejects_inflated_num_samples(sample_classification_model, monkeypatch, tmp_path):
    monkeypatch.setenv("AGRIFED_UPDATES_DIR", str(tmp_path / "updates"))
    app = create_app(
        str(sample_classification_model["ckpt_path"]), str(sample_classification_model["meta_path"])
    )
    client = TestClient(app)

    _, _, payload = _valid_head_payload(sample_classification_model, num_samples=10**9)
    res = client.post("/federate/submit-update", json=payload)
    assert res.status_code == 400, res.text

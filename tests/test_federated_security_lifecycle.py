"""Integration and unit tests for Secure Federated Learning & MLOps Lifecycle:
- Audit of client updates (eligibility, lineage, integrity, non-finite values, HMAC)
- Safe candidate isolation (non-destructive aggregation)
- Quality guardrails & validation dataset compatibility
- Staged candidate promotion and rollback snapshots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from starlette.testclient import TestClient

from agrifed_khaos.catalog import ModelCatalogManager
from agrifed_khaos.crypto import compute_manifest_hmac, compute_sha256
from agrifed_khaos.federated_ops import (
    audit_client_update,
    resolve_validation_directory_for_model,
)
from agrifed_khaos.models import build_model, save_checkpoint_bundle
from scripts.serve import create_app


@pytest.fixture
def fl_test_environment(tmp_path: Path) -> dict[str, Any]:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create Base Model
    base_model = build_model("resnet18", num_classes=3, pretrained=False)
    base_path = models_dir / "base_model.pt"
    bundle_meta = {
        "architecture": "resnet18",
        "num_classes": 3,
        "classes": ["sana", "roña", "mildiu"],
        "img_size": [64, 64],
        "temperature": 1.0,
    }
    save_checkpoint_bundle(base_path, base_model.state_dict(), bundle_meta)
    base_sha = compute_sha256(base_path)

    # 2. Create Active Global Model
    active_path = models_dir / "global_model.pt"
    save_checkpoint_bundle(active_path, base_model.state_dict(), bundle_meta)
    active_sha = compute_sha256(active_path)

    # 3. Create Catalog JSON
    catalog_path = models_dir / "catalog.json"
    catalog_data = {
        "models": [
            {
                "slug": "test-foliage-model",
                "title": "Test Foliage Model",
                "domain": "Agricultura",
                "active_checkpoint": "global_model.pt",
                "versions": ["base_model.pt", "global_model.pt"],
                "version_history": [
                    {
                        "version_id": "v1",
                        "checkpoint": "base_model.pt",
                        "origin": "train_base",
                        "status": "archived",
                    },
                    {
                        "version_id": "v2",
                        "checkpoint": "global_model.pt",
                        "origin": "federated_aggregation",
                        "status": "active",
                    },
                ],
                "architecture": "resnet18",
                "num_classes": 3,
                "classes": ["sana", "roña", "mildiu"],
            }
        ]
    }
    catalog_path.write_text(json.dumps(catalog_data, indent=2), encoding="utf-8")

    # 4. Updates directory
    updates_dir = models_dir / "client_updates"
    updates_dir.mkdir(parents=True, exist_ok=True)

    # 4.a Valid client update matching active model
    valid_update_path = updates_dir / "client_valid_parameters.pt"
    fc_weights = {"fc.weight": torch.randn(3, 512), "fc.bias": torch.randn(3)}
    save_checkpoint_bundle(valid_update_path, fc_weights, {"classes": ["sana", "roña", "mildiu"]})
    valid_sha = compute_sha256(valid_update_path)
    valid_meta_path = updates_dir / "client_valid_parameters.meta.json"
    valid_meta_path.write_text(
        json.dumps(
            {
                "client_id": "coop_test_valid",
                "sha256": valid_sha,
                "base_model_sha256": active_sha,
                "training_scope": "head_only",
                "num_samples": 250,
            }
        ),
        encoding="utf-8",
    )

    # 4.b Mock tiny stub (like coop_sevilla_parameters.pt 20 bytes)
    stub_path = updates_dir / "coop_sevilla_parameters.pt"
    stub_path.write_text("signed client update", encoding="utf-8")
    stub_meta = updates_dir / "coop_sevilla_parameters.meta.json"
    stub_meta.write_text(
        json.dumps(
            {
                "client_id": "coop_sevilla",
                "sha256": "0d8f1e74078c...",
                "base_model_sha256": "a" * 64,
                "num_samples": 12,
            }
        ),
        encoding="utf-8",
    )

    # 4.c Lineage mismatch update
    mismatch_path = updates_dir / "client_mismatch_parameters.pt"
    save_checkpoint_bundle(mismatch_path, fc_weights, {})
    mismatch_meta = updates_dir / "client_mismatch_parameters.meta.json"
    mismatch_meta.write_text(
        json.dumps(
            {
                "client_id": "coop_mismatch",
                "sha256": compute_sha256(mismatch_path),
                "base_model_sha256": "0" * 64,
                "training_scope": "head_only",
                "num_samples": 100,
            }
        ),
        encoding="utf-8",
    )

    # 4.d Non-finite (NaN) update
    nan_path = updates_dir / "client_nan_parameters.pt"
    nan_weights = {"fc.weight": torch.full((3, 512), float("nan")), "fc.bias": torch.zeros(3)}
    save_checkpoint_bundle(nan_path, nan_weights, {})
    nan_meta = updates_dir / "client_nan_parameters.meta.json"
    nan_meta.write_text(
        json.dumps(
            {
                "client_id": "coop_nan",
                "sha256": compute_sha256(nan_path),
                "base_model_sha256": active_sha,
                "training_scope": "head_only",
                "num_samples": 80,
            }
        ),
        encoding="utf-8",
    )

    app = create_app(model_path=str(active_path), metadata_path=str(models_dir / "metadata.json"))
    client = TestClient(app)

    return {
        "client": client,
        "tmp_path": tmp_path,
        "models_dir": models_dir,
        "active_path": active_path,
        "active_sha": active_sha,
        "base_path": base_path,
        "base_sha": base_sha,
        "valid_update_path": valid_update_path,
        "stub_path": stub_path,
        "mismatch_path": mismatch_path,
        "nan_path": nan_path,
    }


def test_audit_valid_update_is_eligible(fl_test_environment: dict[str, Any]) -> None:
    env = fl_test_environment
    res = audit_client_update(
        env["valid_update_path"],
        target_base_checkpoint=env["active_path"],
    )
    assert res.is_eligible is True
    assert res.lineage_matched is True
    assert res.all_finite is True
    assert len(res.rejection_reasons) == 0
    assert res.num_samples == 250


def test_audit_rejects_mock_stub_artifact(fl_test_environment: dict[str, Any]) -> None:
    env = fl_test_environment
    res = audit_client_update(
        env["stub_path"],
        target_base_checkpoint=env["active_path"],
    )
    assert res.is_eligible is False
    assert any("stub no reconocible" in r for r in res.rejection_reasons)


def test_audit_rejects_lineage_mismatch(fl_test_environment: dict[str, Any]) -> None:
    env = fl_test_environment
    res = audit_client_update(
        env["mismatch_path"],
        target_base_checkpoint=env["active_path"],
    )
    assert res.is_eligible is False
    assert res.lineage_matched is False
    assert any("Incompatibilidad de linaje" in r for r in res.rejection_reasons)


def test_audit_rejects_non_finite_tensor_values(fl_test_environment: dict[str, Any]) -> None:
    env = fl_test_environment
    res = audit_client_update(
        env["nan_path"],
        target_base_checkpoint=env["active_path"],
    )
    assert res.is_eligible is False
    assert res.all_finite is False
    assert any("valores numéricos no finitos" in r for r in res.rejection_reasons)


def test_audit_hmac_validation(fl_test_environment: dict[str, Any]) -> None:
    env = fl_test_environment
    secret_key = "super_secret_fl_key"

    # 1. Test missing HMAC signature when required
    res_no_sig = audit_client_update(
        env["valid_update_path"],
        target_base_checkpoint=env["active_path"],
        hmac_key=secret_key,
    )
    assert res_no_sig.is_eligible is False
    assert res_no_sig.hmac_valid is False
    assert any("no está firmado" in r for r in res_no_sig.rejection_reasons)

    # 2. Add valid signature
    valid_sig = compute_manifest_hmac(
        key=secret_key,
        client_id="coop_test_valid",
        client_sha256=compute_sha256(env["valid_update_path"]),
        base_model_sha256=env["active_sha"],
        training_scope="head_only",
        num_samples=250,
    )
    meta_p = env["valid_update_path"].with_suffix(".meta.json")
    mdata = json.loads(meta_p.read_text())
    mdata["hmac_signature"] = valid_sig
    meta_p.write_text(json.dumps(mdata))

    res_signed = audit_client_update(
        env["valid_update_path"],
        target_base_checkpoint=env["active_path"],
        hmac_key=secret_key,
    )
    assert res_signed.is_eligible is True
    assert res_signed.hmac_valid is True


def test_federated_updates_api_filters_eligible_vs_rejected(
    fl_test_environment: dict[str, Any],
) -> None:
    client: TestClient = fl_test_environment["client"]

    res = client.get("/api/master/federated/updates?target_model=test-foliage-model")
    assert res.status_code == 200
    data = res.json()

    assert data["target_model"] == "test-foliage-model"
    assert data["target_base_checkpoint"] == "global_model.pt"
    assert data["eligible_count"] >= 1
    assert data["rejected_count"] >= 2

    by_file = {u["filename"]: u for u in data["updates"]}
    assert by_file["client_valid_parameters.pt"]["is_eligible"] is True
    assert by_file["coop_sevilla_parameters.pt"]["is_eligible"] is False
    assert by_file["client_mismatch_parameters.pt"]["is_eligible"] is False
    assert by_file["client_nan_parameters.pt"]["is_eligible"] is False


def test_federated_aggregate_endpoint_security_rejection(
    fl_test_environment: dict[str, Any],
) -> None:
    client: TestClient = fl_test_environment["client"]

    # Attempt to aggregate the stub/ineligible update
    bad_payload = {
        "target_model_slug": "test-foliage-model",
        "selected_update_paths": [str(fl_test_environment["stub_path"])],
        "server_momentum": 0.9,
    }
    res = client.post("/api/master/federated/aggregate", json=bad_payload)
    assert res.status_code == 400
    assert "RECHAZO DE SEGURIDAD" in res.json()["detail"]


def test_candidate_promotion_and_rollback_lifecycle(
    fl_test_environment: dict[str, Any],
) -> None:
    client: TestClient = fl_test_environment["client"]
    models_dir: Path = fl_test_environment["models_dir"]

    # 1. Create candidate checkpoint file
    cand_path = models_dir / "test-foliage-model_candidate.pt"
    cand_model = build_model("resnet18", num_classes=3, pretrained=False)
    save_checkpoint_bundle(
        cand_path,
        cand_model.state_dict(),
        {"classes": ["sana", "roña", "mildiu"]},
    )

    # 2. Promote candidate
    res_promote = client.post("/api/master/models/test-foliage-model/promote-candidate?force=true")
    assert res_promote.status_code == 200
    promote_data = res_promote.json()
    assert promote_data["success"] is True
    assert promote_data["promoted_version"]["version_id"] == "v3"
    assert promote_data["promoted_version"]["status"] == "active"

    # Verify backup was created
    backup_file = models_dir / "global_model.pt.backup"
    assert backup_file.is_file()

    # 3. Rollback
    res_rollback = client.post("/api/master/models/test-foliage-model/rollback")
    assert res_rollback.status_code == 200
    rollback_data = res_rollback.json()
    assert rollback_data["success"] is True

    # Check catalog shows reverted active checkpoint
    cm = ModelCatalogManager(catalog_path=models_dir / "catalog.json", models_dir=models_dir)
    entry = cm.get_by_slug("test-foliage-model")
    assert entry is not None
    # Previous active was v2 (global_model.pt)
    assert entry.active_checkpoint == "global_model.pt"


def test_validation_directory_compatibility_resolver(tmp_path: Path) -> None:
    # 1. Matching val directory
    val_dir = tmp_path / "valid_val"
    for c in ["sana", "roña", "mildiu"]:
        (val_dir / c).mkdir(parents=True, exist_ok=True)

    resolved, status = resolve_validation_directory_for_model(
        classes=["sana", "roña", "mildiu"],
        explicit_val_dir=str(val_dir),
    )
    assert status == "MATCH_EXACT"
    assert resolved == val_dir

    # 2. Mismatched directory (e.g. avocado classes on foliage model)
    avocado_dir = tmp_path / "avocado_val"
    for c in ["optimo", "defecto_menor", "descarte"]:
        (avocado_dir / c).mkdir(parents=True, exist_ok=True)

    resolved_bad, status_bad = resolve_validation_directory_for_model(
        classes=["sana", "roña", "mildiu"],
        explicit_val_dir=str(avocado_dir),
    )
    assert status_bad == "INCOMPATIBLE_OR_MISSING"
    assert resolved_bad is None

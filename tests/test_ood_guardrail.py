import io
import json

import torch
from fastapi.testclient import TestClient
from PIL import Image

from agrifed_khaos.models import build_model
from scripts.serve import compute_ood_guardrail, create_app


def test_compute_ood_guardrail_confident_in_distribution() -> None:
    # Confident, high-activation logits
    logits = torch.tensor([[10.0, 1.0, 0.5]], dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)

    metrics = compute_ood_guardrail(
        logits=logits,
        probs=probs,
        num_classes=3,
        temperature=1.0,
        threshold=2.0,
    )

    assert metrics["ood_flag"] is False
    assert metrics["energy"] < -5.0  # Strongly negative Free Energy
    assert metrics["energy_score"] > 5.0
    assert metrics["ood_score"] < 0.5


def test_compute_ood_guardrail_anomalous_low_energy() -> None:
    # Low magnitude logits indicating weak activation across all classes (e.g. out of domain)
    logits = torch.tensor([[0.5, -0.2, 0.1]], dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)

    metrics = compute_ood_guardrail(
        logits=logits,
        probs=probs,
        num_classes=3,
        temperature=1.0,
        threshold=2.0,
    )

    assert metrics["ood_flag"] is True
    assert metrics["energy_score"] < 2.0
    assert metrics["ood_score"] > 0.0


def test_compute_ood_guardrail_high_entropy() -> None:
    # High entropy / flat uncertain distribution
    logits = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)

    metrics = compute_ood_guardrail(
        logits=logits,
        probs=probs,
        num_classes=3,
        temperature=1.0,
        threshold=1.0,  # energy_score is ~3.1, above threshold
        entropy_threshold=0.65,
    )

    # Triggered due to high entropy
    assert metrics["ood_flag"] is True
    assert metrics["normalized_entropy"] > 0.90


def test_compute_ood_guardrail_fallback_without_logits() -> None:
    # When logits are None, it gracefully approximates from probabilities
    probs = torch.tensor([0.9, 0.05, 0.05], dtype=torch.float32)
    metrics = compute_ood_guardrail(
        logits=None,
        probs=probs,
        num_classes=3,
        temperature=1.0,
    )

    assert "ood_flag" in metrics
    assert "ood_score" in metrics
    assert "energy" in metrics


def test_predict_endpoint_includes_ood_guardrail(tmp_path) -> None:
    # Setup test model and metadata
    base_file = tmp_path / "model.pt"
    m = build_model("resnet18", num_classes=2)
    torch.save(m.state_dict(), str(base_file))

    meta_file = tmp_path / "metadata.json"
    meta_file.write_text(
        json.dumps({
            "classes": ["clase_a", "clase_b"],
            "architecture": "resnet18",
            "ood_threshold": 3.0,
        }),
        encoding="utf-8",
    )

    app = create_app(str(base_file), str(meta_file))
    client = TestClient(app)

    # Create dummy test image
    im = Image.new("RGB", (224, 224), color=(120, 80, 40))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    buf.seek(0)

    res = client.post(
        "/predict?save_for_training=false",
        files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")},
    )

    assert res.status_code == 200
    data = res.json()
    assert "ood_flag" in data
    assert isinstance(data["ood_flag"], bool)
    assert "ood_score" in data
    assert isinstance(data["ood_score"], float)
    assert "energy" in data
    assert "free_energy" in data
    assert "entropy" in data


def test_dataset_deduplicate_endpoint(tmp_path) -> None:
    base_file = tmp_path / "model.pt"
    m = build_model("resnet18", num_classes=2)
    torch.save(m.state_dict(), str(base_file))
    meta_file = tmp_path / "metadata.json"
    meta_file.write_text(
        json.dumps({"classes": ["healthy", "diseased"], "architecture": "resnet18"}),
        encoding="utf-8",
    )

    # Build temporary dataset with exact MD5 duplicates and a non-image file
    ds_dir = tmp_path / "dataset" / "train"
    cls_a = ds_dir / "healthy"
    cls_b = ds_dir / "diseased"
    cls_a.mkdir(parents=True)
    cls_b.mkdir(parents=True)

    im1 = Image.new("RGB", (64, 64), color=(10, 200, 10))
    b1 = io.BytesIO()
    im1.save(b1, format="PNG")
    raw_bytes_1 = b1.getvalue()

    im2 = Image.new("RGB", (64, 64), color=(200, 10, 10))
    b2 = io.BytesIO()
    im2.save(b2, format="PNG")
    raw_bytes_2 = b2.getvalue()

    # Write 1 original + 2 exact duplicates of image 1, plus 1 unique image 2
    (cls_a / "img1_orig.png").write_bytes(raw_bytes_1)
    (cls_a / "img1_dup1.png").write_bytes(raw_bytes_1)
    (cls_b / "img1_dup2_cross_class.png").write_bytes(raw_bytes_1)
    (cls_b / "img2_unique.png").write_bytes(raw_bytes_2)

    # Non-image file that must never be deleted even if identical
    notes_file = cls_a / "notes.txt"
    notes_file.write_text("metadata keep me", encoding="utf-8")
    notes_dup = cls_b / "notes_copy.txt"
    notes_dup.write_text("metadata keep me", encoding="utf-8")

    app = create_app(str(base_file), str(meta_file))
    client = TestClient(app)

    res = client.post(
        "/api/master/datasets/deduplicate",
        json={"dataset_path": str(ds_dir)},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["deleted_count"] == 2
    assert payload["total_images"] == 2
    assert payload["health_profile"]["duplicates_count"] == 0
    assert payload["health_profile"]["health_score"] == 100
    assert notes_file.exists()
    assert notes_dup.exists()


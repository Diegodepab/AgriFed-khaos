"""Tests for Advanced Dataset Profiler and Train/Val/Test Splits."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agrifed_khaos.catalog import LogicalModelEntry, ModelCatalogManager
from agrifed_khaos.models import build_model, save_checkpoint_bundle
from scripts.serve import create_app

DUMMY_1X1_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff"
    b"?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    active_path = models_dir / "global_model.pt"
    m = build_model("resnet18", num_classes=2, pretrained=False)
    save_checkpoint_bundle(active_path, m.state_dict(), {"classes": ["sana", "dano"]})

    meta_path = models_dir / "metadata.json"
    meta_path.write_text('{"classes": ["sana", "dano"], "arch": "resnet18"}')

    catalog_path = models_dir / "catalog.json"
    cm = ModelCatalogManager(catalog_path=catalog_path, models_dir=models_dir)
    entry = LogicalModelEntry(
        slug="test-model",
        title="Modelo Test",
        domain="Agricultura",
        active_checkpoint="global_model.pt",
        versions=["global_model.pt"],
        classes=["sana", "dano"],
    )
    cm.save_catalog([entry])

    app = create_app(str(active_path), str(meta_path))
    app.state.catalog = cm
    return TestClient(app)


def test_scenario_1_flat_dataset(tmp_path: Path, client: TestClient):
    """Scenario 1: Flat dataset where classes are direct subfolders."""
    ds_dir = tmp_path / "flat_dataset"
    (ds_dir / "Sano").mkdir(parents=True)
    (ds_dir / "Repilo").mkdir(parents=True)
    (ds_dir / "Sano" / "img1.png").write_bytes(DUMMY_1X1_PNG)
    (ds_dir / "Sano" / "img2.png").write_bytes(DUMMY_1X1_PNG)
    (ds_dir / "Repilo" / "img3.png").write_bytes(DUMMY_1X1_PNG)

    res = client.post("/api/master/datasets/inspect", json={"dataset_path": str(ds_dir)})
    assert res.status_code == 200
    data = res.json()

    assert data["valid"] is True
    assert data["is_pre_split"] is False
    assert data["split_distribution"] is None
    assert data["unlabeled_count"] == 0
    assert sorted(data["classes"]) == ["Repilo", "Sano"]
    assert data["total_images"] == 3


def test_scenario_2_presplit_dataset(tmp_path: Path, client: TestClient):
    """Scenario 2: Dataset with train/val/test split directories."""
    ds_dir = tmp_path / "presplit_dataset"
    (ds_dir / "train" / "Sano").mkdir(parents=True)
    (ds_dir / "train" / "Repilo").mkdir(parents=True)
    (ds_dir / "val" / "Sano").mkdir(parents=True)
    (ds_dir / "test" / "Repilo").mkdir(parents=True)

    # 4 in train (3 Sano, 1 Repilo), 2 in val, 1 in test
    for i in range(3):
        (ds_dir / "train" / "Sano" / f"t_{i}.png").write_bytes(DUMMY_1X1_PNG)
    (ds_dir / "train" / "Repilo" / "t_rep.png").write_bytes(DUMMY_1X1_PNG)
    for i in range(2):
        (ds_dir / "val" / "Sano" / f"v_{i}.png").write_bytes(DUMMY_1X1_PNG)
    (ds_dir / "test" / "Repilo" / "test_0.png").write_bytes(DUMMY_1X1_PNG)

    res = client.post("/api/master/datasets/inspect", json={"dataset_path": str(ds_dir)})
    assert res.status_code == 200
    data = res.json()

    assert data["valid"] is True
    assert data["is_pre_split"] is True
    assert data["split_distribution"] == {"train": 4, "val": 2, "test": 1}
    assert data["split_percentages"]["train"] == pytest.approx(57.1, rel=1e-1)
    assert data["split_percentages"]["val"] == pytest.approx(28.6, rel=1e-1)
    assert data["split_percentages"]["test"] == pytest.approx(14.3, rel=1e-1)
    assert data["unlabeled_count"] == 0


def test_scenario_3_unlabeled_data(tmp_path: Path, client: TestClient):
    """Scenario 3: Dataset with root-level and explicit unlabeled folder images."""
    ds_dir = tmp_path / "unlabeled_dataset"
    (ds_dir / "Sano").mkdir(parents=True)
    (ds_dir / "Repilo").mkdir(parents=True)
    (ds_dir / "unlabeled").mkdir(parents=True)

    (ds_dir / "Sano" / "img1.png").write_bytes(DUMMY_1X1_PNG)
    (ds_dir / "Repilo" / "img2.png").write_bytes(DUMMY_1X1_PNG)

    # 2 root-level images
    (ds_dir / "loose_1.png").write_bytes(DUMMY_1X1_PNG)
    (ds_dir / "loose_2.png").write_bytes(DUMMY_1X1_PNG)

    # 3 images inside unlabeled/ folder
    for i in range(3):
        (ds_dir / "unlabeled" / f"raw_{i}.png").write_bytes(DUMMY_1X1_PNG)

    res = client.post("/api/master/datasets/inspect", json={"dataset_path": str(ds_dir)})
    assert res.status_code == 200
    data = res.json()

    assert data["valid"] is True
    assert data["is_pre_split"] is False
    assert data["unlabeled_count"] == 5
    # The 'unlabeled' folder should NOT be treated as a disease/crop class!
    assert "unlabeled" not in data["classes"]
    assert sorted(data["classes"]) == ["Repilo", "Sano"]
    assert data["total_images"] == 2


def test_train_base_split_execution(tmp_path: Path):
    """Test running train_base.py with flat dataset and train/val/test splits."""
    import subprocess
    import sys

    ds_dir = tmp_path / "flat_split_dataset"
    (ds_dir / "class_a").mkdir(parents=True)
    (ds_dir / "class_b").mkdir(parents=True)
    for i in range(10):
        (ds_dir / "class_a" / f"a_{i}.png").write_bytes(DUMMY_1X1_PNG)
        (ds_dir / "class_b" / f"b_{i}.png").write_bytes(DUMMY_1X1_PNG)

    out_weights = tmp_path / "out_model.pt"
    out_meta = tmp_path / "out_meta.json"

    cmd = [
        sys.executable,
        "scripts/train_base.py",
        "--data-dir",
        str(ds_dir),
        "--output",
        str(out_weights),
        "--metadata",
        str(out_meta),
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--train-split",
        "0.8",
        "--val-split",
        "0.1",
        "--test-split",
        "0.1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "[*] Dataset split applied: Train=16" in res.stdout
    assert "Val=2" in res.stdout
    assert "Test=2" in res.stdout
    assert out_weights.is_file()
    assert out_meta.is_file()
    import json
    meta = json.loads(out_meta.read_text(encoding="utf-8"))
    assert meta["dataset_splits"]["train"] == 0.8
    assert meta["dataset_splits"]["counts"]["train"] == 16
    assert meta["dataset_splits"]["counts"]["val"] == 2
    assert meta["dataset_splits"]["counts"]["test"] == 2

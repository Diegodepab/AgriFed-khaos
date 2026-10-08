from __future__ import annotations

from pathlib import Path

from agrifed_khaos.config import AppConfig, DatasetConfig


def test_dataset_config_resolves_annotations(tmp_path: Path) -> None:
    root = tmp_path / "EnfermedadesHoja_v1.0"
    root.mkdir()
    config = DatasetConfig(root=root)

    assert config.annotations_csv == root / "annotations" / "labels.csv"
    assert config.train_path == (root / "dataset" / "train").resolve()


def test_app_config_validates_nested_models(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    config = AppConfig.model_validate({"dataset": {"root": str(root)}})

    assert config.dataset.root == root
    assert config.federation.rounds == 10

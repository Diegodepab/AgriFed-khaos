"""Command-line interface for dataset audit and configuration validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from .config import AppConfig, DatasetConfig, RuntimeSettings
from .data import load_inventory, load_records, sample_path
from .quality import audit_split_integrity


def load_config(config_path: Path) -> AppConfig:
    """Load YAML config and validate it strictly with Pydantic."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return AppConfig.model_validate(raw)


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(prog="agrifed-audit")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Override dataset root")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable output",
    )
    return parser


def main() -> None:
    """Execute configuration and pre-FL dataset split audit."""
    parser = build_parser()
    args = parser.parse_args()

    settings = RuntimeSettings()
    config_path = args.config or settings.config_path
    if config_path.is_file():
        config = load_config(config_path)
    else:
        config = AppConfig(dataset=DatasetConfig(root=args.dataset_root or Path(".")))

    if args.dataset_root is not None:
        config.dataset = DatasetConfig(
            root=args.dataset_root,
            annotations_csv=args.dataset_root / "annotations" / "labels.csv",
            allow_test_as_client=config.dataset.allow_test_as_client,
            num_clients=config.dataset.num_clients,
            client_split_seed=config.dataset.client_split_seed,
        )

    inventory = load_inventory(config.dataset.root, config.dataset.annotations_csv)
    records = load_records(inventory.annotations_csv)

    train_paths = [sample_path(config.dataset.root, r) for r in records if r.split == "train"]
    val_paths = [sample_path(config.dataset.root, r) for r in records if r.split == "val"]
    test_paths = [sample_path(config.dataset.root, r) for r in records if r.split == "test"]

    integrity_report = audit_split_integrity(train_paths, val_paths, test_paths)

    output = {
        "dataset_root": str(config.dataset.root),
        "total_records": inventory.total_records,
        "splits": {s.split: s.count for s in inventory.split_summaries},
        "zero_leakage_valid": integrity_report.valid,
        "integrity_issues": integrity_report.issues,
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print(f"Dataset root: {config.dataset.root}")
    print(f"Total records: {inventory.total_records}")
    print(f"Split counts: {output['splits']}")
    print(f"Zero-leakage split integrity: {integrity_report.valid}")
    if integrity_report.issues:
        print("Integrity issues:")
        for issue in integrity_report.issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    main()

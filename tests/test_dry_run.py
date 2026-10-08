"""Dry-run instantiation test for vision and federated learning backends.

Verifies:
1. PyTorch & Ultralytics linkage through YOLOv8Wrapper.
2. Flower & Parameter Gate linkage through SanitizedFedProx.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from agrifed_khaos.federated import SanitizedFedProx
from agrifed_khaos.wrappers import ParameterValidator, YOLOv8Wrapper


def test_dry_run_yolov8_and_sanitized_fedprox() -> None:
    # Ensure ultralytics writes settings to tmp if not configured
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    pytest.importorskip("flwr")

    # 1. Instantiate YOLOv8Wrapper (tests PyTorch & Ultralytics integration)
    validator = ParameterValidator(max_update_norm=100.0, enforce_finite=True)
    yolo_wrapper = YOLOv8Wrapper.create(
        model_cfg_or_weights="yolov8n.yaml",
        device="cpu",
        validator=validator,
    )
    assert yolo_wrapper is not None

    parameters = yolo_wrapper.get_parameters()
    assert isinstance(parameters, list)
    assert len(parameters) > 0
    assert all(isinstance(p, np.ndarray) for p in parameters)
    assert all(np.all(np.isfinite(p)) for p in parameters)

    # 2. Instantiate SanitizedFedProx (tests Flower & Parameter Gate integration)
    strategy = SanitizedFedProx(
        validator=validator,
        global_model_wrapper=yolo_wrapper,
        proximal_mu=0.01,
        min_fit_clients=2,
        min_available_clients=2,
    )
    assert strategy is not None
    assert strategy.validator is validator
    assert strategy.global_model_wrapper is yolo_wrapper
    assert strategy.rejection_history == []

    print(
        f"\n[DRY RUN SUCCESS]: YOLOv8 parameters={len(parameters)}, "
        f"Flower strategy={strategy.__class__.__name__} linked."
    )


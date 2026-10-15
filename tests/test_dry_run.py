"""Dry-run instantiation test for vision and federated learning backends.

Verifies:
1. PyTorch & Timm vision model linkage through TorchModuleWrapper.
2. Flower & Parameter Gate linkage through SanitizedFedProx.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from agrifed_khaos.federated import SanitizedFedProx
from agrifed_khaos.models import build_model
from agrifed_khaos.wrappers import ParameterValidator, TorchModuleWrapper


def test_dry_run_vision_and_sanitized_fedprox() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("flwr")

    # 1. Instantiate Vision Backbone through TorchModuleWrapper
    validator = ParameterValidator(max_update_norm=100.0, enforce_finite=True)
    model = build_model("resnet18", num_classes=2, pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    wrapper = TorchModuleWrapper(
        module=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        validator=validator,
    )
    assert wrapper is not None

    parameters = wrapper.get_parameters()
    assert isinstance(parameters, list)
    assert len(parameters) > 0
    assert all(isinstance(p, np.ndarray) for p in parameters)
    assert all(np.all(np.isfinite(p)) for p in parameters)

    # 2. Instantiate SanitizedFedProx (tests Flower & Parameter Gate integration)
    strategy = SanitizedFedProx(
        validator=validator,
        global_model_wrapper=wrapper,
        proximal_mu=0.01,
        min_fit_clients=2,
        min_available_clients=2,
        server_momentum=0.9,
    )
    assert strategy is not None
    assert strategy.validator is validator
    assert strategy.global_model_wrapper is wrapper
    assert strategy.server_momentum == 0.9
    assert strategy.rejection_history == []

    print(
        f"\n[DRY RUN SUCCESS]: Vision parameters={len(parameters)}, "
        f"Flower strategy={strategy.__class__.__name__} linked."
    )

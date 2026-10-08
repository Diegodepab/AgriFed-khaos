from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from agrifed_khaos.wrappers import (
    ParameterValidator,
    TorchModuleWrapper,
)


def test_torch_wrapper_parameters_and_fedprox() -> None:
    torch = pytest.importorskip("torch")

    class ToyModel(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 2)
            self.bn = torch.nn.BatchNorm1d(2)

        def forward(self, x: Any) -> Any:
            return self.bn(self.fc1(x))

    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    validator = ParameterValidator(max_update_norm=50.0)

    wrapper = TorchModuleWrapper(
        module=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        validator=validator,
    )

    params = wrapper.get_parameters()
    # Ensure params only include trainable weights/biases
    assert len(params) == 4

    # Test set_parameters
    modified_params = [p + 0.1 for p in params]
    wrapper.set_parameters(modified_params)
    new_params = wrapper.get_parameters()
    assert np.allclose(new_params[0], modified_params[0])

    # Test local training with FedProx proximal term
    dummy_x = torch.randn(8, 4)
    dummy_y = torch.randn(8, 2)
    dataset = [(dummy_x, dummy_y)]

    stats = wrapper.train_local_epoch(
        train_data=dataset,
        epochs=2,
        prox_mu=0.1,
        reference_parameters=params,
    )
    assert stats.examples == 16
    assert stats.batches == 2
    assert stats.loss > 0.0

from __future__ import annotations

from pathlib import Path

import numpy as np

from agrifed_khaos.federated import ClientResult, aggregate_weighted_metrics
from agrifed_khaos.quality import audit_split_integrity
from agrifed_khaos.wrappers import ParameterValidator


def test_aggregate_weighted_metrics() -> None:
    result = aggregate_weighted_metrics(
        [
            ClientResult(num_examples=10, metrics={"loss": 1.0, "acc": 0.5}),
            ClientResult(num_examples=30, metrics={"loss": 0.5, "acc": 0.75}),
        ]
    )

    assert np.isclose(result["loss"], 0.625)
    assert np.isclose(result["acc"], 0.6875)


def test_parameter_validator_structural_and_numerical() -> None:
    validator = ParameterValidator(max_update_norm=10.0)
    ref = [np.zeros((3, 3), dtype=np.float32), np.ones((5,), dtype=np.float32)]

    # Exact match passes
    candidate_valid = [np.zeros((3, 3), dtype=np.float32), np.ones((5,), dtype=np.float32)]
    res_valid = validator.validate(candidate_valid, ref)
    assert res_valid.valid is True
    assert res_valid.delta_norm == 0.0

    # Shape mismatch fails
    candidate_wrong_shape = [np.zeros((3, 4), dtype=np.float32), np.ones((5,), dtype=np.float32)]
    res_shape = validator.validate(candidate_wrong_shape, ref)
    assert res_shape.valid is False
    assert any("shape mismatch" in issue for issue in res_shape.issues)

    # Tensor count mismatch fails
    candidate_wrong_count = [np.zeros((3, 3), dtype=np.float32)]
    res_count = validator.validate(candidate_wrong_count, ref)
    assert res_count.valid is False
    assert any("tensor count mismatch" in issue for issue in res_count.issues)

    # NaN / Inf fails
    candidate_nan = [
        np.array([[0.0, np.nan, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        np.ones((5,), dtype=np.float32),
    ]
    res_nan = validator.validate(candidate_nan, ref)
    assert res_nan.valid is False
    assert any("non-finite" in issue for issue in res_nan.issues)

    # Divergence norm exceeds tolerance
    candidate_divergent = [
        np.full((3, 3), 10.0, dtype=np.float32),
        np.full((5,), 10.0, dtype=np.float32),
    ]
    res_div = validator.validate(candidate_divergent, ref)
    assert res_div.valid is False
    assert any("divergence norm" in issue for issue in res_div.issues)


def test_audit_split_integrity() -> None:
    train = [Path("/data/img1.jpg"), Path("/data/img2.jpg")]
    val = [Path("/data/img3.jpg")]
    test = [Path("/data/img4.jpg")]

    clean_report = audit_split_integrity(train, val, test)
    assert clean_report.valid is True
    assert clean_report.overlap_detected is False

    # Simulate data leakage
    leaked_val = [Path("/data/img1.jpg")]
    leaked_report = audit_split_integrity(train, leaked_val, test)
    assert leaked_report.valid is False
    assert leaked_report.overlap_detected is True
    assert len(leaked_report.issues) == 1


def test_sanitized_fedprox_preserves_bn_buffers_under_momentum() -> None:
    """Verify that SanitizedFedProx applies Server Momentum only to trainable parameters
    and preserves BatchNorm running statistics bit-for-bit invariant.
    """
    from typing import Any

    import pytest

    pytest.importorskip("flwr")
    from flwr.common import (
        Code,
        FitRes,
        Status,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_proxy import ClientProxy

    from agrifed_khaos.federated import SanitizedFedProx
    from agrifed_khaos.wrappers import ParameterValidator

    validator = ParameterValidator(max_update_norm=300.0)
    param_names = ["conv.weight", "conv.bias", "bn.running_mean", "bn.running_var"]
    w_ref = np.zeros((4, 4), dtype=np.float32)
    b_ref = np.zeros((4,), dtype=np.float32)
    mean_ref = np.array([1.0, 2.0], dtype=np.float32)
    var_ref = np.array([0.5, 0.5], dtype=np.float32)
    ref_arrays = [w_ref, b_ref, mean_ref, var_ref]

    class DummyWrapper:
        def get_parameters(self) -> list[np.ndarray]:
            return [np.copy(a) for a in ref_arrays]

        def get_parameter_names(self) -> list[str]:
            return list(param_names)

        def set_parameters(self, p: list[np.ndarray]) -> None:
            pass

        def validate_parameters(self, p: list[np.ndarray]) -> Any:
            return validator.validate(p, ref_arrays)

    strategy = SanitizedFedProx(
        validator=validator,
        global_model_wrapper=DummyWrapper(),  # type: ignore[arg-type]
        server_momentum=0.9,
        eta=1.0,
        min_fit_clients=1,
        min_available_clients=1,
    )

    class MockClient(ClientProxy):
        def get_run_id(self) -> int:
            return 1

        def get_properties(self, *a: Any, **k: Any) -> Any:
            pass

        def get_parameters(self, *a: Any, **k: Any) -> Any:
            pass

        def fit(self, *a: Any, **k: Any) -> Any:
            pass

        def evaluate(self, *a: Any, **k: Any) -> Any:
            pass

        def reconnect(self, *a: Any, **k: Any) -> Any:
            pass

    client_proxy = MockClient(cid="client_1")
    client_arrays = [w_ref + 1.0, b_ref + 0.5, mean_ref + 99.0, var_ref + 88.0]
    fit_res = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(client_arrays),
        num_examples=100,
        metrics={},
    )

    agg_params, _ = strategy.aggregate_fit(
        server_round=1, results=[(client_proxy, fit_res)], failures=[]
    )
    assert agg_params is not None
    res_ndarrays = parameters_to_ndarrays(agg_params)

    # Trainable parameters must be updated with delta (momentum round 1: v_1 = delta = 1.0)
    assert np.allclose(res_ndarrays[0], w_ref + 1.0)
    assert np.allclose(res_ndarrays[1], b_ref + 0.5)

    # BatchNorm buffers MUST be preserved bit-for-bit identical to reference_arrays!
    assert np.array_equal(res_ndarrays[2], mean_ref)
    assert np.array_equal(res_ndarrays[3], var_ref)


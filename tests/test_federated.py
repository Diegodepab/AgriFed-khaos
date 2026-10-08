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

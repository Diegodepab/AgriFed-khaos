"""Federated learning client, strategy, and server orchestration with Flower.

Implements:
1. AgrifedFlowerClient: NumPyClient adapter executing local proximal training.
2. SanitizedFedProx: Subclass of Flower's native FedProx with parameter sanity & anomaly gating.
3. Server-side evaluation and weighted metrics aggregation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .wrappers import BaseFLWrapper, ParameterValidationResult, ParameterValidator

logger = logging.getLogger(__name__)

try:
    import flwr as fl
    from flwr.common import (
        FitRes,
        Parameters,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedProx

    _FLWR_AVAILABLE = True
except ImportError:  # pragma: no cover - optional flwr dependency
    fl: Any = None  # type: ignore[no-redef]
    FitRes = Any  # type: ignore[misc,assignment]
    Parameters = Any  # type: ignore[misc,assignment]
    Scalar = Any  # type: ignore[misc,assignment]
    ClientProxy = Any  # type: ignore[misc,assignment]
    FedProx = object  # type: ignore[misc,assignment]
    _FLWR_AVAILABLE = False


@dataclass(slots=True)
class ClientResult:
    """Metrics reported by a client after local training or evaluation."""

    num_examples: int
    metrics: dict[str, float]


def aggregate_weighted_metrics(results: Iterable[ClientResult]) -> dict[str, float]:
    """Compute weighted averages of metrics by sample count."""
    total_examples = 0
    accumulator: dict[str, float] = {}

    for result in results:
        if result.num_examples <= 0:
            continue
        total_examples += result.num_examples
        for key, value in result.metrics.items():
            accumulator[key] = accumulator.get(key, 0.0) + float(value) * result.num_examples

    if total_examples == 0:
        return {}

    return {key: value / total_examples for key, value in accumulator.items()}


def validate_client_update(
    wrapper: BaseFLWrapper,
    parameters: Sequence[np.ndarray],
) -> ParameterValidationResult:
    """Validate a client-side parameter update against the local wrapper reference."""
    return wrapper.validate_parameters(parameters)


class AgrifedFlowerClient(fl.client.NumPyClient if _FLWR_AVAILABLE else object):  # type: ignore[misc]
    """Flower NumPyClient wrapper for local training with FedProx regularization."""

    def __init__(
        self,
        client_id: str,
        wrapper: BaseFLWrapper,
        train_data: Any,
        eval_data: Any | None = None,
        default_prox_mu: float = 0.01,
    ) -> None:
        self.client_id = client_id
        self.wrapper = wrapper
        self.train_data = train_data
        self.eval_data = eval_data
        self.default_prox_mu = default_prox_mu

    def get_parameters(self, config: dict[str, Scalar]) -> list[np.ndarray]:
        """Return the current local model parameters."""
        return self.wrapper.get_parameters()

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Scalar],
    ) -> tuple[list[np.ndarray], int, dict[str, Scalar]]:
        """Update local model with global weights and train with FedProx proximal term."""
        self.wrapper.set_parameters(parameters)

        prox_mu = float(config.get("proximal_mu", self.default_prox_mu))
        epochs = int(config.get("local_epochs", 1))

        stats = self.wrapper.train_local_epoch(
            self.train_data,
            epochs=epochs,
            prox_mu=prox_mu,
            reference_parameters=parameters,
        )

        updated_params = self.wrapper.get_parameters()
        metrics: dict[str, Scalar] = {"loss": stats.loss}
        for k, v in stats.metrics.items():
            metrics[k] = float(v)

        return updated_params, stats.examples, metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate local or broadcast model."""
        self.wrapper.set_parameters(parameters)
        if self.eval_data is None:
            return 0.0, 0, {}

        stats = self.wrapper.evaluate(self.eval_data)
        metrics: dict[str, Scalar] = {}
        for k, v in stats.metrics.items():
            metrics[k] = float(v)

        return stats.loss, stats.examples, metrics


class SanitizedFedProx(FedProx):  # type: ignore[misc]
    """FedProx strategy with structural, numerical, and divergence parameter gating.

    Intercepts client updates before aggregation. Dropped or corrupted updates
    (containing NaNs, mismatched shapes, or exceeding max_update_norm) are excluded.
    """

    def __init__(
        self,
        validator: ParameterValidator,
        global_model_wrapper: BaseFLWrapper | None = None,
        proximal_mu: float = 0.01,
        **kwargs: Any,
    ) -> None:
        if not _FLWR_AVAILABLE:
            raise RuntimeError("flwr is required for SanitizedFedProx")
        super().__init__(proximal_mu=proximal_mu, **kwargs)
        self.validator = validator
        self.global_model_wrapper = global_model_wrapper
        self.rejection_history: list[dict[str, Any]] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Filter out anomalous/corrupted client updates before FedProx aggregation."""
        reference_params: list[np.ndarray] | None = None
        if self.global_model_wrapper is not None:
            reference_params = self.global_model_wrapper.get_parameters()

        sanitized_results: list[tuple[ClientProxy, FitRes]] = []
        for client_proxy, fit_res in results:
            candidate_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            validation = self.validator.validate(candidate_ndarrays, reference_params)

            if not validation.valid:
                logger.warning(
                    "Round %d: Rejected update from client %s. Issues: %s",
                    server_round,
                    client_proxy.cid,
                    "; ".join(validation.issues),
                )
                self.rejection_history.append(
                    {
                        "round": server_round,
                        "client_id": client_proxy.cid,
                        "issues": validation.issues,
                        "delta_norm": validation.delta_norm,
                    }
                )
                continue

            sanitized_results.append((client_proxy, fit_res))

        if not sanitized_results:
            logger.error(
                "Round %d: All client updates were rejected by parameter gate!",
                server_round,
            )
            if reference_params is not None:
                return ndarrays_to_parameters(reference_params), {"rejections": len(results)}
            return None, {"rejections": len(results)}

        aggregated_parameters, metrics = super().aggregate_fit(
            server_round,
            sanitized_results,
            failures,
        )

        if aggregated_parameters is not None and self.global_model_wrapper is not None:
            updated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            self.global_model_wrapper.set_parameters(updated_ndarrays)

        return aggregated_parameters, metrics


def build_server_evaluate_fn(
    wrapper: BaseFLWrapper,
    val_data: Any,
) -> Any:
    """Construct an evaluate_fn callback for central evaluation on validation data."""

    def evaluate_fn(
        server_round: int,
        parameters: Sequence[np.ndarray],
        config: dict[str, Scalar],
    ) -> tuple[float, dict[str, Scalar]]:
        wrapper.set_parameters(parameters)
        stats = wrapper.evaluate(val_data)
        metrics: dict[str, Scalar] = {
            "accuracy": stats.metrics.get("accuracy", 0.0),
            "examples": stats.examples,
        }
        return stats.loss, metrics

    return evaluate_fn

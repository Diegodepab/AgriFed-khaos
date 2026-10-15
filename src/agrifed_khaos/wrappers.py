"""Flower-compatible model wrappers, parameter validation, and FedProx adapters.

This module provides the Adapter pattern between vision classification models (PyTorch,
Timm ViT / CNNs) and the Federated Learning runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn


BN_BUFFER_SUFFIXES: tuple[str, ...] = (".running_mean", ".running_var", ".num_batches_tracked")


@dataclass(slots=True)
class ParameterValidationResult:
    """Result of structural, numerical, and divergence sanity checks."""

    valid: bool
    issues: list[str] = field(default_factory=list)
    delta_norm: float | None = None
    relative_norm: float | None = None


@dataclass(slots=True)
class TrainingStats:
    """Summary of local training iterations."""

    examples: int
    batches: int
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationStats:
    """Summary of local or central model evaluation."""

    examples: int
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)


class ParameterValidator:
    """Validates candidate model parameters before incorporation or aggregation."""

    def __init__(
        self,
        max_update_norm: float = 100.0,
        enforce_finite: bool = True,
        enforce_structural_match: bool = True,
        max_relative_norm: float | None = None,
    ) -> None:
        self.max_update_norm = max_update_norm
        self.enforce_finite = enforce_finite
        self.enforce_structural_match = enforce_structural_match
        self.max_relative_norm = max_relative_norm

    def validate(
        self,
        candidate_parameters: Sequence[np.ndarray],
        reference_parameters: Sequence[np.ndarray] | None = None,
        check_delta_norm: bool = True,
    ) -> ParameterValidationResult:
        """Inspect shapes, finite values, and L2 distance from global reference."""
        issues: list[str] = []

        if reference_parameters is not None and self.enforce_structural_match:
            if len(candidate_parameters) != len(reference_parameters):
                issues.append(
                    f"tensor count mismatch: expected {len(reference_parameters)}, "
                    f"got {len(candidate_parameters)}"
                )
                return ParameterValidationResult(valid=False, issues=issues)

            for index, (candidate, reference) in enumerate(
                zip(candidate_parameters, reference_parameters, strict=True)
            ):
                candidate_arr = np.asarray(candidate)
                reference_arr = np.asarray(reference)
                if candidate_arr.shape != reference_arr.shape:
                    issues.append(
                        f"tensor {index} shape mismatch: expected {reference_arr.shape}, "
                        f"got {candidate_arr.shape}"
                    )

        for index, candidate in enumerate(candidate_parameters):
            candidate_arr = np.asarray(candidate)
            if candidate_arr.dtype.kind not in {"f", "i", "u"}:
                issues.append(f"tensor {index} has unsupported dtype {candidate_arr.dtype}")
            if self.enforce_finite and not np.all(np.isfinite(candidate_arr)):
                issues.append(f"tensor {index} contains non-finite values (NaN or Inf)")

        delta_norm: float | None = None
        relative_norm: float | None = None
        if reference_parameters is not None and not issues:
            squared_sum = 0.0
            base_squared_sum = 0.0
            for candidate, reference in zip(
                candidate_parameters, reference_parameters, strict=True
            ):
                cand_arr = np.asarray(candidate)
                ref_arr = np.asarray(reference)
                diff = cand_arr - ref_arr
                squared_sum += float(np.sum(diff**2))
                base_squared_sum += float(np.sum(ref_arr**2))
            delta_norm = float(np.sqrt(squared_sum))
            base_l2_norm = float(np.sqrt(base_squared_sum))
            relative_norm = delta_norm / max(base_l2_norm, 1e-8)

            if check_delta_norm and delta_norm > self.max_update_norm:
                issues.append(
                    f"parameter update divergence norm ({delta_norm:.4f}) "
                    f"exceeds tolerance ({self.max_update_norm:.4f})"
                )
            if (
                check_delta_norm
                and self.max_relative_norm is not None
                and relative_norm > self.max_relative_norm
            ):
                issues.append(
                    f"parameter update relative norm ({relative_norm:.4f}) "
                    f"exceeds relative tolerance ({self.max_relative_norm:.4f})"
                )

        return ParameterValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            delta_norm=delta_norm,
            relative_norm=relative_norm,
        )


def freeze_bn_stats(model: Any) -> None:
    """Freeze running statistics (running_mean, running_var) of all BatchNorm layers.

    Puts all BatchNorm layers into eval() mode so that they use pre-computed global
    normalization statistics from the base model and do not accumulate local batch statistics
    during client training rounds. Crucially, affine parameters (weight and bias) remain
    trainable (requires_grad=True).
    """
    try:
        from torch.nn.modules.batchnorm import _BatchNorm
    except ImportError:
        return

    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.eval()


def apply_adaptive_rms_noise(
    parameters: Sequence[Any] | dict[str, Any],
    noise_std: float,
    min_rms: float = 1e-4,
    parameter_names: Sequence[str] | None = None,
) -> None:
    """Apply heuristic layer-wise RMS-scaled Gaussian parameter noise perturbation.

    Note: This is post-training weight perturbation for heuristic model obfuscation
    against feature reconstruction attacks; it does not constitute formal (epsilon, delta)-DP.
    For mathematically bounded Differential Privacy, use apply_dp_gaussian_noise.

    Scales injected noise relative to the root-mean-square magnitude of each
    trainable tensor:
        rms(p) = max(||p||_2 / sqrt(p.numel()), min_rms)
        noise = N(0, I) * noise_std * rms(p)

    Guarantees that BatchNorm buffers (running_mean, running_var, num_batches_tracked)
    and non-floating tensors are strictly immune and never perturbed.
    """
    if noise_std <= 0.0:
        return
    try:
        import torch
    except ImportError:
        return

    items: list[tuple[str | None, Any]] = []
    if isinstance(parameters, dict):
        items = list(parameters.items())
    elif parameter_names is not None:
        items = list(zip(parameter_names, parameters, strict=False))
    else:
        for p in parameters:
            if isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str):
                items.append((p[0], p[1]))
            else:
                items.append((getattr(p, "name", None), p))

    with torch.no_grad():
        for name, p in items:
            if name is not None and any(name.endswith(suffix) for suffix in BN_BUFFER_SUFFIXES):
                continue
            tensor = p.data if hasattr(p, "data") else p
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                continue
            p_float = tensor.float()
            p_norm = float(torch.linalg.norm(p_float).item())
            numel = max(tensor.numel(), 1)
            rms = max(p_norm / (numel**0.5), min_rms)
            noise = torch.randn_like(tensor) * (noise_std * rms)
            tensor.add_(noise)


def apply_dp_gaussian_noise(
    parameters: Sequence[Any] | dict[str, Any],
    reference_parameters: Sequence[Any] | dict[str, Any] | None = None,
    clip_norm: float = 1.0,
    noise_std: float = 0.0,
    *,
    reference_params: Sequence[Any] | dict[str, Any] | None = None,
    parameter_names: Sequence[str] | None = None,
) -> float:
    """Apply formal Gaussian Differential Privacy mechanism on client parameter updates.

    Computes Delta W = W_client - W_base.
    1. Enforces static joint global L2 clipping:
       Delta W_i <- Delta W_i * min(1, clip_norm / sqrt(sum_k ||Delta W_k||_2^2))
    2. Injects Gaussian perturbation calibrated to clip_norm:
       Delta W_i <- Delta W_i + N(0, (noise_std * clip_norm)^2 * I)
    3. Reconstructs W_client = W_base + Delta W.

    Guarantees:
    - Joint global L2 norm clipping across all trainable parameters (not per-layer).
    - BatchNorm buffers (running_mean, running_var, num_batches_tracked) are strictly
      immune and never perturbed with noise.
    - Noise standard deviation is strictly independent of the client's tensor weights.

    Returns:
        Original joint global update L2 norm before clipping.
    """
    if clip_norm <= 0.0:
        raise ValueError("clip_norm must be positive for DP clipping")
    if noise_std < 0.0:
        raise ValueError("noise_std cannot be negative")

    ref_list_raw = reference_parameters if reference_parameters is not None else reference_params

    try:
        import torch
    except ImportError:
        return 0.0

    param_items: list[tuple[str | None, Any]] = []
    if isinstance(parameters, dict):
        param_items = list(parameters.items())
    elif parameter_names is not None:
        param_items = list(zip(parameter_names, parameters, strict=False))
    else:
        for p in parameters:
            if isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str):
                param_items.append((p[0], p[1]))
            else:
                param_items.append((getattr(p, "name", None), p))

    ref_tensors_raw: list[Any] = []
    if ref_list_raw is None:
        ref_tensors_raw = [
            torch.zeros_like(p.data if hasattr(p, "data") else p)
            for _, p in param_items
        ]
    elif isinstance(ref_list_raw, dict):
        ref_tensors_raw = [
            ref_list_raw.get(name, torch.zeros_like(p.data if hasattr(p, "data") else p))
            if name is not None
            else torch.zeros_like(p.data if hasattr(p, "data") else p)
            for name, p in param_items
        ]
    else:
        ref_tensors_raw = list(ref_list_raw)

    with torch.no_grad():
        active_triplets: list[tuple[Any, Any, Any]] = []
        for (name, p), ref in zip(param_items, ref_tensors_raw, strict=True):
            p_t = p.data if hasattr(p, "data") else p
            ref_t = ref.data if hasattr(ref, "data") else ref
            if not isinstance(p_t, torch.Tensor) or not isinstance(ref_t, torch.Tensor):
                continue
            if name is not None and any(name.endswith(suffix) for suffix in BN_BUFFER_SUFFIXES):
                continue
            if not p_t.is_floating_point():
                continue

            delta = p_t - ref_t
            active_triplets.append((p_t, ref_t, delta))

        total_norm_sq = sum(torch.sum(delta.float() ** 2).item() for _, _, delta in active_triplets)
        total_norm = total_norm_sq**0.5

        scale = min(1.0, clip_norm / max(total_norm, 1e-8))
        for p_t, ref_t, delta in active_triplets:
            clipped_delta = delta * scale
            if noise_std > 0.0:
                noise = torch.randn_like(p_t) * (noise_std * clip_norm)
                clipped_delta = clipped_delta + noise
            p_t.copy_(ref_t + clipped_delta)

    return float(total_norm)


class BaseFLWrapper(ABC):
    """Abstract base adapter decoupling FL orchestrator from vision architectures."""

    def __init__(self, validator: ParameterValidator | None = None) -> None:
        self.validator = validator or ParameterValidator()

    @abstractmethod
    def get_parameters(self) -> list[np.ndarray]:
        """Extract trainable parameters as a list of NumPy arrays."""
        raise NotImplementedError

    @abstractmethod
    def get_parameter_names(self) -> list[str]:
        """Extract parameter names corresponding to get_parameters()."""
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        """Inject parameters into the underlying model."""
        raise NotImplementedError

    @abstractmethod
    def train_local_epoch(
        self,
        train_data: Any,
        *,
        epochs: int,
        prox_mu: float = 0.0,
        reference_parameters: Sequence[np.ndarray] | None = None,
        max_grad_norm: float | None = 1.0,
        missing_classes: Sequence[int] | None = None,
    ) -> TrainingStats:
        """Run local optimization round with optional FedProx proximal penalty
        and missing class masking.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, eval_data: Any) -> EvaluationStats:
        """Evaluate the model on an evaluation dataloader or dataset."""
        raise NotImplementedError

    def validate_parameters(
        self,
        parameters: Sequence[np.ndarray],
        reference_parameters: Sequence[np.ndarray] | None = None,
        check_delta_norm: bool = True,
    ) -> ParameterValidationResult:
        """Validate candidate parameters against model reference."""
        reference = (
            reference_parameters if reference_parameters is not None else self.get_parameters()
        )
        return self.validator.validate(
            parameters, reference, check_delta_norm=check_delta_norm
        )


class TorchModuleWrapper(BaseFLWrapper):
    """Generic wrapper for PyTorch nn.Module architectures.

    Ensures that only trainable parameters (requires_grad) are federated,
    preventing state_dict buffer mismatches (e.g. BatchNorm running stats)
    during proximal loss calculation.
    """

    def __init__(
        self,
        module: nn.Module,
        optimizer: Any,
        criterion: Any,
        device: str = "cpu",
        validator: ParameterValidator | None = None,
    ) -> None:
        super().__init__(validator=validator)
        self.module = module
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def _require_torch(self) -> Any:
        try:
            import torch

            return torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for TorchModuleWrapper") from exc

    def get_parameters(self) -> list[np.ndarray]:
        """Extract trainable parameters in consistent order."""
        return [
            param.detach().cpu().numpy().copy()
            for param in self.module.parameters()
            if param.requires_grad
        ]

    def get_parameter_names(self) -> list[str]:
        """Extract parameter names corresponding to get_parameters()."""
        return [
            name
            for name, param in self.module.named_parameters()
            if param.requires_grad
        ]

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        """Inject parameters directly into trainable layers."""
        torch = self._require_torch()
        validation = self.validate_parameters(parameters, check_delta_norm=False)
        if not validation.valid:
            raise ValueError(f"Invalid parameters rejected: {'; '.join(validation.issues)}")

        trainable_params = [p for p in self.module.parameters() if p.requires_grad]
        for param, candidate in zip(trainable_params, parameters, strict=True):
            tensor = torch.as_tensor(np.asarray(candidate), device=param.device, dtype=param.dtype)
            param.data.copy_(tensor)

    def train_local_epoch(
        self,
        train_data: Any,
        *,
        epochs: int,
        prox_mu: float = 0.0,
        reference_parameters: Sequence[np.ndarray] | None = None,
        max_grad_norm: float | None = 1.0,
        missing_classes: Sequence[int] | None = None,
    ) -> TrainingStats:
        """Train local epochs with exact FedProx proximal regularization and gradient clipping."""
        torch = self._require_torch()
        self.module.to(self.device)

        trainable_params = [p for p in self.module.parameters() if p.requires_grad]
        ref_tensors: list[Any] | None = None
        if reference_parameters is not None and prox_mu > 0.0:
            ref_tensors = [
                torch.as_tensor(np.asarray(p), device=self.device, dtype=p_orig.dtype)
                for p, p_orig in zip(reference_parameters, trainable_params, strict=True)
            ]

        # Explicitly isolate classifier head parameters to avoid masking hidden layers
        head_params: list[Any] = []
        for n, p in self.module.named_parameters():
            if any(h in n for h in ("fc", "classifier", "head")) and p.requires_grad:
                head_params.append(p)
        if not head_params:
            if trainable_params:
                head_params = trainable_params[-2:]

        missing_set = set(missing_classes) if missing_classes else None
        saved_missing_weights: list[tuple[Any, Any]] = []
        if missing_set:
            miss_list = list(missing_set)
            max_cls = max(miss_list)
            for p in head_params:
                if p.dim() in (1, 2) and p.shape[0] > max_cls:
                    saved_missing_weights.append((p, p.detach()[miss_list].clone()))

        examples = 0
        batches = 0
        total_loss = 0.0

        for _ in range(epochs):
            self.module.train()
            freeze_bn_stats(self.module)
            for batch in train_data:
                batches += 1
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.module(inputs)
                loss = self.criterion(outputs, targets)

                if ref_tensors is not None and prox_mu > 0.0:
                    prox_penalty = torch.zeros((), device=self.device)
                    for param, ref in zip(trainable_params, ref_tensors, strict=True):
                        prox_penalty = prox_penalty + torch.sum((param - ref) ** 2)
                    loss = loss + 0.5 * prox_mu * prox_penalty

                loss.backward()

                # Zero gradients for missing classes in classification head
                if missing_set:
                    miss_list = list(missing_set)
                    max_cls = max(miss_list)
                    for p in head_params:
                        if p.grad is not None and p.dim() in (1, 2) and p.shape[0] > max_cls:
                            if p.dim() == 2:
                                p.grad[miss_list, :] = 0.0
                            elif p.dim() == 1:
                                p.grad[miss_list] = 0.0

                if max_grad_norm is not None and max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
                self.optimizer.step()

                # Hard projection: strictly restore base rows for missing classes after step
                if saved_missing_weights:
                    with torch.no_grad():
                        for p, base_rows in saved_missing_weights:
                            if p.dim() == 2:
                                p[miss_list, :] = base_rows
                            elif p.dim() == 1:
                                p[miss_list] = base_rows

                batch_size = int(inputs.shape[0]) if hasattr(inputs, "shape") else 1
                examples += batch_size
                total_loss += float(loss.detach().cpu().item()) * batch_size

        avg_loss = total_loss / max(examples, 1)
        return TrainingStats(examples=examples, batches=batches, loss=avg_loss)

    def evaluate(self, eval_data: Any) -> EvaluationStats:
        """Evaluate accuracy and loss on evaluation dataset with exact sample weighting."""
        torch = self._require_torch()
        self.module.to(self.device)
        self.module.eval()

        total_loss = 0.0
        correct = 0
        examples = 0

        with torch.no_grad():
            for batch in eval_data:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.module(inputs)
                loss = self.criterion(outputs, targets)

                preds = torch.argmax(outputs, dim=1)
                correct += int(torch.sum(preds == targets).item())
                batch_size = int(inputs.shape[0]) if hasattr(inputs, "shape") else 1
                examples += batch_size
                total_loss += float(loss.detach().cpu().item()) * batch_size

        avg_loss = total_loss / max(examples, 1)
        accuracy = float(correct / max(examples, 1))
        return EvaluationStats(
            examples=examples,
            loss=avg_loss,
            metrics={"accuracy": accuracy},
        )


class TimmClassificationWrapper(TorchModuleWrapper):
    """Adapter for Timm Vision Transformer (ViT) and CNN classification models."""

    @classmethod
    def create(
        cls,
        model_name: str,
        num_classes: int,
        learning_rate: float = 1e-3,
        pretrained: bool = False,
        device: str = "cpu",
        validator: ParameterValidator | None = None,
    ) -> TimmClassificationWrapper:
        """Factory method to instantiate a Timm classification model."""
        try:
            import timm
            import torch.nn as nn
            import torch.optim as optim
        except ImportError as exc:
            raise RuntimeError("timm and torch are required for TimmClassificationWrapper") from exc

        module = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(module.parameters(), lr=learning_rate)
        return cls(
            module=module,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            validator=validator,
        )


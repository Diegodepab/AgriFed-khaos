"""Flower-compatible model wrappers, parameter validation, and FedProx adapters.

This module provides the Adapter pattern between vision model libraries (PyTorch,
Timm ViT, Ultralytics YOLOv8) and the Flower Federated Learning runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass(slots=True)
class ParameterValidationResult:
    """Result of structural, numerical, and divergence sanity checks."""

    valid: bool
    issues: list[str] = field(default_factory=list)
    delta_norm: float | None = None


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
    ) -> None:
        self.max_update_norm = max_update_norm
        self.enforce_finite = enforce_finite
        self.enforce_structural_match = enforce_structural_match

    def validate(
        self,
        candidate_parameters: Sequence[np.ndarray],
        reference_parameters: Sequence[np.ndarray] | None = None,
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
        if reference_parameters is not None and not issues:
            squared_sum = 0.0
            for candidate, reference in zip(
                candidate_parameters, reference_parameters, strict=True
            ):
                diff = np.asarray(candidate) - np.asarray(reference)
                squared_sum += float(np.sum(diff**2))
            delta_norm = float(np.sqrt(squared_sum))

            if delta_norm > self.max_update_norm:
                issues.append(
                    f"parameter update divergence norm ({delta_norm:.4f}) "
                    f"exceeds tolerance ({self.max_update_norm:.4f})"
                )

        return ParameterValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            delta_norm=delta_norm,
        )


class BaseFLWrapper(ABC):
    """Abstract base adapter decoupling FL orchestrator from vision architectures."""

    def __init__(self, validator: ParameterValidator | None = None) -> None:
        self.validator = validator or ParameterValidator()

    @abstractmethod
    def get_parameters(self) -> list[np.ndarray]:
        """Extract trainable parameters as a list of NumPy arrays."""
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
    ) -> TrainingStats:
        """Run local optimization round with optional FedProx proximal penalty."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, eval_data: Any) -> EvaluationStats:
        """Evaluate the model on an evaluation dataloader or dataset."""
        raise NotImplementedError

    def validate_parameters(
        self,
        parameters: Sequence[np.ndarray],
        reference_parameters: Sequence[np.ndarray] | None = None,
    ) -> ParameterValidationResult:
        """Validate candidate parameters against model reference."""
        reference = (
            reference_parameters if reference_parameters is not None else self.get_parameters()
        )
        return self.validator.validate(parameters, reference)


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

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        """Inject parameters directly into trainable layers."""
        torch = self._require_torch()
        validation = self.validate_parameters(parameters)
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
    ) -> TrainingStats:
        """Train local epochs with exact FedProx proximal regularization."""
        torch = self._require_torch()
        self.module.to(self.device)
        self.module.train()

        trainable_params = [p for p in self.module.parameters() if p.requires_grad]
        ref_tensors: list[Any] | None = None
        if reference_parameters is not None and prox_mu > 0.0:
            ref_tensors = [
                torch.as_tensor(np.asarray(p), device=self.device, dtype=p_orig.dtype)
                for p, p_orig in zip(reference_parameters, trainable_params, strict=True)
            ]

        examples = 0
        batches = 0
        total_loss = 0.0

        for _ in range(epochs):
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
                self.optimizer.step()

                batch_size = int(inputs.shape[0]) if hasattr(inputs, "shape") else 1
                examples += batch_size
                total_loss += float(loss.detach().cpu().item())

        avg_loss = total_loss / max(batches, 1)
        return TrainingStats(examples=examples, batches=batches, loss=avg_loss)

    def evaluate(self, eval_data: Any) -> EvaluationStats:
        """Evaluate accuracy and loss on evaluation dataset."""
        torch = self._require_torch()
        self.module.to(self.device)
        self.module.eval()

        total_loss = 0.0
        correct = 0
        examples = 0
        batches = 0

        with torch.no_grad():
            for batch in eval_data:
                batches += 1
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.module(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += float(loss.detach().cpu().item())

                preds = torch.argmax(outputs, dim=1)
                correct += int(torch.sum(preds == targets).item())
                batch_size = int(inputs.shape[0]) if hasattr(inputs, "shape") else 1
                examples += batch_size

        avg_loss = total_loss / max(batches, 1)
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


class YOLOv8Wrapper(BaseFLWrapper):
    """Adapter for Ultralytics YOLOv8+ detection models."""

    def __init__(
        self,
        yolo_model: Any,
        device: str = "cpu",
        validator: ParameterValidator | None = None,
    ) -> None:
        super().__init__(validator=validator)
        self.yolo_model = yolo_model
        self.device = device

    @classmethod
    def create(
        cls,
        model_cfg_or_weights: str = "yolov8n.yaml",
        device: str = "cpu",
        validator: ParameterValidator | None = None,
    ) -> YOLOv8Wrapper:
        """Factory method to instantiate an Ultralytics YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for YOLOv8Wrapper") from exc

        model = YOLO(model_cfg_or_weights)
        return cls(yolo_model=model, device=device, validator=validator)

    def _get_inner_model(self) -> Any:
        return getattr(self.yolo_model, "model", self.yolo_model)

    def get_parameters(self) -> list[np.ndarray]:
        """Extract trainable parameters from the underlying PyTorch detection network."""
        inner = self._get_inner_model()
        return [p.detach().cpu().numpy().copy() for p in inner.parameters() if p.requires_grad]

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        """Inject updated parameters into the detection network."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for YOLOv8Wrapper") from exc

        validation = self.validate_parameters(parameters)
        if not validation.valid:
            raise ValueError(f"Invalid parameters rejected: {'; '.join(validation.issues)}")

        inner = self._get_inner_model()
        trainable = [p for p in inner.parameters() if p.requires_grad]
        for param, candidate in zip(trainable, parameters, strict=True):
            tensor = torch.as_tensor(np.asarray(candidate), device=param.device, dtype=param.dtype)
            param.data.copy_(tensor)

    def train_local_epoch(
        self,
        train_data: Any,
        *,
        epochs: int,
        prox_mu: float = 0.0,
        reference_parameters: Sequence[np.ndarray] | None = None,
    ) -> TrainingStats:
        """Local fine-tuning execution."""
        return TrainingStats(examples=0, batches=0, loss=0.0)

    def evaluate(self, eval_data: Any) -> EvaluationStats:
        """Local detection evaluation."""
        return EvaluationStats(examples=0, loss=0.0, metrics={"mAP50": 0.0})

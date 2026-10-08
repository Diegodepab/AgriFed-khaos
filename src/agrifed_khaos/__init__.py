"""AgriFed-KHAOS federated learning scaffold for agricultural computer vision."""

from .config import (
    AppConfig,
    DatasetConfig,
    FederationConfig,
    ModelConfig,
    ParameterGateConfig,
    QualityConfig,
    RuntimeSettings,
)
from .data import (
    DatasetInventory,
    DatasetRecord,
    create_torch_dataloader,
    load_inventory,
    load_records,
    partition_train_records,
)
from .federated import (
    AgrifedFlowerClient,
    ClientResult,
    SanitizedFedProx,
    aggregate_weighted_metrics,
    build_server_evaluate_fn,
    validate_client_update,
)
from .quality import (
    BaselineQualityReport,
    SplitIntegrityReport,
    audit_split_integrity,
    evaluate_pre_fl_baseline,
)
from .wrappers import (
    BaseFLWrapper,
    EvaluationStats,
    ParameterValidationResult,
    ParameterValidator,
    TimmClassificationWrapper,
    TorchModuleWrapper,
    TrainingStats,
    YOLOv8Wrapper,
)

__all__ = [
    "AgrifedFlowerClient",
    "AppConfig",
    "BaseFLWrapper",
    "BaselineQualityReport",
    "ClientResult",
    "DatasetConfig",
    "DatasetInventory",
    "DatasetRecord",
    "EvaluationStats",
    "FederationConfig",
    "ModelConfig",
    "ParameterGateConfig",
    "ParameterValidationResult",
    "ParameterValidator",
    "QualityConfig",
    "RuntimeSettings",
    "SanitizedFedProx",
    "SplitIntegrityReport",
    "TimmClassificationWrapper",
    "TorchModuleWrapper",
    "TrainingStats",
    "YOLOv8Wrapper",
    "aggregate_weighted_metrics",
    "audit_split_integrity",
    "build_server_evaluate_fn",
    "create_torch_dataloader",
    "evaluate_pre_fl_baseline",
    "load_inventory",
    "load_records",
    "partition_train_records",
    "validate_client_update",
]

"""
Core inference components - Pure Python, zero ROS dependencies.

This module provides the building blocks for inference pipelines:
- PureInferenceEngine: Stateless GPU inference engine
- TensorPreprocessor: Tensor normalization
- TensorPostprocessor: Tensor denormalization
- InferenceCoordinator: Zero-copy composition of all components

All components can be tested independently in Jupyter/PyTest without ROS.
"""

from importlib import import_module
from typing import Any

from inference_service.core.coordinator import (
    CoordinatorConfig,
    CoordinatorResult,
    InferenceCoordinator,
)
from inference_service.core.postprocessor import (
    MockPostprocessor,
    PostprocessorBase,
    TensorPostprocessor,
)
from inference_service.core.preprocessor import (
    MockPreprocessor,
    PreprocessorBase,
    TensorPreprocessor,
)
from inference_service.core.pure_inference_engine import (
    InferenceResult,
    MockPolicyWrapper,
    PolicyWrapper,
    PureInferenceEngine,
    resolve_device,
)

_LAZY_EXPORTS = {
    "AscendOM3403PolicyWrapper": (
        "inference_service.core.ascend_om",
        "AscendOM3403PolicyWrapper",
    ),
    "AscendOMPolicyWrapper": (
        "inference_service.core.ascend_om",
        "AscendOMPolicyWrapper",
    ),
    "create_ascend_om_policy_wrapper": (
        "inference_service.core.ascend_om",
        "create_ascend_om_policy_wrapper",
    ),
    "resolve_om_model_path": (
        "inference_service.core.ascend_om",
        "resolve_om_model_path",
    ),
    "RKNNPolicyWrapper": (
        "inference_service.core.rknn",
        "RKNNPolicyWrapper",
    ),
    "create_rknn_policy_wrapper": (
        "inference_service.core.rknn",
        "create_rknn_policy_wrapper",
    ),
}


def __getattr__(name: str) -> Any:
    module_name, attr_name = _LAZY_EXPORTS.get(name, (None, None))
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "PureInferenceEngine",
    "InferenceResult",
    "PolicyWrapper",
    "MockPolicyWrapper",
    "resolve_device",
    "AscendOM3403PolicyWrapper",
    "AscendOMPolicyWrapper",
    "create_ascend_om_policy_wrapper",
    "resolve_om_model_path",
    "RKNNPolicyWrapper",
    "create_rknn_policy_wrapper",
    "TensorPreprocessor",
    "PreprocessorBase",
    "MockPreprocessor",
    "TensorPostprocessor",
    "PostprocessorBase",
    "MockPostprocessor",
    "InferenceCoordinator",
    "CoordinatorConfig",
    "CoordinatorResult",
]

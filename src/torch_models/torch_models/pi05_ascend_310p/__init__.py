"""PI0.5 model optimized for Ascend310P native Torch inference."""

from torch_models.pi05_ascend_310p.modeling_pi05_ascend_310p import (
    PI05Ascend310PPolicy,
    configure_pi05_ascend_310p_config,
)

__all__ = [
    "PI05Ascend310PPolicy",
    "configure_pi05_ascend_310p_config",
]

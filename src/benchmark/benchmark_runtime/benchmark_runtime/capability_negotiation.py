"""Provider-independent startup capability negotiation.

A benchmark plan describes features requested by the user and resolved by the
provider plugin.  The adapter separately advertises what its native runtime can
supply.  This module compares those two views before the first reset so an
unsupported feature never degrades silently at runtime.

Reward and success are intentionally optional result fields.  Their capability
flags are recorded in the agreement but are not requirements unless a future
plan schema explicitly requests them.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_runtime.models import BenchmarkCapabilities
from benchmark_runtime.plan import BenchmarkPlan


class CapabilityNegotiationError(ValueError):
    """Raised when a requested evaluation feature is unsupported."""


@dataclass(frozen=True, slots=True)
class BenchmarkCapabilityAgreement:
    """Immutable result of startup negotiation for one provider and plan."""

    required: tuple[str, ...]
    optional_available: tuple[str, ...]


def negotiate_provider_capabilities(
    capabilities: BenchmarkCapabilities,
    plan: BenchmarkPlan,
) -> BenchmarkCapabilityAgreement:
    """Validate plan requirements against adapter capabilities.

    Initial-state selection and provider-native artifacts are explicit plan
    requirements.  Reward and success remain optional wire fields, so a
    provider may advertise either value without blocking startup.
    """
    if not isinstance(capabilities, BenchmarkCapabilities):
        raise CapabilityNegotiationError(
            f"adapter.capabilities must return BenchmarkCapabilities, got {type(capabilities).__name__}"
        )
    if not isinstance(plan, BenchmarkPlan):
        raise CapabilityNegotiationError(f"plan must be BenchmarkPlan, got {type(plan).__name__}")

    requested: list[tuple[str, bool]] = [
        ("init_state", plan.init_state_policy.use_init_state_id),
        (
            "native_artifact",
            any(
                (
                    plan.artifacts.save_sim_states,
                    plan.artifacts.video_enabled,
                    plan.artifacts.write_native,
                )
            ),
        ),
    ]
    support = {
        "init_state": capabilities.supports_init_state,
        "native_artifact": capabilities.supports_native_artifact,
    }
    required = tuple(name for name, enabled in requested if enabled)
    missing = tuple(name for name in required if not support[name])
    if missing:
        raise CapabilityNegotiationError(
            "provider capability negotiation failed; requested features are unsupported: " + ", ".join(missing)
        )
    if plan.lane_count > capabilities.max_lane_count:
        raise CapabilityNegotiationError(
            f"provider supports at most {capabilities.max_lane_count} evaluation lane(s); "
            f"requested lane_count={plan.lane_count}"
        )

    optional_available = tuple(
        name
        for name, available in (
            ("render", capabilities.supports_render),
            ("reward", capabilities.supports_reward),
            ("success", capabilities.supports_success),
        )
        if available
    )
    return BenchmarkCapabilityAgreement(required=required, optional_available=optional_available)

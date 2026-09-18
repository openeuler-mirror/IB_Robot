from types import SimpleNamespace

import pytest

from torch_models import policy_provider


def test_none_architecture_uses_default_policy_provider() -> None:
    assert policy_provider.resolve_policy_provider(None) is None


def test_pi05_ascend_architecture_is_loaded_lazily(monkeypatch) -> None:
    policy_class = type("Policy", (), {})
    configure = object()
    module = SimpleNamespace(
        PI05Ascend310PPolicy=policy_class,
        configure_pi05_ascend_310p_config=configure,
    )
    calls = []
    monkeypatch.setattr(policy_provider, "import_module", lambda name: calls.append(name) or module)

    provider = policy_provider.resolve_policy_provider("pi05-ascend-310p")

    assert calls == ["torch_models.pi05_ascend_310p"]
    assert provider.policy_class is policy_class
    assert provider.configure_config is configure


def test_unknown_architecture_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown torch_models architecture_class"):
        policy_provider.resolve_policy_provider("unknown-policy")

from types import SimpleNamespace

import pytest

from torch_models import policy_provider


@pytest.mark.parametrize(
    "identity",
    [("act", "torch", "cpu"), ("pi05", "torch", "cuda"), ("pi05", "ascend", "npu"), ("pi05", "torch", "cpu")],
)
def test_unregistered_runtime_uses_default_policy_provider(monkeypatch, identity) -> None:
    def unexpected_import(_name):
        pytest.fail("default policy routing must not import an optional custom model")

    monkeypatch.setattr(policy_provider, "import_module", unexpected_import)
    assert policy_provider.resolve_policy_provider(*identity) is None


def test_pi05_npu_provider_is_loaded_lazily(monkeypatch) -> None:
    policy_class = type("Policy", (), {})
    configure = object()
    validate = object()
    prepare = object()
    expected = policy_provider.PolicyProvider(
        policy_class=policy_class,
        configure_config=configure,
        validate=validate,
        prepare=prepare,
        load_options={"skip_weight_init": True},
    )
    module = SimpleNamespace(create_provider=lambda: expected)
    calls = []
    monkeypatch.setattr(policy_provider, "import_module", lambda name: calls.append(name) or module)

    provider = policy_provider.resolve_policy_provider("pi05", "torch", "npu")

    assert calls == ["torch_models.pi05_ascend_310p.provider"]
    assert provider.policy_class is policy_class
    assert provider.configure_config is configure
    assert provider.validate is validate
    assert provider.prepare is prepare
    assert provider is expected


def test_other_model_backend_is_default() -> None:
    assert policy_provider.resolve_policy_provider("smolvla", "torch", "npu") is None


def test_registered_provider_import_failure_is_not_a_fallback(monkeypatch) -> None:
    def broken_import(_name):
        raise ImportError("missing model dependency")

    monkeypatch.setattr(policy_provider, "import_module", broken_import)
    with pytest.raises(ImportError, match="missing model dependency"):
        policy_provider.resolve_policy_provider("pi05", "torch", "npu")

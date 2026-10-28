from __future__ import annotations

from inference_service.backends.ascend.acl_runtime import AclRuntimeManager


class _FakeRuntime:
    def __init__(self, owner: _FakeAcl) -> None:
        self.owner = owner

    def set_device(self, device_id: int) -> int:
        self.owner.set_device_calls.append(device_id)
        return 0

    def reset_device(self, device_id: int) -> int:
        self.owner.reset_device_calls.append(device_id)
        return 0

    def create_context(self, device_id: int) -> tuple[object, int]:
        context = ("context", device_id, len(self.owner.contexts))
        self.owner.contexts.add(context)
        return context, 0

    def set_context(self, context: object) -> int:
        return 0 if context in self.owner.contexts else 1

    def destroy_context(self, context: object) -> int:
        self.owner.contexts.remove(context)
        return 0


class _FakeAcl:
    def __init__(self) -> None:
        self.init_calls = 0
        self.finalize_calls = 0
        self.set_device_calls: list[int] = []
        self.reset_device_calls: list[int] = []
        self.contexts: set[object] = set()
        self.rt = _FakeRuntime(self)

    def init(self) -> int:
        self.init_calls += 1
        return 0

    def finalize(self) -> int:
        self.finalize_calls += 1
        return 0


def test_runtime_manager_uses_default_acl_init_for_all_leases() -> None:
    acl = _FakeAcl()
    manager = AclRuntimeManager(lambda: acl)

    assert not hasattr(manager, "_config_path")
    first = manager.acquire(0)
    second = manager.acquire(0)

    assert acl.init_calls == 1
    first.close()
    assert acl.finalize_calls == 0
    second.close()
    assert acl.finalize_calls == 1
    assert acl.set_device_calls == [0]
    assert acl.reset_device_calls == [0]

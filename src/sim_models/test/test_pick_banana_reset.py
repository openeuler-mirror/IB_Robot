"""Legacy episode reset ordering for the pick-banana simulation task."""

from types import SimpleNamespace

import pytest

from sim_models.tasks.pick_banana import PickBananaTask


@pytest.mark.parametrize("randomize", [False, True])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("world_ok", [False, True])
def test_world_reset_preserves_legacy_episode_order(randomize, resume, world_ok):
    task = object.__new__(PickBananaTask)
    task._disp_stop = object()
    task._disp_start = object()
    task._disp_reset = object()
    task._node = SimpleNamespace(get_logger=lambda: SimpleNamespace(warning=lambda _message: None))
    calls = []
    task._call_service_sync = lambda _client, _request, label, **_kwargs: calls.append(label) or True
    task._publish_rest_pose = lambda: calls.append("rest")
    task.reset = lambda: (calls.append("reset") or world_ok, "world result")
    task.randomize = lambda: (calls.append("randomize") or world_ok, "world result")

    result = task._clean_world_reset(randomize=randomize, resume=resume, settle_s=0)

    expected = ["dispatcher/stop", "rest", "randomize" if randomize else "reset"]
    if resume:
        if world_ok:
            expected.append("dispatcher/reset")
        expected.append("dispatcher/start")
    assert result == (world_ok, "world result")
    assert calls == expected

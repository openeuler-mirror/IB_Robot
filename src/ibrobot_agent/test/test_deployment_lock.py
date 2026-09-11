from __future__ import annotations

import pytest

from ibrobot_agent.deployment_lock import DeploymentLock


def test_deployment_lock_rejects_second_owner_and_releases(tmp_path):
    path = tmp_path / "agent.lock"
    first = DeploymentLock(path)
    try:
        with pytest.raises(RuntimeError, match="already held"):
            DeploymentLock(path)
    finally:
        first.close()

    second = DeploymentLock(path)
    second.close()

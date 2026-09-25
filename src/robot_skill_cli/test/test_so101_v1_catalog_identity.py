"""Frozen catalog identity for the so101_arm_v1 base profile.

``compile_local_snapshot`` is robot_skill_cli's, so this assertion lives here:
skill_catalog cannot declare a test dependency on robot_skill_cli, which already
depends on skill_catalog, and colcon refuses to order a workspace with a cycle.

The digests are pinned to the profile as it stood at SO101_V1_BASE_COMMIT. A
change here means the compiled registry or capability surface moved, which is a
contract change for every client of the so101_arm_v1 profile - update the digest
only together with that decision.
"""

from __future__ import annotations

from pathlib import Path

from robot_config.loader import load_robot_config_dict
from robot_skill_cli.catalog import compile_local_snapshot

SRC_ROOT = Path(__file__).resolve().parents[2]
ROBOT_CONFIG_DIR = SRC_ROOT / "robot_config" / "config" / "robots"

# Rebased onto upstream master 830a9d2fd: the robot-context contract moved to
# schema v3 (supported_control_modes, versioned execution endpoints, new
# delegated executors), so the registry preimage changed while the compiled
# capability surface stayed identical.
SO101_V1_BASE_COMMIT = "830a9d2fd0361d20070cb5e9ea98149f5124bb08"
SO101_V1_REGISTRY_DIGEST = "7842f9cd052dcbbced1d8a54c84aa98695409a2060601a21f271b3195384a408"
SO101_V1_CAPABILITY_DIGEST = "9899815166f5684ba69b8401c94e623a90c41460be308e7250d3d3850395f6a4"


def test_so101_v1_registry_and_capability_digests_match_base_identity(monkeypatch) -> None:
    monkeypatch.setenv("WORKSPACE", str(SRC_ROOT.parent))
    config_path = ROBOT_CONFIG_DIR / "so101_single_arm.yaml"
    # The compiler test reads declarations only; the public model binder
    # supplies the live arm context that is deferred here.
    config = load_robot_config_dict(config_path, defer_interface_binding=True)
    config["joints"] = {"arm": ["1", "2", "3", "4", "5"]}
    snapshot = compile_local_snapshot(config, config_path)

    assert snapshot.registry_digest == SO101_V1_REGISTRY_DIGEST, SO101_V1_BASE_COMMIT
    assert snapshot.capability_digest == SO101_V1_CAPABILITY_DIGEST, SO101_V1_BASE_COMMIT

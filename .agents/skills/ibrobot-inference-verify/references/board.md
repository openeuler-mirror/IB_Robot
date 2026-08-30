# Board Verification Commands

## When to Read

- Running tiers 5–6 of the inference verification matrix
- Need the SSH, OM script, and board ROS mock launch commands

## Prerequisites

- SSH access: `ssh OPi_20T` (Ascend 310B, openEuler Embedded aarch64)
- Board has `/IB_Robot` repo with latest code checked out
- LeRobot patch stack applied (`scripts/setup.sh --only-patch --yes`)
- Board colcon workspace built (`install/` overlay exists)
- CANN environment: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- `decorator` pip package installed (needed by torch_npu)

## Tier 5 — Board OM Script Verification

Run each model independently through a Python script via SSH. Each script
creates `RuntimeProviders` with `AclRuntimeManager`, loads the model through
`ModelRuntimeHandle`, and executes one inference.

### Board OM script template

```python
# Save as /IB_Robot/rerun_board_om_native.py
import sys, time, traceback
from pathlib import Path
import numpy as np
from inference_manifest import load_inference_manifest
from inference_service.backends.ascend.acl_runtime import AclRuntimeManager
from inference_service.backends.admission import ResourceDomainAdmissions
from inference_service.backends.types import RuntimeContext
from inference_service.model_sessions.ascend import AscendOmModelSession
from inference_service.runtime_composition import RuntimeProviders
from inference_service.unified_runtime import (
    ExecutionContext, ExecutionContract, ModelRequest,
    ModelRuntimeHandle, RuntimeAssembly,
)
from perception_service.ram_plus_adapter import RAMPlusAdapter
from perception_service.semantic_model_adapters import SAM2PromptAdapter, SigLIP2ImageAdapter, SigLIP2TextAdapter
from perception_service.siglip2_ascend_session import SigLIP2AscendSession

IMAGE = np.zeros((480, 640, 3), np.uint8)
IMAGE[120:360, 180:460] = [220, 40, 30]
MASK = np.zeros((480, 640), np.uint8)
MASK[120:360, 180:460] = 1
LABELS = ["red block", "banana", "blue cup"]

def _handle(session, context):
    return ModelRuntimeHandle(RuntimeAssembly(
        runtime_executor=session, session=session,
        execution_contract=ExecutionContract(), load_context=context))

def run_act(providers):
    # ... load ACT ascend_310b1, execute with zero-valued batched inputs
    # Expected: action shape (1, 100, 6), finite=True, infer ~80-90ms

def run_ram(providers):
    # ... load RAM++ ascend_310b, execute with IMAGE
    # Expected: logits (1, 4585), 6 tags, infer ~420-425ms

def run_sam(providers):
    # ... load SAM2 ascend_310b, execute with box prompt
    # Expected: logits (1, 1, 256, 256), mask (480, 640), infer ~700ms

def run_siglip(providers):
    # ... load SigLIP2 ascend_310b, execute image + text
    # Expected: embeddings (1, 1152) + (3, 1152), norm=1.0

mode = sys.argv[1]
providers = RuntimeProviders.create(AclRuntimeManager(), ResourceDomainAdmissions())
try: {"act": run_act, "ram": run_ram, "sam": run_sam, "siglip": run_siglip}[mode](providers)
finally: providers.close()
```

### Run commands

```bash
ssh OPi_20T "cd /IB_Robot && source .shrc_local && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && \
  python3 rerun_board_om_native.py act"

ssh OPi_20T "cd /IB_Robot && source .shrc_local && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && \
  python3 rerun_board_om_native.py ram"

ssh OPi_20T "cd /IB_Robot && source .shrc_local && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && \
  python3 rerun_board_om_native.py sam"

ssh OPi_20T "cd /IB_Robot && source .shrc_local && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && \
  python3 rerun_board_om_native.py siglip"
```

Pass criteria: `status=passed` with correct output shapes and `finite=True`.

## Tier 6 — Board ROS Mock (OM)

### Generate board mock YAML

Copy the production `so101_single_arm.yaml`, override:

- `simulation.platform: mock`
- `control_modes.model_inference.inference.pipelines.policy.deployment: ascend_310b1`
- `perception_services.services`: RAM++ and SigLIP2 with `deployment: ascend_310b`
- Disable `voice_tts` and `scheduler_enabled`

```python
import yaml, copy
base = yaml.safe_load(open("/IB_Robot/install/share/robot_config/config/robots/so101_single_arm.yaml"))
c = copy.deepcopy(base)
c["robot"]["simulation"]["platform"] = "mock"
c["robot"]["simulation"]["scene"] = None
c["robot"]["default_control_mode"] = "model_inference"
c["robot"]["control_modes"]["model_inference"]["scheduler_enabled"] = False
c["robot"]["control_modes"]["model_inference"]["inference"]["pipelines"]["policy"]["model_path"] = "/IB_Robot/models/ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515"
c["robot"]["control_modes"]["model_inference"]["inference"]["pipelines"]["policy"]["deployment"] = "ascend_310b1"
c["robot"]["voice_tts"]["enabled"] = False
c["robot"]["perception_services"] = {"services": [
    {"id": "ram_plus_tags", "enabled": True, "required": True,
     "bundle_path": "/IB_Robot/models/ram_plus_swin_large_14m", "deployment": "ascend_310b",
     "adapter_class": "perception_service.model_service_plugins:RAMPlusRecognizeTagsPlugin",
     "service_type": "ibrobot_msgs/srv/RecognizeTags",
     "endpoint": "/perception/ram_plus/recognize_tags", "node_name": "ram_plus_tags",
     "runtime_options": {"device_id": 0}},
    {"id": "siglip2_image", "enabled": True, "required": True,
     "bundle_path": "/IB_Robot/models/siglip2_so400m_patch14_384", "deployment": "ascend_310b",
     "adapter_class": "perception_service.model_service_plugins:SigLIP2EncodeEmbeddingsPlugin",
     "service_type": "ibrobot_msgs/srv/EncodeEmbeddings",
     "endpoint": "/perception/siglip2/encode_embeddings", "node_name": "siglip2_image",
     "runtime_options": {"device_id": 0}},
]}
with open("/IB_Robot/board_mock_verify.yaml", "w") as f:
    yaml.dump(c, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
```

### Launch on board

```bash
ssh OPi_20T "cd /IB_Robot && export ROS_DOMAIN_ID=51 && source .shrc_local && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && \
  source install/setup.bash 2>/dev/null && \
  ros2 launch robot_config robot.launch.py \
    config_path:=/IB_Robot/board_mock_verify.yaml \
    control_mode:=model_inference use_sim:=true"
```

### Call typed services from board

Use the same Python `rclpy` service-call pattern as host Tier 4, but run via
SSH on the board. Use `ROS_DOMAIN_ID=51` for board sessions.

Pass criteria:
- `contract_mock active`
- `Unified pipeline started: ... backend=ascend`
- RAM++ service: `success=True, tags>0, state=ready`
- SigLIP2 service: `success=True, emb_dim=1152, state=ready`

## Tier 7 — Board Torch NPU (Optional)

Only required when LeRobot patch stack or `lerobot_torch.py` changes affect
the NPU path.

```bash
ssh OPi_20T "cd /IB_Robot && source .shrc_local && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && \
  pip install decorator -q 2>/dev/null && \
  ASCEND_LAUNCH_BLOCKING=1 timeout 900 python3 rerun_board_npu_native.py"
```

The script loads `LeRobotTorchModelSession("npu")` with `model_dtype=fp16`,
runs `predict_action_chunk`, and checks output shape `(100, 6)`.

Known issues:
- Load takes ~145 s; inference takes ~610 s under blocking mode.
- CANN tiling warnings are expected; this is a functional pass, not a
  performance baseline.
- If input has wrong batch dimension, `conv2d` will report 5D shape error.
  Ensure preprocessor output is already batched before passing to the session.

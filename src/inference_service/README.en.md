# Inference Service

`inference_service` is the core AI execution package for IB-Robot. It provides a standardized framework for running end-to-end Machine Learning policies (like ACT, pi0, etc.) on physical robots with strict temporal alignment and zero-copy latency optimizations.

## Architecture: Composition over Inheritance

The inference pipeline is decoupled into three pure-Python core components (`inference_service.core`):
1. **TensorPreprocessor**: Converts raw ROS 2 sensor data (images, joint states) into normalized PyTorch Tensors.
2. **PureInferenceEngine**: A completely stateless, ROS-agnostic GPU execution engine.
3. **TensorPostprocessor**: Denormalizes output action tensors back into physical control commands.

By separating the core math from the ROS 2 transport layer, this package supports two distinct deployment modes, toggleable via a single YAML parameter.

---

## 🚀 Execution Modes

### Mode A: Monolithic (Single-Machine Zero-Copy)
**Best for**: Robots equipped with high-performance onboard GPUs (e.g., RTX 4060).

In this mode, `lerobot_policy_node.py` instantiates an `InferenceCoordinator` that chains the Preprocessor, Engine, and Postprocessor together.
* **Data Flow**: Sensor data stays entirely within the RAM/VRAM of the single process. Tensors are passed by reference.
* **Performance**: Absolute lowest latency. Zero serialization/deserialization overhead.
* **Config**: `execution_mode: "monolithic"`

### Mode B: Device-Edge-Cloud Synergy (Distributed)
**Best for**: Lightweight robots (Device) running on low-power CPUs (e.g., Raspberry Pi) paired with a high-end computation node (Edge) or tower server (Cloud) over a LAN.

To preserve compatibility with the pull-based `action_dispatch` system without clogging the network with 30fps video streams, the Device node acts as an **Asynchronous Proxy**.
1. **Device Node (`lerobot_policy_node.py`)**: Receives the action goal, reads the cameras *on-demand*, runs the **Preprocessor** on CPU, and publishes the lightweight Tensor batch to `/preprocessed/batch`. The action callback is then suspended using an asynchronous `threading.Event`.
2. **Edge/Cloud Node (`pure_inference_node.py`)**: Subscribes to the batch, crunches the numbers on the GPU using `PureInferenceEngine`, and returns the raw action to `/inference/action`.
3. **Device Node**: Wakes up, runs the **Postprocessor**, and completes the Action sequence.

* **Performance**: Achieves "Compute Offloading" perfectly. The Device only sends the exact frames needed for inference (e.g., 20Hz), saving massive network bandwidth.
* **Config**: `execution_mode: "distributed"`

```
Device Machine (Robot / Sim)               GPU Machine (Edge/Cloud)
┌──────────────────────────────┐          ┌──────────────────────────┐
│  action_dispatcher_node      │          │                          │
│       ↓                      │          │  pure_inference_node     │
│  lerobot_policy_node (Proxy) │          │  ├─ Subscribe            │
│  ├─ TensorPreprocessor (CPU) │          │  │  /preprocessed/batch  │
│  ├─ threading.Event          │          │  ├─ PureInferenceEngine  │
│  └─ TensorPostprocessor(CPU) │          │  │  (GPU)                │
│       ↓ Pub        ↑ Sub     │          │  └─ Publish              │
│  /preprocessed  /inference   │          │     /inference/action    │
│  /batch         /action      │          │                          │
└──────────┬──────────┬────────┘          └───────┬──────────┬───────┘
           │          │      LAN (same ROS_DOMAIN_ID)        │          │
           └──────────┴──────────────────────────┴──────────┘
```

---

## ⚙️ Configuration & Usage

The execution mode is controlled seamlessly via your `robot_config` YAML files. You do not need to change launch files to switch modes on the device.

```yaml
# src/robot_config/config/robots/your_robot.yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      execution_mode: "distributed"  # Or "monolithic"
      model: so101_act
```

### Launching

#### Scenario 1: Cross-Machine Distributed Deployment (Recommended for Production)

Both machines must share the **same `ROS_DOMAIN_ID`** and be on the same LAN.

**Step 1 — On the Robot (Device)**:

The Device launches only the Edge proxy node (pre/post-processing), without loading GPU models:
```bash
export ROS_DOMAIN_ID=<your_domain_id>
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    execution_mode:=distributed \
    use_sim:=true   # For simulation; omit for real hardware
```

**Step 2 — On the GPU Server (Edge/Cloud)**:

```bash
export ROS_DOMAIN_ID=<same_domain_id_as_device>
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/path/to/models/pretrained_model \
    device:=cuda
```

For models exported through the ATC/SVP toolchain, `inference_service` can own
the OM wrappers migrated from the original LeRobot patches:

```bash
# Generic Ascend ACL .om backend
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/path/to/pretrained_model \
    device:=ascend_om

# SD3403 worker binary protocol backend
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/path/to/pretrained_model \
    device:=ascend_om_3403
```

`device:=ascend_om` resolves the single OM model from `artifacts.policy` in
`policy_path/config.om.json`. `device:=ascend_om_3403` requires both
`artifacts.policy` and `artifacts.worker`, with `execution` set to
`["policy", "worker"]`. Preprocessing, postprocessing, ROS topics, and
distributed transport remain the existing `inference_service` pipeline.

For RK3588 / OpenHarmony boards running RKNN Lite, switch the cloud node to:

```bash
export ROS_DOMAIN_ID=<same_domain_id_as_device>
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/path/to/models/pretrained_model \
    device:=rknn
```

`device:=rknn` still uses the LeRobot metadata under `policy_path/config.json`
for preprocessing and postprocessing, while expecting the actual RKNN artifact to
live inside `policy_path`, using `model.rknn` as the default filename.

### Compiled Model Backend Boundaries

`PureInferenceEngine` still depends only on the common `PolicyWrapper` interface.
For `device:=ascend_om`, `device:=ascend_om_3403`, and `device:=rknn`, it uses an
internal `CompiledPolicyWrapper` facade:

- `ACTCompiledAdapter` and `PI05CompiledAdapter` read `type`, `input_features`,
  and `output_features` from `config.json` and own model-family input ordering,
  image/language inputs, action chunk decoding, `policy_type`, and chunk-size
  semantics.
- `OMRuntimeSession`, `PI05OMRuntimeSession`, `SD3403RuntimeSession`, and
  `RKNNRuntimeSession` own backend artifact resolution, runtime loading,
  execution, and resource cleanup.
- `policy_type` identifies the model family, such as `act`; `backend_type`
  identifies the runtime backend, such as `ascend_om`, `ascend_om_3403`, or
  `rknn`.

The runtime device comes from launch/ROS parameters. If the LeRobot-exported
`config.json` still records the training device, such as `"device": "cuda"`,
`inference_service` overrides that field in a local temporary config copy with
the current runtime tensor device (CPU for compiled OM/RKNN backends). The
source model directory is not modified, and training-device metadata does not
constrain backend selection.

Compiled conversion tools should emit a separate `config.om.json` next to the
LeRobot `config.json` so compiled runtime metadata does not pollute the LeRobot
schema. The sidecar uses a role-to-path artifact map and can declare a generic
serial pipeline through `execution`:

```json
{
  "schema_version": 1,
  "policy_type": "pi05",
  "backend": "ascend_om",
  "artifact_dir": "om",
  "artifacts": {
    "vlm": "vlm.om",
    "action_expert": "action_expert.om"
  },
  "execution": ["vlm", "action_expert"]
}
```

Single-OM ACT policies use `artifacts.policy` with `execution: ["policy"]`. OM
artifacts are no longer read from LeRobot `config.json`, environment variables,
or directory guesses; conversion tools must generate `config.om.json`.

Compiled backend dependencies remain lazily loaded. Ascend ACL, PI05 OM, the
SD3403 worker stack, and RKNNLite are imported only when the matching backend is
loaded. ROS topics, preprocessing, postprocessing, and launch arguments stay
unchanged.

#### Scenario 2: Single-Machine Debug (Development)

Run both Edge + Cloud nodes on one machine by adding `cloud_local:=true`:

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    execution_mode:=distributed \
    use_sim:=true \
    cloud_local:=true
```

### Verifying Distributed Mode

```bash
# 1. Confirm both inference nodes are online
ros2 node list | grep -E 'act_inference|pure_inference'
# Expected:
#   /act_inference_node      ← Edge (pre/post-processing)
#   /pure_inference           ← Cloud (GPU inference)

# 2. Confirm distributed topics exist
ros2 topic list | grep -E 'preprocessed|inference/action'
# Expected:
#   /preprocessed/batch      ← Edge → Cloud
#   /inference/action         ← Cloud → Edge

# 3. Monitor inference frequency
ros2 topic hz /inference/action
```

### Logging Reference

After launch, each node prints key lifecycle messages for quick status diagnosis:

| Node | Example Log | Meaning |
|------|-------------|---------|
| `pure_inference` | `Waiting for preprocessed batches from edge node...` | Cloud node ready, waiting for Edge data |
| `pure_inference` | `✓ First inference completed: latency=XXms` | First inference succeeded, end-to-end link confirmed |
| `pure_inference` | `[stats] count=XX, avg=XXms, last=XXms` | Performance stats every 5 seconds |
| `act_inference_node` | `✓ First inference complete (distributed): total=XXms` | Edge node completed full inference round-trip |
| `action_dispatcher` | `✓ First inference received: chunk=XX, latency=XXms` | Dispatcher received first executable actions |
| `action_dispatcher` | `[stats] inferences=XX, avg_latency=XXms, queue=XX, hold=XX` | Dispatch stats every 5s; `hold` = times queue exhausted and last frame held |

---

## 🧪 Testing
Because the core components are isolated from ROS, they can be validated entirely offline:
```bash
pytest src/inference_service/tests/
```

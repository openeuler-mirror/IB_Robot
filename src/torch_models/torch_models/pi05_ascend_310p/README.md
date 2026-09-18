# PI0.5 Ascend310P

This directory owns the repository-native PI0.5 implementation for Ascend310P.
It contains the NPU-safe PI0.5 model, local SigLIP tower, fixed ten-step
prefix/denoise TorchAir graphs, internal-format preparation, persistent graph
cache, strict no-initialization checkpoint loading, and the runtime provider.

The provider is selected by the stable runtime identity:

```text
model_type: pi05
backend:    torch
device:     npu
```

No model-specific `architecture_class` or packaging flag is required. The
normal deployment name remains `torch-npu`:

```bash
source .shrc_local
ros2 run model_utils package-torch-deployment \
  --bundle-root /path/to/pi05-bundle \
  --devices npu
```

The same bundle may contain CPU/CUDA native Torch deployments and compiled
Ascend OM deployments. CPU/CUDA use the existing LeRobot policy factory; OM
uses the existing Ascend session and its VLM/Action Expert artifacts. Only the
native `(pi05, torch, npu)` identity resolves this provider.

## Environment and Execution

On CANN 8.1, both `./scripts/setup.sh --profile full` and
`./scripts/setup.sh --profile inference` select the same compatibility runtime:
LeRobot 0.6.0, Torch/Torch-NPU 2.5.1, TorchVision 0.20.1, and Transformers
5.3.0. Full adds functionality without selecting a different Transformers.
The shared `requirements/lerobot-v0.6-cann-8.1-compat.txt` and CANN constraints
preserve this combination; the model provider still validates it before loading
weights. Newer Transformers compatibility is separate model-adaptation work.

Set the robot's inference pipeline to the normal bundle and deployment:

```yaml
model_path: models/pi05/pi05-doublecam-fp32/019200-torch-npu
deployment: torch-npu
runtime_options:
  model_dtype: fp16
```

The `so101_pi05_ascend_310p_mock` profile supplies this pipeline and mock
observations. From the workspace root, after building:

```bash
source .shrc_local
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_pi05_ascend_310p_mock \
  use_sim:=true control_mode:=model_inference
```

The provider validates the local tokenizer, the fixed `num_inference_steps=10`
contract, Transformers `5.3.0`, and the physical Ascend310P SKU before loading
weights. On 310P, the default graph boundary is compiled vision plus compiled
prefix prefill and a fixed ten-step denoise graph.

Diagnostic switches:

- `LEROBOT_PI05_COMPILE_VISION_EMBED=0` selects eager vision only for diagnosis.
- `IBROBOT_PI05_GRAPH_COMPILE=0` disables graph compilation for isolation tests.
- `IBROBOT_PI05_NPU_FUSED_OPS=0` disables NPU fused operations for isolation tests.
- `IBROBOT_PI05_STAGE_TIMING=1` adds synchronized prefix/denoise timings to result metadata.

The graph/fused-op overrides are intended for diagnostics. Do not use
synchronized stage timing in formal latency measurements.

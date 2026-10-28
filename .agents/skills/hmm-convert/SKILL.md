---
name: hmm-convert
description: "Package PI0.5 or SmolVLA models as Houmo HMM (.hmm) deployments for XH2 NPU (LQ50 / M50). Use when users mention 'convert to hmm', 'houmo', 'xh2', '后摩', 'tcim', 'xhquant', 'pi05 hmm', 'smolvla hmm', 'LQ50', 'M50', 'HMM 模型转换', 'w8a8', 'sefp', 'hmonnx', or package-hmm-deployment. Explicitly reject ACT HMM requests and route ACT to torch, ascend, hisilicon, or rknn."
---

# HMM Model Conversion Skill

Package Houmo compiler outputs for PI0.5 and SmolVLA into strict IB-Robot HMM deployments for XH2 NPU (LQ50 / M50).

## Supported Matrix

| Policy | HMM support | Graph |
|--------|-------------|-------|
| PI0.5 | Supported | vision + embedding + prefill + action projections + time MLP + decode |
| SmolVLA | Supported | vision + embedding/state projection + prefill + action |
| ACT | Unsupported | Reject; use `torch`, `ascend`, `hisilicon`, or `rknn` |

Never restore or recommend the removed ACT single-module HMM path, `config.hmm.json`, `HMM_MODEL_PATH`, directory scanning, `HMMRuntimeSession`, or `device:=hmm`.

## Internal References

Read only the references needed for the current step:

| Purpose | Reference |
|---------|-----------|
| PI0.5 / SmolVLA 的 compiler 输出 role 集合和 v1.3.0 仓库导出校验项 | `references/compiler-outputs.md` |
| PI0.5 / SmolVLA 的 packaging spec JSON 示例 | `references/packaging-spec.md` |

Do not expose these references as separate skills.

## Required Workflow

1. Use Houmo's current `houmo-examples-xh2` workflow to export, quantize, and compile the supported policy modules（详见 `references/compiler-outputs.md`）。
2. Write every conversion intermediate — ONNX, HMONNX, TCIM build caches, calibration data, `provenance.json` — to `models/_work/<bundle>/`, never inside a bundle. `convert_hmm.sh` does this by default (`SMOLVLA_HMM_WORK` / `PI05_HMM_WORK` overrides); manual runs must pass `--output-dir models/_work/...` plus `--bundle-root models/<bundle>`.
3. Preserve the compiler-emitted TCIM `model.json` next to each `.hmm`; it is the source of runtime names, indices, dtypes, and shapes.
4. Keep the original LeRobot bundle metadata read-only: `config.json`, processor JSON/state, tokenizer assets, and policy metadata.
5. Create a path-only packaging spec for `package-hmm-deployment`（详见 `references/packaging-spec.md`）。
6. Run the packager to write or update the bundle's only deployment manifest: `inference_manifest.json`. The bundle must end up containing only manifest-referenced artifacts plus LeRobot metadata.
7. Configure a named inference pipeline that selects the generated deployment.
8. Validate on the target or approved TCIM mocks.

Bundle hygiene: before releasing or archiving a bundle, verify it contains no `onnx/`, `hmonnx/`, `calibration/`, `tcim/`, or `model_utils_work/` residue — those belong under `models/_work/` and can be archived or deleted independently.

The packager owns artifact generations, bindings, execution order, device links, UUID/revision updates, lightweight structural identities, and strict-loader verification. Do not hand-edit generated manifest identities or tensor bindings.

## Environment

The Houmo compiler toolchain may conflict with the IB-Robot Python environment. Use a dedicated venv or Houmo's current 24.04 toolchain image. Do not install `xhquant` or `tcim` into the main venv.

```bash
python3 -m venv .venv-hmm
source .venv-hmm/bin/activate
pip install xhquant tcim onnx onnxsim torch
```

Use `source .shrc_local` before running IB-Robot packaging, manifest validation, ROS, or project commands.

## Run The Packager

```bash
source .shrc_local
package-hmm-deployment \
    --bundle-root models/<policy_bundle> \
    --deployment hmm_lq50 \
    --target-soc lq50 \
    --target-runtime tcim \
    --spec /path/to/hmm-package-spec.json
```

Source form:

```bash
source .shrc_local
python3 -m model_utils.hmm_export \
    --bundle-root models/<policy_bundle> \
    --deployment hmm_lq50 \
    --target-soc lq50 \
    --target-runtime tcim \
    --spec /path/to/hmm-package-spec.json
```

Success means `<bundle-root>/inference_manifest.json` was written and accepted by the production strict loader.

## Named Pipeline Configuration

```yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      pipelines:
        policy:
          model_path: models/<policy_bundle>
          deployment: hmm_lq50
          execution_mode: monolithic
          request_timeout: 10.0
          default_task: "pick up the object"
    executor:
      type: topic
      mode: model_inference
      inference_pipeline: policy
```

Do not configure backend-valued `device`, a global model table, or a per-backend manifest filename.

## Validation

Run focused tests after changing HMM packaging or docs:

```bash
source .shrc_local
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/src/inference_manifest:$PWD/src/inference_service:$PWD/src/model_utils:$PYTHONPATH" \
pytest -q \
    src/model_utils/test/test_hmm_export.py \
    src/inference_service/test/test_hmm_backend.py \
    src/inference_service/test/test_backend_contract.py
```

## Troubleshooting

### ACT HMM request

Reject the request as unsupported. Offer ACT deployments on `torch`, `ascend`, `hisilicon`, or `rknn`; do not provide a compatibility workflow.

### TCIM ABI mismatch

Use the `model.json` emitted for the exact `.hmm` file. Do not infer tensor order from Python dictionaries or copy ABI metadata from another build.

### Manifest hash mismatch

Rerun `package-hmm-deployment`. Do not edit artifact identity, revision, or bundle digest manually.

### External tokenizer dependency

Vendor tokenizer/processor assets into the bundle and update the LeRobot metadata to reference the local directory before packaging.

### Board runtime initialization

Read `oh-constraints` first. Source the RoboFrame environment and `scripts/setup/houmo_hmm_env.sh`; native Houmo 1.3 uses `TCIM_BACKEND=xh2` and `HOUMO_TARGET=xh2`. `Xh2HalBackend` is only for the legacy runtime.

## References

- `docs/Houmo_HMM_Conversion.md`
- `src/model_utils/model_utils/hmm_export.py`
- `src/model_utils/model_utils/pi05_export/`
- `src/model_utils/model_utils/inference_manifest_export.py`
- `src/inference_service/inference_service/backends/hmm/backend.py`
- `src/model_utils/test/test_hmm_export.py`
- `src/inference_service/test/test_hmm_backend.py`

## When To Use This Skill

Use for:

- PI0.5 or SmolVLA Houmo HMM conversion and packaging
- `xhquant`, `tcim`, `hmonnx`, LQ50, M50, or XH2 compiler troubleshooting
- `package-hmm-deployment` specs and unified manifest generation
- HMM execution roles, bindings, embedding artifacts, state projection, or device links

Do not use for:

- ACT HMM conversion; it is unsupported
- RKNN conversion; use `rknn-convert`
- Ascend OM conversion; use the Ascend exporter workflow
- General OpenHarmony board operations; read `oh-constraints` and use the relevant board skill

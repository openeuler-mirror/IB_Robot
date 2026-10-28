---
name: rknn-convert
description: "Convert ACT or SmolVLA models to RKNN deployments for RK3588. Use when users mention 'rknn', 'RK3588', 'rknn-toolkit2', 'convert to rknn', 'RKNN deployment', '模型转换', 'rknn转换', or 'NPU 推理'. Covers isolated ONNX-to-RKNN compilation, compiler ABI metadata, unified inference_manifest.json packaging, and named pipeline configuration."
---

# RKNN Model Conversion Skill

Compile ACT or SmolVLA ONNX graphs with `rknn-toolkit2`, then package the compiler outputs as a
strict IB-Robot RKNN deployment for RK3588.

## Supported Matrix

| Policy | RKNN support | Graph |
|--------|--------------|-------|
| ACT | Supported | Single `policy` RKNN artifact |
| SmolVLA | Supported | `vision -> embedding -> prefill -> action` |
| PI0.5 | Unsupported | Use `torch`, `ascend`, or `hmm` |

The deployable result is not a standalone `.rknn` file. It is a LeRobot policy bundle containing
compiler ABI metadata, packaged artifacts, and one named deployment in the bundle's only
`inference_manifest.json`.

Never restore or recommend `config.rknn.json`, `RKNN_MODEL_PATH`, directory scanning, backend-valued
`device`, or a per-backend manifest filename.

## Internal References

Read only the references needed for the current scenario:

| Purpose | Reference |
|---------|-----------|
| ACT high-level/low-level 编译命令、SmolVLA 三模块导出与 package-only 模式 | `references/policy-workflows.md` |

Do not expose these references as separate skills.

## Required Workflow

1. Export ONNX from the original LeRobot bundle in the main workspace environment. Intermediates go to `models/_work/<bundle>/rknn/` (exporter default), never inside the bundle.
2. Compile ONNX to RKNN in the isolated `.venv-rknn` environment.
3. Preserve the compiler-emitted `*.rknn.abi.json`; it is the source of runtime tensor names,
   indices, dtypes, shapes, and image layouts.
4. Package the RKNN artifacts and ABI bindings into the original bundle: pass `--bundle_root models/<policy_bundle>` with the work directory as `--output_dir`. The bundle receives only manifest-referenced artifacts and LeRobot metadata.
5. Write or update the named deployment in `<bundle-root>/inference_manifest.json`.
6. Configure a named inference pipeline that selects that deployment.
7. Validate the strict manifest loader and, when available, the RK3588 runtime.

The exporter or packager owns artifact generations, bindings, UUID/revision updates, lightweight bundle digest,
and strict loader verification. Do not hand-edit generated identities or tensor bindings.

ACT 和 SmolVLA 的具体导出与编译命令见 `references/policy-workflows.md`。

## Environment Split

`rknn-toolkit2` requires `torch<=2.4.0` and `numpy<=1.26.4`, which conflicts with the main LeRobot
environment. Keep model export and RKNN compilation separate.

| Task | Environment |
|------|-------------|
| Export LeRobot checkpoint to ONNX | Main workspace after `source .shrc_local` |
| Compile ONNX to RKNN | Dedicated `.venv-rknn` in a clean shell |
| Package or validate a deployment | Main workspace after `source .shrc_local` |
| Run board inference | Board runtime using `rknnlite` or `librknnrt.so` |

Do not install `rknn-toolkit2` into the main venv. Do not source `.shrc_local` before invoking the
`.venv-rknn` compiler step in the same shell.

### Create The Dedicated Environment

```bash
python3 -m venv .venv-rknn
.venv-rknn/bin/pip install rknn-toolkit2 onnx onnxruntime
```

The repository conversion helper patches the `onnx.mapping` compatibility break used by
`rknn-toolkit2==2.3.2` with newer ONNX releases.

## Generic Packaging

`package-compiled-deployment` is available when vendor compilation is separate from the policy
exporter. Its JSON spec must define execution order, artifact formats, compiler ABI files, semantic
bindings, and image layouts. For an ACT RKNN deployment, use one `policy` role with format `rknn`,
runtime ABI format, semantic inputs matching `config.json`, and output semantic `action`.

```bash
source .shrc_local
package-compiled-deployment \
    --bundle-root models/<policy_bundle> \
    --deployment rknn_rk3588 \
    --backend rknn \
    --target-soc rk3588 \
    --target-runtime rknn-lite2 \
    --spec /path/to/rknn-package-spec.json
```

Success means `<bundle-root>/inference_manifest.json` was written and accepted by the production
strict loader.

## Conversion Modes

| Mode | Accuracy | Typical use |
|------|----------|-------------|
| `float16` | Highest | ACT and Transformer-based models; default recommendation |
| `int8` | Calibration-dependent | CNN-heavy models with representative calibration data |
| `hybrid` | Mixed | Graphs where full INT8 causes unacceptable loss |

Do not choose INT8 only for file size. Validate representative outputs against the source model.

## Named Pipeline Configuration

```yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      pipelines:
        policy:
          model_path: models/<policy_bundle>
          deployment: rknn_rk3588
          execution_mode: monolithic
          request_timeout: 10.0
          default_task: "pick up the object"
    executor:
      type: topic
      mode: model_inference
      inference_pipeline: policy
```

Do not configure a global model table, backend-valued `device`, or a per-backend manifest path.

## Validation

Run focused tests after changing RKNN export, packaging, or this workflow:

```bash
source .shrc_local
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/src/inference_manifest:$PWD/src/inference_service:$PWD/src/model_utils:$PYTHONPATH" \
pytest -q \
    src/model_utils/test/test_export_onnx_rknn.py \
    src/model_utils/test/test_smolvla_rknn_export.py \
    src/inference_service/test/test_rknn_backend.py \
    src/inference_service/test/test_backend_contract.py
```

Before board deployment, load the generated deployment through the strict manifest loader and
confirm that every artifact hash, tensor index, dtype, shape, and image layout matches the exact
compiler output.

## Board Deployment

Read `oh-constraints` before OpenHarmony board work, then use the repository RKNN inference guide.
The board consumes the packaged bundle and selected deployment, not an arbitrary `.rknn` path.

Runtime options include:

- `rknn-toolkit-lite2` / `rknnlite` for Python runtimes
- `librknnrt.so` for native runtimes

Reference: `docs/OpenHarmony_EmbodiedAI_RKNN_Inference.md`.

## Troubleshooting

### RKNN ABI JSON is missing

The artifact is not packageable. Re-run the repository conversion helper or compiler integration
that emits `*.rknn.abi.json`; do not infer tensor metadata from ONNX or `config.json` alone.

### Runtime inputs do not match policy inputs

Use the ABI emitted for the exact RKNN file. ACT inputs must match the ordered semantic runtime
inputs from `config.json`; unrelated feature keys are not valid model inputs.

### Image layout is absent or invalid

Every image input in the RKNN ABI must declare `NCHW` or `NHWC`. Fix the compiler ABI export rather
than adding a guessed layout to the manifest.

### Manifest hash mismatch

Rerun the exporter or packager after replacing an artifact. Do not edit artifact identity, revision, or bundle
digest values manually.

### Torch or NumPy version conflict

The compiler ran in the main LeRobot environment. Use `.venv-rknn` only for ONNX-to-RKNN conversion
and return to the sourced main workspace for packaging and validation.

### `onnx.mapping` AttributeError

Use `.agents/skills/rknn-convert/convert_to_rknn.py`, which carries the compatibility patch required
by `rknn-toolkit2==2.3.2`.

### Unsupported NPU operators

Inspect RKNN build logs for CPU fallback or unsupported operators. Prefer `float16` for ACT and
Transformer graphs, then validate numerical outputs and target latency on RK3588.

## References

- `.agents/skills/rknn-convert/convert_to_rknn.py`
- `src/model_utils/model_utils/export_onnx_rknn.py`
- `src/model_utils/model_utils/smolvla_export/export_rknn_modules.py`
- `src/model_utils/model_utils/package_compiled_deployment.py`
- `src/model_utils/model_utils/inference_manifest_export.py`
- `src/model_utils/test/test_export_onnx_rknn.py`
- `src/model_utils/test/test_smolvla_rknn_export.py`
- `src/inference_service/test/test_rknn_backend.py`

## When To Use This Skill

Use for:

- ACT or SmolVLA ONNX-to-RKNN compilation
- RK3588 and `rknn-toolkit2` compatibility troubleshooting
- RKNN compiler ABI generation and validation
- Unified RKNN deployment packaging and named pipeline configuration

Do not use for:

- PI0.5 deployment; use `torch`, `ascend`, or `hmm`
- Ascend OM conversion; use the Ascend exporter workflow
- Hisilicon SD3403 conversion; use the Hisilicon exporter and generic compiled packager
- General OpenHarmony board operations; read `oh-constraints` and use the relevant board skill

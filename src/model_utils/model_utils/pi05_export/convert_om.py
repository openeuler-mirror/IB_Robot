# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Convert exported PI05 ONNX (VLM and/or Action Expert) to Ascend ``.om``.

This is the missing piece that puts PI05 on par with the single-OM ACT flow
(``export_onnx_atc.py``): it wraps the ``atc`` compiler so users no longer have
to hand-write the command or construct the unified deployment manifest.

By default, ATC is invoked without ``--input_shape`` and with
``--precision_mode_v2=origin``. If a board/toolkit needs an explicit shape,
``--input-shape auto`` derives it from each ONNX graph's static inputs. The
manifest is finalized only when both roles and both ONNX ABIs are available.

Examples
--------
Convert both segments and write the manifest::

    python -m model_utils.pi05_export.convert_om \\
        --pretrained-policy-path /path/to/pi05_ckpt \\
        --soc-version Ascend310P3 \\
        --vlm-onnx   outputs/onnx/pi05-vlm_op17_nodyn_fp16_cpu.onnx \\
        --ae-onnx    outputs/onnx/pi05-action_expert_op17_nodyn_fp16_cpu.onnx

Convert only the VLM (e.g. after re-exporting just that segment)::

    python -m model_utils.pi05_export.convert_om \\
        --pretrained-policy-path /path/to/pi05_ckpt \\
        --soc-version Ascend310P3 \\
        --vlm-onnx outputs/onnx/pi05-vlm_op17_nodyn_fp16_cpu.onnx \\
        --skip-manifest
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from inference_manifest import (
    AscendRuntimeProfile,
    CompiledDeployment,
    DeploymentArtifact,
    DeploymentTarget,
    DeviceLink,
    ExecutionContract,
    RoleRuntimeProfile,
    load_inference_manifest,
)
from inference_manifest.json_utils import load_json_strict
from inference_service.pi05_schedule import load_pi05_schedule, uniform_pi05_schedule, write_pi05_schedule
from model_utils.acl_abi_inspection import write_acl_om_abi
from model_utils.inference_manifest_export import (
    artifact_bindings,
    deployment_artifact,
    load_published_artifact_index,
    package_deployment_artifact,
    read_runtime_abi,
    update_deployment,
    upsert_deployment,
)
from model_utils.pi05_export._cli_ui import Stage, print_summary, setup_logging

LOGGER = logging.getLogger("pi05_export.convert_om")


def _derive_input_shape(onnx_path: Path) -> str:
    """Build the ATC ``--input_shape`` string from an ONNX graph's static dims.

    Returns a string like ``name0:1,3,480,640;name1:1,200`` covering every
    non-initializer graph input. Raises if any input has a dynamic dim, since
    ATC needs fully static shapes for these single-shape OM artifacts.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    initializer_names = {init.name for init in model.graph.initializer}

    parts: list[str] = []
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        dims = inp.type.tensor_type.shape.dim
        sizes: list[int] = []
        for d in dims:
            if not d.HasField("dim_value") or d.dim_value <= 0:
                raise ValueError(
                    f"ONNX input {inp.name!r} in {onnx_path.name} has a dynamic/unknown dim; "
                    "re-export with fixed shapes before ATC conversion."
                )
            sizes.append(int(d.dim_value))
        parts.append(f"{inp.name}:{','.join(map(str, sizes))}")
    if not parts:
        raise ValueError(f"No graph inputs found in {onnx_path}")
    return ";".join(parts)


def _has_atc_arg(extra_args: list[str], name: str) -> bool:
    """Return True when raw ATC args already include ``--<name>``."""
    return any((token := arg.lstrip("-")) == name or token.startswith(f"{name}=") for arg in extra_args)


def _default_om_output(manifest_dir: Path, role: str) -> Path:
    """Default OM work output under ``models/_work`` so the bundle stays releasable."""

    for models_root in manifest_dir.parents:
        if models_root.name == "models":
            return models_root / "_work" / manifest_dir.relative_to(models_root) / "ascend" / "pi05" / f"{role}.om"
    return manifest_dir.parent / "_work" / manifest_dir.name / "ascend" / "pi05" / f"{role}.om"


def _run_atc(
    onnx_path: Path,
    om_output: Path,
    soc_version: str,
    *,
    extra_args: list[str],
    input_shape_mode: str,
) -> bool:
    """Invoke ``atc`` to compile a single ONNX to an OM. Returns success flag.

    ``atc`` appends ``.om`` itself, so ``--output`` is the path without suffix.
    The subprocess inherits stdout/stderr so the user sees ATC progress live.
    """
    onnx_path = onnx_path.expanduser().resolve(strict=True)
    output_stem = str(om_output.with_suffix(""))
    om_output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "atc",
        "--framework=5",
        f"--soc_version={soc_version}",
        f"--model={onnx_path}",
        f"--output={output_stem}",
    ]
    has_input_shape_arg = _has_atc_arg(extra_args, "input_shape")
    if input_shape_mode == "auto" and not has_input_shape_arg:
        input_shape = _derive_input_shape(onnx_path)
        LOGGER.info("  input_shape (auto): %s", input_shape)
        command.append(f"--input_shape={input_shape}")
    elif has_input_shape_arg:
        LOGGER.info("  input_shape: provided via --atc-arg")
    else:
        LOGGER.info("  input_shape: omitted")

    if not _has_atc_arg(extra_args, "precision_mode_v2"):
        command.append("--precision_mode_v2=origin")
    command.extend(extra_args)

    LOGGER.info("  $ %s", " ".join(command))
    # nosec B603 — command is built from validated paths / args, not shell.
    # CANN 8.1 resolves ONNX external-data locations relative to the process
    # working directory rather than the --model path.
    return subprocess.run(command, check=False, cwd=onnx_path.parent).returncode == 0


def convert_role(
    *,
    role: str,
    onnx_path: Path,
    om_output: Path,
    soc_version: str,
    extra_args: list[str],
    input_shape_mode: str,
    index: int,
    total: int,
) -> Path:
    """Compile one role's ONNX to OM.

    Returns the produced ``.om`` path. Raises on ATC failure so the orchestrator
    can stop while leaving any already-produced artifacts in place (resumable).
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"{role} ONNX not found: {onnx_path}")

    with Stage(f"ATC compile {role}", index=index, total=total):
        om_output.unlink(missing_ok=True)
        ok = _run_atc(onnx_path, om_output, soc_version, extra_args=extra_args, input_shape_mode=input_shape_mode)
        if not ok:
            raise RuntimeError(f"ATC failed for {role} ({onnx_path.name}); see ATC log above")
        if not om_output.is_file():
            raise RuntimeError(f"ATC reported success but {om_output} was not produced")

    return om_output


def write_pi05_ascend_deployment(
    bundle_root: Path,
    deployment_name: str,
    soc_version: str,
    vlm_abi_path: Path,
    vlm_om: Path,
    action_abi_path: Path,
    action_om: Path,
    schedule_file: Path | None = None,
    reuse_artifact_roles: frozenset[str] = frozenset(),
    *,
    prefer_hardlink: bool = False,
) -> Path:
    """Write one complete PI0.5 Ascend deployment from compiled runtime ABIs."""

    created_artifacts: list[Path] = []
    existing: CompiledDeployment | None = None
    reusable_artifacts: dict[str, DeploymentArtifact] = {}
    if reuse_artifact_roles:
        selected = load_inference_manifest(bundle_root, deployment_name)
        candidate = selected.deployment
        if not isinstance(candidate, CompiledDeployment) or candidate.backend != "ascend":
            raise ValueError(f"Cannot reuse PI0.5 artifacts from non-Ascend deployment {deployment_name!r}")
        if candidate.target.soc != soc_version:
            raise ValueError(
                f"Cannot reuse {candidate.target.soc!r} artifacts for requested target SoC {soc_version!r}"
            )
        existing = candidate
        for role in reuse_artifact_roles:
            try:
                reusable_artifacts[role] = candidate.artifacts[role]
            except KeyError as exc:
                raise ValueError(f"Deployment {deployment_name!r} has no reusable artifact role {role!r}") from exc

    vlm_abi = None if "vlm" in reuse_artifact_roles else read_runtime_abi(vlm_abi_path)
    action_abi = None if "action_expert" in reuse_artifact_roles else read_runtime_abi(action_abi_path)

    def semantic_name(runtime_name: str) -> str:
        known = {
            "lang_tokens",
            "lang_masks",
            "observation.language.tokens",
            "observation.language.attention_mask",
            "past_kv_tensor",
            "prefix_pad_masks",
            "time",
            "noise",
            "action",
            "velocity",
            "v_t",
        }
        return next((part for part in reversed(runtime_name.split(":")) if part in known), runtime_name)

    if action_abi is None:
        assert existing is not None
        action_bindings = existing.bindings["action_expert"]
        action_runtime_name = action_bindings.outputs[0].runtime_name or action_bindings.outputs[0].semantic
        action_output = semantic_name(action_runtime_name)
    elif len(action_abi.outputs) != 1:
        raise ValueError(f"PI0.5 Action Expert ABI must have exactly one output, got {len(action_abi.outputs)}")
    else:
        action_output = semantic_name(action_abi.outputs[0].name)
    if action_output not in {"action", "velocity", "v_t"}:
        output_name = action_abi.outputs[0].name if action_abi is not None else action_runtime_name
        raise ValueError(f"PI0.5 Action Expert ABI output must be 'action', 'velocity', or 'v_t', got {output_name!r}")
    velocity_mode = action_output in {"velocity", "v_t"}
    if not velocity_mode and schedule_file is not None:
        raise ValueError("--schedule-file is only valid for a velocity/v_t Action Expert ABI output")

    if vlm_abi is None:
        assert existing is not None
        vlm_bindings = existing.bindings["vlm"]
    else:
        vlm_input_semantics = {
            tensor.name: (
                "observation.language.tokens"
                if semantic_name(tensor.name) in {"lang_tokens", "observation.language.tokens"}
                else "observation.language.attention_mask"
                if semantic_name(tensor.name) in {"lang_masks", "observation.language.attention_mask"}
                else semantic_name(tensor.name)
            )
            for tensor in vlm_abi.inputs
        }
        vlm_bindings = artifact_bindings(
            vlm_abi,
            input_semantics=vlm_input_semantics,
            output_semantics={
                tensor.name: {
                    "past_kv_tensor": "internal.past_kv",
                    "prefix_pad_masks": "internal.prefix_pad_masks",
                }[semantic_name(tensor.name)]
                for tensor in vlm_abi.outputs
            },
            image_layouts={
                semantic: "NCHW"
                for semantic in vlm_input_semantics.values()
                if semantic.startswith("observation.images.")
                or semantic in {"prefix_att_2d_masks_4d", "observation.prefix_att_2d_masks_4d"}
            },
        )
    if action_abi is not None:
        action_bindings = artifact_bindings(
            action_abi,
            input_semantics={
                tensor.name: {
                    "past_kv_tensor": "internal.past_kv",
                    "prefix_pad_masks": "internal.prefix_pad_masks",
                    "time": "time",
                    "noise": "noise",
                }[semantic_name(tensor.name)]
                for tensor in action_abi.inputs
            },
            output_semantics={action_abi.outputs[0].name: "action"},
        )
    links = tuple(
        DeviceLink(
            semantic=semantic,
            producer="vlm",
            consumer="action_expert",
            transport="device_pointer",
            owner="producer",
            lifetime="inference",
        )
        for semantic in ("internal.past_kv", "internal.prefix_pad_masks")
    )

    published = load_published_artifact_index(bundle_root)

    def package(role: str, source: Path) -> Path:
        artifact = package_deployment_artifact(
            bundle_root,
            source,
            backend="ascend",
            deployment_name=deployment_name,
            role=role,
            force_copy=True,
            prefer_hardlink=prefer_hardlink,
            published=published,
        )
        created_artifacts.append(artifact)
        return artifact

    try:
        artifact_values = {
            "vlm": reusable_artifacts.get("vlm") or package("vlm", vlm_om),
            "action_expert": reusable_artifacts.get("action_expert") or package("action_expert", action_om),
        }
        if velocity_mode:
            if "denoising_schedule" in reusable_artifacts and schedule_file is None:
                artifact_values["denoising_schedule"] = reusable_artifacts["denoising_schedule"]
                return _upsert_pi05_deployment(
                    bundle_root,
                    deployment_name,
                    soc_version,
                    artifact_values,
                    vlm_bindings,
                    action_bindings,
                    links,
                )
            temporary_schedule: TemporaryDirectory[str] | None = None
            try:
                if schedule_file is not None:
                    schedule_source = Path(schedule_file).expanduser()
                    load_pi05_schedule(schedule_source)
                else:
                    config_path = bundle_root / "config.json"
                    config = load_json_strict(config_path)
                    num_inference_steps = config.get("num_inference_steps") if isinstance(config, dict) else None
                    schedule = uniform_pi05_schedule(num_inference_steps, name=f"uniform{num_inference_steps}")
                    temporary_schedule = TemporaryDirectory(prefix="pi05-schedule-")
                    schedule_source = write_pi05_schedule(
                        schedule,
                        Path(temporary_schedule.name) / "denoising_schedule.json",
                    )
                artifact_values["denoising_schedule"] = package("denoising_schedule", schedule_source)
            finally:
                if temporary_schedule is not None:
                    temporary_schedule.cleanup()
        return _upsert_pi05_deployment(
            bundle_root,
            deployment_name,
            soc_version,
            artifact_values,
            vlm_bindings,
            action_bindings,
            links,
        )
    except Exception:
        _remove_unreferenced_artifacts(bundle_root, created_artifacts)
        raise


def _remove_unreferenced_artifacts(bundle_root: Path, candidates: list[Path]) -> None:
    referenced: set[Path] = set()
    manifest_path = bundle_root / "inference_manifest.json"
    if manifest_path.is_file():
        manifest = load_json_strict(manifest_path)
        if isinstance(manifest, dict) and isinstance(manifest.get("deployments"), dict):
            referenced = {
                bundle_root.joinpath(*artifact["path"].split("/")).resolve()
                for deployment in manifest["deployments"].values()
                if isinstance(deployment, dict) and isinstance(deployment.get("artifacts"), dict)
                for artifact in deployment["artifacts"].values()
                if isinstance(artifact, dict) and isinstance(artifact.get("path"), str)
            }
    generations_root = (bundle_root / "artifacts").resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in referenced or not resolved.is_relative_to(generations_root):
            continue
        resolved.unlink(missing_ok=True)
        parent = resolved.parent
        if parent.name and parent.parent.name == "generations":
            parent.rmdir()


def _upsert_pi05_deployment(
    bundle_root: Path,
    deployment_name: str,
    soc_version: str,
    artifacts: dict[str, DeploymentArtifact | Path],
    vlm_bindings,
    action_bindings,
    links: tuple[DeviceLink, ...],
) -> Path:
    artifact_values = {
        role: artifact
        if isinstance(artifact, DeploymentArtifact)
        else DeploymentArtifact(
            path=artifact.relative_to(bundle_root.resolve(strict=True)).as_posix(),
            format="json" if role == "denoising_schedule" else "om",
        )
        for role, artifact in artifacts.items()
    }
    deployment = CompiledDeployment(
        execution_contract=ExecutionContract(
            state_scope="request",
            execution_structure="iterative",
            orchestration_visibility="executor",
            cancellation_granularity="checkpoint",
        ),
        runtime_profile=RoleRuntimeProfile(
            backend="ascend",
            target=DeploymentTarget(soc=soc_version, runtime="acl"),
            profile=AscendRuntimeProfile(device_id=0),
        ),
        artifacts=artifact_values,
        execution=("vlm", "action_expert"),
        bindings={"vlm": vlm_bindings, "action_expert": action_bindings},
        device_links=links,
    )
    return upsert_deployment(bundle_root, deployment_name, deployment).manifest_path


def replace_pi05_ascend_schedule(
    bundle_root: str | Path,
    deployment_name: str,
    schedule_path: str | Path,
) -> Path:
    """Replace only one velocity PI0.5 deployment's schedule artifact."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    selected = load_inference_manifest(root, deployment_name)
    deployment = selected.deployment
    if selected.policy.policy_type != "pi05":
        raise ValueError(f"deployment {deployment_name!r} is not a PI0.5 deployment")
    if not isinstance(deployment, CompiledDeployment) or deployment.backend != "ascend":
        raise ValueError(f"deployment {deployment_name!r} must be a compiled Ascend deployment")
    if deployment.execution != ("vlm", "action_expert"):
        raise ValueError(f"deployment {deployment_name!r} is not a PI0.5 Ascend execution plan")

    action_outputs = [
        binding for binding in deployment.bindings["action_expert"].outputs if binding.semantic == "action"
    ]
    if len(action_outputs) != 1:
        raise ValueError(f"deployment {deployment_name!r} must contain exactly one Action Expert action output")
    runtime_name = action_outputs[0].runtime_name or ""
    output_name = next(
        (part for part in reversed(runtime_name.split(":")) if part in {"action", "velocity", "v_t"}),
        runtime_name,
    )
    if output_name not in {"velocity", "v_t"}:
        raise ValueError(f"deployment {deployment_name!r} does not expose a velocity/v_t Action Expert output")

    schedule_source = Path(schedule_path).expanduser().resolve(strict=True)
    load_pi05_schedule(schedule_source)
    packaged_schedule = package_deployment_artifact(
        root,
        schedule_source,
        backend="ascend",
        deployment_name=deployment_name,
        role="denoising_schedule",
        force_copy=True,
    )
    artifacts = dict(deployment.artifacts)
    artifacts["denoising_schedule"] = deployment_artifact(root, packaged_schedule, "json")
    replacement = deployment.model_copy(update={"artifacts": artifacts})
    try:
        return update_deployment(
            root,
            deployment_name,
            replacement,
            expected_uuid=deployment.uuid,
            expected_revision=deployment.revision,
        ).manifest_path
    except Exception:
        _remove_unreferenced_artifacts(root, [packaged_schedule])
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compile exported PI05 ONNX (VLM / Action Expert) to Ascend OM via ATC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--policy-path",
        "--pretrained-policy-path",
        dest="pretrained_policy_path",
        type=str,
        required=True,
        help="Local PI05 policy bundle directory. Alias: --pretrained-policy-path.",
    )
    p.add_argument(
        "--soc-version",
        type=str,
        required=True,
        help="Target Ascend SoC, e.g. Ascend310P3 (see `npu-smi info`).",
    )
    p.add_argument("--vlm-onnx", type=str, default=None, help="VLM ONNX to compile (omit to skip the VLM).")
    p.add_argument("--ae-onnx", type=str, default=None, help="Action Expert ONNX to compile (omit to skip the AE).")
    p.add_argument(
        "--vlm-om",
        type=str,
        default=None,
        help="ATC VLM OM work output (default: models/_work/<bundle>/ascend/pi05/vlm.om).",
    )
    p.add_argument(
        "--ae-om",
        type=str,
        default=None,
        help="ATC Action Expert OM work output (default: models/_work/<bundle>/ascend/pi05/action_expert.om).",
    )
    p.add_argument(
        "--vlm-abi",
        type=str,
        default=None,
        help="Existing compiler/runtime-introspected VLM OM ABI JSON input (default: <vlm-om>.abi.json).",
    )
    p.add_argument(
        "--ae-abi",
        type=str,
        default=None,
        help="Existing compiler/runtime-introspected Action Expert OM ABI JSON input (default: <ae-om>.abi.json).",
    )
    p.add_argument(
        "--schedule-file",
        type=str,
        default=None,
        help="PI0.5 velocity denoising schedule JSON. Defaults to a uniform schedule from bundle config.json.",
    )
    p.add_argument(
        "--bundle-root",
        type=str,
        default=None,
        help="Policy bundle root for inference_manifest.json (default: pretrained policy path).",
    )
    p.add_argument("--skip-manifest", action="store_true", help="Do not finalize inference_manifest.json.")
    p.add_argument(
        "--manifest-only",
        action="store_true",
        help="Skip ATC and finalize the manifest from existing ONNX and OM artifacts.",
    )
    p.add_argument(
        "--reuse-artifact-role",
        action="append",
        choices=("vlm", "action_expert", "denoising_schedule"),
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--deployment", default="ascend", help="Unified manifest deployment name.")
    p.add_argument("--abi-device-id", type=int, default=0, help="Ascend device used to inspect compiled OM ABI.")
    p.add_argument("--acl-config-path", default=None, help="Optional ACL initialization config for ABI inspection.")
    p.add_argument(
        "--input-shape",
        choices=("none", "auto"),
        default="none",
        help="ATC --input_shape handling: omit by default, or derive from static ONNX graph inputs.",
    )
    p.add_argument(
        "--atc-arg",
        action="append",
        default=None,
        metavar="ARG",
        help="Extra raw argument forwarded to atc (repeatable). Default adds --precision_mode_v2=origin unless overridden.",
    )
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress the final result block (used when an orchestrator prints its own summary).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    setup_logging(args.log_level)

    if not args.vlm_onnx and not args.ae_onnx and not args.manifest_only:
        LOGGER.error("Nothing to do: pass --vlm-onnx and/or --ae-onnx.")
        return 1

    if not args.manifest_only and shutil.which("atc") is None:
        LOGGER.error(
            "`atc` not found on PATH. Run this on an Ascend host with the CANN "
            "toolkit installed (and `source` the CANN environment)."
        )
        return 1

    policy_path = Path(args.pretrained_policy_path).expanduser()
    if args.bundle_root is not None:
        manifest_dir = Path(args.bundle_root).expanduser().resolve()
    else:
        if not policy_path.is_dir():
            LOGGER.error(
                "--pretrained-policy-path %s is not a local directory; pass --bundle-root explicitly.",
                policy_path,
            )
            return 1
        manifest_dir = policy_path.resolve()

    extra_args = list(args.atc_arg or [])
    reuse_roles = frozenset(args.reuse_artifact_role or ())

    jobs: list[tuple[str, Path, Path]] = []
    if args.vlm_onnx:
        vlm_onnx = Path(args.vlm_onnx).expanduser()
        vlm_om = Path(args.vlm_om).expanduser().resolve() if args.vlm_om else _default_om_output(manifest_dir, "vlm")
        jobs.append(("vlm", vlm_onnx, vlm_om))
    if args.ae_onnx:
        ae_onnx = Path(args.ae_onnx).expanduser()
        ae_om = (
            Path(args.ae_om).expanduser().resolve() if args.ae_om else _default_om_output(manifest_dir, "action_expert")
        )
        jobs.append(("action_expert", ae_onnx, ae_om))

    produced: list[tuple[str, str]] = []
    total = len(jobs)
    for i, (role, onnx_path, om_output) in enumerate(jobs, start=1):
        if role in reuse_roles:
            continue
        if args.manifest_only:
            if not onnx_path.is_file():
                raise FileNotFoundError(f"{role} ONNX not found: {onnx_path}")
            if not om_output.is_file():
                raise FileNotFoundError(f"{role} OM not found: {om_output}")
            om_path = om_output
        else:
            om_path = convert_role(
                role=role,
                onnx_path=onnx_path,
                om_output=om_output,
                soc_version=args.soc_version,
                extra_args=extra_args,
                input_shape_mode=args.input_shape,
                index=i,
                total=total,
            )
        explicit_abi = args.vlm_abi if role == "vlm" else args.ae_abi
        abi_path = Path(explicit_abi).expanduser() if explicit_abi else Path(f"{om_path}.abi.json")
        if explicit_abi:
            read_runtime_abi(abi_path)
        elif not args.manifest_only or not abi_path.is_file():
            abi_path.unlink(missing_ok=True)
            write_acl_om_abi(
                om_path,
                abi_path,
                device_id=args.abi_device_id,
                acl_config_path=args.acl_config_path,
            )
        produced.append((role, str(om_path)))

    if not args.skip_manifest:
        vlm_om = (
            Path(args.vlm_om).expanduser().resolve()
            if args.vlm_om
            else next((Path(path) for role, path in produced if role == "vlm"), _default_om_output(manifest_dir, "vlm"))
        )
        action_om = (
            Path(args.ae_om).expanduser().resolve()
            if args.ae_om
            else next(
                (Path(path) for role, path in produced if role == "action_expert"),
                _default_om_output(manifest_dir, "action_expert"),
            )
        )
        for role, om_path in (("vlm", vlm_om), ("action_expert", action_om)):
            if role in reuse_roles:
                continue
            if not om_path.is_file():
                raise FileNotFoundError(f"Cannot finalize PI0.5 manifest: {role} OM not found: {om_path}")
        vlm_abi = Path(args.vlm_abi).expanduser() if args.vlm_abi else Path(f"{vlm_om}.abi.json")
        action_abi = Path(args.ae_abi).expanduser() if args.ae_abi else Path(f"{action_om}.abi.json")
        if "vlm" not in reuse_roles and not vlm_abi.is_file():
            write_acl_om_abi(
                vlm_om,
                vlm_abi,
                device_id=args.abi_device_id,
                acl_config_path=args.acl_config_path,
            )
        if "action_expert" not in reuse_roles and not action_abi.is_file():
            write_acl_om_abi(
                action_om,
                action_abi,
                device_id=args.abi_device_id,
                acl_config_path=args.acl_config_path,
            )
        manifest_path = write_pi05_ascend_deployment(
            manifest_dir,
            args.deployment,
            args.soc_version,
            vlm_abi,
            vlm_om,
            action_abi,
            action_om,
            Path(args.schedule_file).expanduser() if args.schedule_file else None,
            reuse_roles,
        )
        LOGGER.info("  unified manifest → %s", manifest_path)

    rows = [(role, path) for role, path in produced]
    if not args.skip_manifest:
        rows.append(("manifest", str(manifest_dir / "inference_manifest.json")))
    if not args.no_summary:
        print_summary("PI05 OM conversion complete", rows, status="✅ DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

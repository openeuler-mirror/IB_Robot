from __future__ import annotations

import json
from argparse import Namespace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_manifest import load_inference_manifest
from inference_service.pi05_schedule import load_pi05_schedule
from model_utils.pi05_export import _cli
from model_utils.pi05_export.convert_om import (
    _run_atc,
    build_arg_parser,
    convert_role,
    replace_pi05_ascend_schedule,
    write_pi05_ascend_deployment,
)

pipeline = import_module("model_utils.pi05_export.__main__")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _create_pi05_bundle(root: Path) -> None:
    _write_json(
        root / "config.json",
        {
            "type": "pi05",
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.top": {"type": "VISUAL", "shape": [3, 16, 24]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
            "chunk_size": 2,
            "max_action_dim": 8,
            "num_inference_steps": 2,
        },
    )
    _write_json(root / "policy_preprocessor.json", {"name": "pre", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "post", "steps": []})


def test_convert_om_parser_uses_unified_manifest_options():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--pretrained-policy-path",
            "/tmp/policy",
            "--soc-version",
            "Ascend310P3",
            "--bundle-root",
            "/tmp/bundle",
            "--skip-manifest",
        ]
    )

    assert args.bundle_root == "/tmp/bundle"
    assert args.skip_manifest is True
    assert args.schedule_file is None
    assert "--schedule-file" in parser.format_help()
    assert "--skip-om-manifest" not in parser.format_help()
    assert "--om-manifest-dir" not in parser.format_help()


def test_convert_role_removes_stale_om_before_atc(tmp_path, monkeypatch):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    om_path = tmp_path / "model.om"
    om_path.write_bytes(b"stale")
    monkeypatch.setattr("model_utils.pi05_export.convert_om._run_atc", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="ATC failed"):
        convert_role(
            role="vlm",
            onnx_path=onnx_path,
            om_output=om_path,
            soc_version="Ascend310P3",
            extra_args=[],
            input_shape_mode="none",
            index=1,
            total=1,
        )

    assert not om_path.exists()


def test_run_atc_resolves_external_data_from_onnx_directory(tmp_path, monkeypatch):
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    onnx_path = onnx_dir / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    recorded = {}

    def fake_run(command, *, check, cwd):
        recorded.update(command=command, check=check, cwd=cwd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("model_utils.pi05_export.convert_om.subprocess.run", fake_run)

    assert _run_atc(
        onnx_path,
        tmp_path / "model.om",
        "Ascend310P3",
        extra_args=[],
        input_shape_mode="none",
    )
    assert recorded["check"] is False
    assert recorded["cwd"] == onnx_dir
    assert f"--model={onnx_path}" in recorded["command"]


def test_run_atc_resolves_relative_model_before_changing_cwd(tmp_path, monkeypatch):
    onnx_dir = tmp_path / "run" / "onnx"
    onnx_dir.mkdir(parents=True)
    onnx_path = onnx_dir / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    recorded = {}

    def fake_run(command, *, check, cwd):
        recorded.update(command=command, check=check, cwd=cwd)
        return SimpleNamespace(returncode=0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("model_utils.pi05_export.convert_om.subprocess.run", fake_run)

    assert _run_atc(
        Path("run/onnx/model.onnx"),
        tmp_path / "model.om",
        "Ascend310P3",
        extra_args=[],
        input_shape_mode="none",
    )
    assert recorded["cwd"] == onnx_dir
    assert f"--model={onnx_path}" in recorded["command"]


def test_write_pi05_ascend_deployment_uses_compiled_abis_and_strict_loader(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _create_pi05_bundle(bundle)
    compiler_output = tmp_path / "compiler"
    compiler_output.mkdir()
    vlm_om = compiler_output / "vlm.om"
    action_om = compiler_output / "action.om"
    vlm_om.write_bytes(b"vlm")
    action_om.write_bytes(b"action")
    vlm_abi = compiler_output / "vlm.abi.json"
    action_abi = compiler_output / "action.abi.json"
    _write_json(
        vlm_abi,
        {
            "inputs": [
                {
                    "name": "observation.images.top",
                    "index": 0,
                    "dtype": "float32",
                    "shape": [1, 3, 16, 24],
                    "layout": "NCHW",
                },
                {"name": "lang_tokens", "index": 1, "dtype": "int64", "shape": [1, 4]},
                {"name": "lang_masks", "index": 2, "dtype": "bool", "shape": [1, 4]},
                {
                    "name": "prefix_att_2d_masks_4d",
                    "index": 3,
                    "dtype": "float32",
                    "shape": [1, 1, 8, 8],
                },
            ],
            "outputs": [
                {"name": "/Concat_12:0:past_kv_tensor", "index": 0, "dtype": "float16", "shape": [1, 2]},
                {"name": "/Cast_12:0:prefix_pad_masks", "index": 1, "dtype": "bool", "shape": [1, 4]},
            ],
        },
    )
    _write_json(
        action_abi,
        {
            "inputs": [
                {"name": "past_kv_tensor", "index": 0, "dtype": "float16", "shape": [1, 2]},
                {"name": "prefix_pad_masks", "index": 1, "dtype": "bool", "shape": [1, 4]},
                {"name": "time", "index": 2, "dtype": "float32", "shape": [1]},
                {"name": "noise", "index": 3, "dtype": "float32", "shape": [1, 2, 8]},
            ],
            "outputs": [{"name": "/Add_2:0:action", "index": 0, "dtype": "float32", "shape": [1, 2, 8]}],
        },
    )

    manifest_path = write_pi05_ascend_deployment(
        bundle,
        "ascend",
        "Ascend310P3",
        vlm_abi,
        vlm_om,
        action_abi,
        action_om,
    )
    validated = load_inference_manifest(bundle, "ascend")

    assert manifest_path == bundle / "inference_manifest.json"
    assert validated.deployment.execution == ("vlm", "action_expert")
    assert "functional_execution" not in validated.deployment.model_dump()
    from inference_service.codecs.policies import PI05PolicyCodec

    PI05PolicyCodec.validate_staged_deployment(validated.deployment)
    assert {link.semantic for link in validated.deployment.device_links} == {
        "internal.past_kv",
        "internal.prefix_pad_masks",
    }
    assert validated.deployment.bindings["vlm"].inputs[1].semantic == "observation.language.tokens"
    prefix_mask = validated.deployment.bindings["vlm"].inputs[3]
    assert prefix_mask.semantic == "prefix_att_2d_masks_4d"
    assert prefix_mask.layout == "NCHW"
    assert all(
        artifact.path.startswith("artifacts/ascend/ascend/") for artifact in validated.deployment.artifacts.values()
    )


def test_velocity_deployment_packages_schedule_and_tracks_it_in_fingerprint(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _create_pi05_bundle(bundle)
    compiler_output = tmp_path / "compiler"
    compiler_output.mkdir()
    vlm_om = compiler_output / "vlm.om"
    action_om = compiler_output / "action.om"
    vlm_om.write_bytes(b"vlm")
    action_om.write_bytes(b"velocity")
    vlm_abi = compiler_output / "vlm.abi.json"
    action_abi = compiler_output / "action.abi.json"
    _write_pi05_abis(vlm_abi, action_abi)
    action_value = json.loads(action_abi.read_text(encoding="utf-8"))
    action_value["outputs"][0]["name"] = "/Identity_7:0:velocity"
    _write_json(action_abi, action_value)

    write_pi05_ascend_deployment(
        bundle,
        "ascend",
        "Ascend310P3",
        vlm_abi,
        vlm_om,
        action_abi,
        action_om,
    )
    first = load_inference_manifest(bundle, "ascend")
    deployment = first.deployment
    schedule_artifact = deployment.artifacts["denoising_schedule"]

    assert deployment.execution == ("vlm", "action_expert")
    assert schedule_artifact.format == "json"
    assert deployment.bindings["action_expert"].outputs[0].semantic == "action"
    assert deployment.bindings["action_expert"].outputs[0].runtime_name == "/Identity_7:0:velocity"
    generated = load_pi05_schedule(bundle / schedule_artifact.path)
    assert generated.name == "uniform2"
    assert generated.timesteps == (1.0, 0.5, 0.0)
    assert not (bundle / "model_utils_work").exists()

    custom_schedule = tmp_path / "custom-schedule.json"
    _write_json(
        custom_schedule,
        {
            "format": "pi05-denoising-schedule-v1",
            "name": "non-uniform",
            "algorithm": "euler",
            "model_output": "velocity",
            "timesteps": [1.0, 0.8, 0.0],
        },
    )
    write_pi05_ascend_deployment(
        bundle,
        "ascend",
        "Ascend310P3",
        vlm_abi,
        vlm_om,
        action_abi,
        action_om,
        custom_schedule,
    )
    second = load_inference_manifest(bundle, "ascend")

    assert second.fingerprint != first.fingerprint
    assert second.deployment.uuid == first.deployment.uuid
    assert second.deployment.revision == first.deployment.revision + 1
    assert second.deployment.artifacts["denoising_schedule"].path != schedule_artifact.path


def test_replace_pi05_ascend_schedule_reuses_existing_om_artifacts(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _create_pi05_bundle(bundle)
    compiler_output = tmp_path / "compiler"
    compiler_output.mkdir()
    vlm_om = compiler_output / "vlm.om"
    action_om = compiler_output / "action.om"
    vlm_om.write_bytes(b"vlm-large-placeholder")
    action_om.write_bytes(b"velocity-large-placeholder")
    vlm_abi = compiler_output / "vlm.abi.json"
    action_abi = compiler_output / "action.abi.json"
    _write_pi05_abis(vlm_abi, action_abi)
    write_pi05_ascend_deployment(
        bundle,
        "ascend-velocity",
        "Ascend310P3",
        vlm_abi,
        vlm_om,
        action_abi,
        action_om,
    )
    before = load_inference_manifest(bundle, "ascend-velocity")
    before_oms = {role: before.deployment.artifacts[role] for role in ("vlm", "action_expert")}
    before_bytes = {role: (bundle / artifact.path).read_bytes() for role, artifact in before_oms.items()}
    schedule_path = tmp_path / "selected.json"
    _write_json(
        schedule_path,
        {
            "format": "pi05-denoising-schedule-v1",
            "name": "selected",
            "algorithm": "euler",
            "model_output": "velocity",
            "timesteps": [1.0, 0.7, 0.2, 0.0],
        },
    )

    manifest_path = replace_pi05_ascend_schedule(bundle, "ascend-velocity", schedule_path)
    after = load_inference_manifest(bundle, "ascend-velocity")

    assert manifest_path == (bundle / "inference_manifest.json").resolve()
    assert {role: after.deployment.artifacts[role] for role in before_oms} == before_oms
    assert {(bundle / artifact.path).read_bytes() for artifact in before_oms.values()} == set(before_bytes.values())
    installed = after.deployment.artifacts["denoising_schedule"]
    assert load_pi05_schedule(bundle / installed.path).name == "selected"


def test_replace_pi05_schedule_rejects_concurrent_deployment_revision(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _create_pi05_bundle(bundle)
    compiler_output = tmp_path / "compiler"
    compiler_output.mkdir()
    vlm_om = compiler_output / "vlm.om"
    action_om = compiler_output / "action.om"
    vlm_om.write_bytes(b"vlm")
    action_om.write_bytes(b"velocity")
    vlm_abi = compiler_output / "vlm.abi.json"
    action_abi = compiler_output / "action.abi.json"
    _write_pi05_abis(vlm_abi, action_abi)
    write_pi05_ascend_deployment(bundle, "ascend", "Ascend310P3", vlm_abi, vlm_om, action_abi, action_om)
    schedule = tmp_path / "schedule.json"
    _write_json(
        schedule,
        {
            "format": "pi05-denoising-schedule-v1",
            "name": "replacement",
            "algorithm": "euler",
            "model_output": "velocity",
            "timesteps": [1.0, 0.0],
        },
    )

    convert_om = import_module("model_utils.pi05_export.convert_om")
    real_package = convert_om.package_deployment_artifact

    def concurrent_package(*args, **kwargs):
        packaged = real_package(*args, **kwargs)
        current = load_inference_manifest(bundle, "ascend").deployment
        changed_profile = current.runtime_profile.model_copy(
            update={"profile": current.runtime_profile.profile.model_copy(update={"device_id": 1})}
        )
        changed = current.model_copy(update={"runtime_profile": changed_profile})
        from model_utils.inference_manifest_export import upsert_deployment

        upsert_deployment(bundle, "ascend", changed)
        return packaged

    monkeypatch.setattr(convert_om, "package_deployment_artifact", concurrent_package)

    with pytest.raises(ValueError, match="revision conflict"):
        replace_pi05_ascend_schedule(bundle, "ascend", schedule)


def test_failed_pi05_publication_removes_created_generations(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _create_pi05_bundle(bundle)
    compiler_output = tmp_path / "compiler"
    compiler_output.mkdir()
    vlm_om = compiler_output / "vlm.om"
    action_om = compiler_output / "action.om"
    vlm_om.write_bytes(b"vlm")
    action_om.write_bytes(b"velocity")
    vlm_abi = compiler_output / "vlm.abi.json"
    action_abi = compiler_output / "action.abi.json"
    _write_pi05_abis(vlm_abi, action_abi)
    invalid_schedule = tmp_path / "invalid.json"
    invalid_schedule.write_text("{}")

    with pytest.raises(ValueError):
        write_pi05_ascend_deployment(
            bundle,
            "ascend",
            "Ascend310P3",
            vlm_abi,
            vlm_om,
            action_abi,
            action_om,
            invalid_schedule,
        )

    generations = bundle / "artifacts" / "ascend" / "ascend" / "generations"
    assert not generations.exists() or not list(generations.iterdir())


def test_legacy_action_deployment_rejects_explicit_schedule(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _create_pi05_bundle(bundle)
    compiler_output = tmp_path / "compiler"
    compiler_output.mkdir()
    vlm_om = compiler_output / "vlm.om"
    action_om = compiler_output / "action.om"
    vlm_om.write_bytes(b"vlm")
    action_om.write_bytes(b"action")
    vlm_abi = compiler_output / "vlm.abi.json"
    action_abi = compiler_output / "action.abi.json"
    _write_pi05_abis(vlm_abi, action_abi)
    action_value = json.loads(action_abi.read_text(encoding="utf-8"))
    action_value["outputs"][0]["name"] = "action"
    _write_json(action_abi, action_value)
    schedule = tmp_path / "schedule.json"
    _write_json(
        schedule,
        {
            "format": "pi05-denoising-schedule-v1",
            "name": "uniform",
            "algorithm": "euler",
            "model_output": "velocity",
            "timesteps": [1.0, 0.0],
        },
    )

    with pytest.raises(ValueError, match="only valid for a velocity"):
        write_pi05_ascend_deployment(
            bundle,
            "ascend",
            "Ascend310P3",
            vlm_abi,
            vlm_om,
            action_abi,
            action_om,
            schedule,
        )


def _write_pi05_abis(vlm_path: Path, action_path: Path) -> None:
    _write_json(
        vlm_path,
        {
            "inputs": [
                {
                    "name": "observation.images.top",
                    "index": 0,
                    "dtype": "float32",
                    "shape": [1, 3, 16, 24],
                    "layout": "NCHW",
                },
                {"name": "lang_tokens", "index": 1, "dtype": "int64", "shape": [1, 4]},
                {"name": "lang_masks", "index": 2, "dtype": "bool", "shape": [1, 4]},
                {
                    "name": "prefix_att_2d_masks_4d",
                    "index": 3,
                    "dtype": "float32",
                    "shape": [1, 1, 8, 8],
                },
            ],
            "outputs": [
                {"name": "past_kv_tensor", "index": 0, "dtype": "float16", "shape": [1, 2]},
                {"name": "prefix_pad_masks", "index": 1, "dtype": "bool", "shape": [1, 4]},
            ],
        },
    )
    _write_json(
        action_path,
        {
            "inputs": [
                {"name": "past_kv_tensor", "index": 0, "dtype": "float16", "shape": [1, 2]},
                {"name": "prefix_pad_masks", "index": 1, "dtype": "bool", "shape": [1, 4]},
                {"name": "time", "index": 2, "dtype": "float32", "shape": [1]},
                {"name": "noise", "index": 3, "dtype": "float32", "shape": [1, 2, 8]},
            ],
            "outputs": [{"name": "velocity", "index": 0, "dtype": "float32", "shape": [1, 2, 8]}],
        },
    )


def test_published_pi05_runs_staged_factory_iteration_postprocess_and_reset(tmp_path, monkeypatch):
    import numpy as np

    from inference_service.backends import BackendCapabilities, InferenceRequest
    from inference_service.pipeline.factory import create_inference_pipeline
    from inference_service.pipeline.staged_executor import StagedModelExecutor
    from inference_service.pipeline.stages import IterativeStage
    from inference_service.runtime_composition import build_policy_runtime_dependencies

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _create_pi05_bundle(bundle)
    vlm_abi, action_abi = tmp_path / "vlm.json", tmp_path / "action.json"
    _write_pi05_abis(vlm_abi, action_abi)
    vlm_om, action_om = tmp_path / "vlm.om", tmp_path / "action.om"
    vlm_om.write_bytes(b"test-vlm")
    action_om.write_bytes(b"test-action")
    write_pi05_ascend_deployment(bundle, "ascend", "Ascend310P3", vlm_abi, vlm_om, action_abi, action_om)
    manifest = load_inference_manifest(bundle, "ascend")
    calls, postprocessed = [], []

    class Session:
        from inference_service.backends.types import BackendPriorityMapping

        capabilities = BackendCapabilities(
            supports_isolated_stage_execution=True, priority_mapping=BackendPriorityMapping(tuple(range(8)))
        )

        def execute_role(self, role, inputs, request, context, *, isolated, operation_factory):
            assert isolated
            operation, registry = operation_factory(role)
            assert operation is not None and registry is not None
            calls.append((role, context.request_id))
            # Isolated operations must reach a known outcome before the stub
            # returns, otherwise the registry keeps them and staged reset
            # refuses to run (same contract as the real Ascend session).
            from inference_service.scheduler.operations import Certainty

            registry.finish(operation.operation_id, certainty=Certainty.COMPLETED)
            registry.detach_waiter(operation.operation_id, context.request_id)
            if role == "vlm":
                return {
                    "internal.past_kv": np.ones((1, 2), dtype=np.float16),
                    "internal.prefix_pad_masks": np.ones((1, 4), dtype=bool),
                }
            return {"action": np.ones_like(inputs["noise"])}

    def postprocess(action):
        postprocessed.append(action.copy())
        return action + 10

    # Keep manifest validation, factory, codec, stages and lifecycle real;
    # substitute only external processors and the vendor model session.
    monkeypatch.setattr(
        "inference_service.pipeline.factory.create_lerobot_processor_views",
        lambda: (lambda inputs: inputs, postprocess),
    )
    dependencies = build_policy_runtime_dependencies()
    policy = None
    try:
        policy = create_inference_pipeline(
            "policy",
            manifest,
            stage_policy="independent",
            priority_scheduling=True,
            request_timeout=2,
            registry_set=dependencies.registry_set,
            providers=dependencies.providers,
            model_session_factory=lambda *_: Session(),
        )
        assert isinstance(policy._session_executor, StagedModelExecutor)
        assert any(isinstance(stage, IterativeStage) for stage in policy._session_executor.stages)
        policy.load()
        inputs = {
            "observation.images.top": np.zeros((1, 3, 16, 24), dtype=np.float32),
            "lang_tokens": np.zeros((1, 4), dtype=np.int64),
            "lang_masks": np.ones((1, 4), dtype=bool),
            "noise": np.zeros((1, 2, 8), dtype=np.float32),
        }
        assert policy.submit_frame(InferenceRequest("camera", inputs=inputs)).result(timeout=2) == 1
        result = policy.infer(InferenceRequest("dispatch", inputs=inputs), capture_raw_action=True)
        assert [role for role, _ in calls] == ["vlm", "action_expert", "action_expert"]
        assert result.actual_chunk_size == 2
        assert result.deployment_fingerprint == manifest.fingerprint
        np.testing.assert_allclose(postprocessed[0], -np.ones((1, 2, 6)))
        np.testing.assert_allclose(result.action, np.full((1, 2, 6), 9))
        policy.reset()
        policy.infer(InferenceRequest("after-reset", inputs=inputs))
        assert [role for role, _ in calls[3:]] == ["vlm", "action_expert", "action_expert"]
        assert calls[3][1].startswith("after-reset:refresh:")
    finally:
        if policy is not None:
            policy.close()
        dependencies.providers.close()


def _resolved(tmp_path: Path, steps: str) -> SimpleNamespace:
    policy = tmp_path / "bundle"
    policy.mkdir(exist_ok=True)
    _create_pi05_bundle(policy)
    args = Namespace(
        policy_path=str(policy),
        work_dir=str(tmp_path / "work"),
        dtype="fp16",
        device="cpu",
        donor_device="cpu",
        fast_gelu=False,
        fast_gelu_scope="none",
        soc_version="Ascend310P3",
        schedule_file=None,
        deployment="ascend",
        abi_device_id=0,
        acl_config_path=None,
        steps=steps,
        batch_path=str(tmp_path / "batches.json"),
        num_calib=2,
        amp_num=0,
        amp_rank_samples=1,
        amp_scratch_dir=None,
        task="pick",
        log_level="INFO",
    )
    return SimpleNamespace(args=args, sources={}, config_path=str(tmp_path / "config.yaml"), profile=None)


def _mock_pipeline(monkeypatch, resolved, *, skip_product: str | None = None):
    monkeypatch.setattr(pipeline._cli, "resolve", lambda: resolved)
    monkeypatch.setattr(pipeline._cli, "print_effective", lambda value: None)
    monkeypatch.setattr(pipeline._cli, "write_last", lambda value: None)
    monkeypatch.setattr(pipeline, "print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_capture_ae_calibration", lambda _ctx: None)

    def runner(step):
        def run(ctx):
            if step == skip_product:
                return
            product = pipeline._product(ctx, step)
            if product is not None:
                product.parent.mkdir(parents=True, exist_ok=True)
                product.write_bytes(step.encode())
            if step == "vlm_onnx":
                ctx.runtime_save_dir.mkdir(parents=True, exist_ok=True)
                (ctx.runtime_save_dir / "past_kv_tensor.pth").write_bytes(b"kv")
                (ctx.runtime_save_dir / "prefix_pad_masks.pth").write_bytes(b"mask")
            if step.endswith("_om"):
                abi_path = Path(f"{product}.abi.json")
                if "vlm" in step:
                    temporary_action = ctx.om_dir / "temporary-action.abi.json"
                    _write_pi05_abis(abi_path, temporary_action)
                    temporary_action.unlink()
                else:
                    temporary_vlm = ctx.om_dir / "temporary-vlm.abi.json"
                    _write_pi05_abis(temporary_vlm, abi_path)
                    temporary_vlm.unlink()

        return run

    monkeypatch.setattr(pipeline, "_RUNNERS", {name: runner(name) for name in _cli.STEP_NAMES})

    def run_module(module, argv):
        assert module == "model_utils.pi05_export.convert_om"

        def value(flag):
            return argv[argv.index(flag) + 1]

        reuse_roles = frozenset(argv[index + 1] for index, item in enumerate(argv) if item == "--reuse-artifact-role")

        write_pi05_ascend_deployment(
            Path(value("--pretrained-policy-path")),
            value("--deployment"),
            value("--soc-version"),
            Path(f"{value('--vlm-om')}.abi.json"),
            Path(value("--vlm-om")),
            Path(f"{value('--ae-om')}.abi.json"),
            Path(value("--ae-om")),
            Path(value("--schedule-file")) if "--schedule-file" in argv else None,
            reuse_roles,
        )

    monkeypatch.setattr(pipeline, "_run_module", run_module)


def test_pipeline_defaults_to_normalized_work_directory(tmp_path, monkeypatch):
    resolved = _resolved(tmp_path, "vlm_onnx")
    resolved.args.work_dir = None
    _mock_pipeline(monkeypatch, resolved)

    assert pipeline.main() == 0

    expected = tmp_path / "_work" / "bundle" / "ascend" / "pi05" / "onnx"
    assert (expected / "pi05-vlm_op17_nodyn_fp16_cpu.onnx").is_file()


def test_pipeline_rejects_bundle_local_work_directory_before_creation(tmp_path, monkeypatch):
    resolved = _resolved(tmp_path, "vlm_onnx")
    unsafe = Path(resolved.args.policy_path) / "work"
    resolved.args.work_dir = str(unsafe)
    monkeypatch.setattr(pipeline._cli, "resolve", lambda: resolved)
    monkeypatch.setattr(pipeline._cli, "print_effective", lambda value: None)

    with pytest.raises(SystemExit, match="must be outside"):
        pipeline.main()

    assert not unsafe.exists()


def test_pipeline_rejects_bundle_local_amp_scratch_before_creation(tmp_path, monkeypatch):
    resolved = _resolved(tmp_path, "vlm_onnx")
    unsafe = Path(resolved.args.policy_path) / "amp"
    resolved.args.amp_scratch_dir = str(unsafe)
    monkeypatch.setattr(pipeline._cli, "resolve", lambda: resolved)
    monkeypatch.setattr(pipeline._cli, "print_effective", lambda value: None)

    with pytest.raises(SystemExit, match="must be outside"):
        pipeline.main()

    assert not unsafe.exists()


def test_default_pipeline_mock_produces_onnx_om_abi_and_manifest(tmp_path, monkeypatch):
    resolved = _resolved(tmp_path, ",".join(_cli.DEFAULT_STEPS))
    _mock_pipeline(monkeypatch, resolved)

    assert pipeline.main() == 0

    assert len(list((tmp_path / "work" / "onnx").glob("*.onnx"))) == 2
    assert len(list((tmp_path / "work" / "om").glob("*.om"))) == 2
    assert len(list((tmp_path / "work" / "om").glob("*.abi.json"))) == 2
    assert load_inference_manifest(tmp_path / "bundle", "ascend").deployment.execution == ("vlm", "action_expert")


def test_profiled_mixed_deployment_packages_fp_vlm_and_quantized_ae(tmp_path, monkeypatch):
    steps = "vlm_onnx,ae_onnx,vlm_om,ae_om,ae_quant,ae_quant_om"
    resolved = _resolved(tmp_path, steps)
    resolved.args.device = "npu"
    resolved.args.fast_gelu_scope = "vlm-text"
    resolved.quantization_profile = pipeline._cli.resolve_quantization_profile("pi05-vlm-text-ae-attn-mlp-sq-v1", {})
    _mock_pipeline(monkeypatch, resolved)

    assert pipeline.main() == 0

    selected = load_inference_manifest(tmp_path / "bundle", "ascend")
    vlm = tmp_path / "bundle" / selected.deployment.artifacts["vlm"].path
    action_expert = tmp_path / "bundle" / selected.deployment.artifacts["action_expert"].path
    assert vlm.read_bytes() == b"vlm_om"
    assert action_expert.read_bytes() == b"ae_quant_om"


def test_profile_with_fp_only_steps_publishes_named_deployment(tmp_path, monkeypatch):
    resolved = _resolved(tmp_path, ",".join(_cli.DEFAULT_STEPS))
    resolved.args.device = "npu"
    resolved.args.fast_gelu_scope = "vlm-text"
    resolved.quantization_profile = pipeline._cli.resolve_quantization_profile("pi05-vlm-text-ae-attn-mlp-sq-v1", {})
    _mock_pipeline(monkeypatch, resolved)

    assert pipeline.main() == 0

    manifest = json.loads((tmp_path / "bundle" / "inference_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["deployments"]) == {"ascend"}


def test_profiled_full_rerun_reuses_identical_artifacts(tmp_path, monkeypatch):
    profile = pipeline._cli.resolve_quantization_profile("pi05-vlm-text-ae-attn-mlp-sq-v1", {})
    initial = _resolved(tmp_path, "vlm_onnx,ae_onnx,vlm_om,ae_om,ae_quant,ae_quant_om")
    initial.args.device = "npu"
    initial.args.fast_gelu_scope = "vlm-text"
    initial.quantization_profile = profile
    _mock_pipeline(monkeypatch, initial)
    assert pipeline.main() == 0
    first = load_inference_manifest(tmp_path / "bundle", "ascend")

    rerun = _resolved(tmp_path, "vlm_onnx,ae_onnx,vlm_om,ae_om,ae_quant,ae_quant_om")
    rerun.args.device = "npu"
    rerun.args.fast_gelu_scope = "vlm-text"
    rerun.quantization_profile = profile
    _mock_pipeline(monkeypatch, rerun)
    assert pipeline.main() == 0
    second = load_inference_manifest(tmp_path / "bundle", "ascend")

    assert second.deployment.artifacts["vlm"] == first.deployment.artifacts["vlm"]
    assert second.deployment.artifacts["action_expert"] == first.deployment.artifacts["action_expert"]
    assert second.deployment.revision == first.deployment.revision


def test_partial_quant_publication_skips_missing_target_deployment(tmp_path, monkeypatch):
    initial = _resolved(tmp_path, ",".join(_cli.DEFAULT_STEPS))
    _mock_pipeline(monkeypatch, initial)
    assert pipeline.main() == 0

    rerun = _resolved(tmp_path, "vlm_quant_om")
    quant_onnx = tmp_path / "work" / "onnx" / "pi05-vlm_op17_nodyn_fp16_cpu_w8a8.onnx"
    quant_onnx.parent.mkdir(parents=True, exist_ok=True)
    quant_onnx.write_bytes(b"quantized")
    _mock_pipeline(monkeypatch, rerun)

    assert pipeline.main() == 0
    manifest = json.loads((tmp_path / "bundle" / "inference_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["deployments"]) == {"ascend"}


def test_individual_om_rerun_finalizes_with_existing_counterpart(tmp_path, monkeypatch):
    initial = _resolved(tmp_path, ",".join(_cli.DEFAULT_STEPS))
    _mock_pipeline(monkeypatch, initial)
    assert pipeline.main() == 0
    action_artifact = load_inference_manifest(tmp_path / "bundle", "ascend").deployment.artifacts["action_expert"]

    rerun = _resolved(tmp_path, "vlm_om")
    _mock_pipeline(monkeypatch, rerun)
    assert pipeline.main() == 0

    deployment = load_inference_manifest(tmp_path / "bundle", "ascend").deployment
    assert deployment.artifacts["action_expert"] == action_artifact


def test_action_expert_only_rerun_reuses_existing_vlm_generation(tmp_path, monkeypatch):
    initial = _resolved(tmp_path, ",".join(_cli.DEFAULT_STEPS))
    _mock_pipeline(monkeypatch, initial)
    assert pipeline.main() == 0
    first = load_inference_manifest(tmp_path / "bundle", "ascend")
    vlm_artifact = first.deployment.artifacts["vlm"]

    rerun = _resolved(tmp_path, "ae_om")
    _mock_pipeline(monkeypatch, rerun)
    assert pipeline.main() == 0

    second = load_inference_manifest(tmp_path / "bundle", "ascend")
    assert second.deployment.artifacts["vlm"] == vlm_artifact
    assert second.deployment.uuid == first.deployment.uuid
    assert second.deployment.revision == first.deployment.revision


def test_action_expert_content_change_updates_existing_deployment(tmp_path, monkeypatch):
    initial = _resolved(tmp_path, ",".join(_cli.DEFAULT_STEPS))
    _mock_pipeline(monkeypatch, initial)
    assert pipeline.main() == 0
    first = load_inference_manifest(tmp_path / "bundle", "ascend")

    rerun = _resolved(tmp_path, "ae_om")
    _mock_pipeline(monkeypatch, rerun)
    original = pipeline._RUNNERS["ae_om"]

    def changed_action_expert(ctx):
        original(ctx)
        ctx.ae_om.write_bytes(b"changed-action-expert")

    pipeline._RUNNERS["ae_om"] = changed_action_expert
    assert pipeline.main() == 0

    second = load_inference_manifest(tmp_path / "bundle", "ascend")
    assert second.deployment.uuid == first.deployment.uuid
    assert second.deployment.revision == first.deployment.revision + 1
    assert second.deployment.artifacts["vlm"] == first.deployment.artifacts["vlm"]
    assert second.deployment.artifacts["action_expert"] != first.deployment.artifacts["action_expert"]


def test_new_deployment_name_is_added_and_reuses_identical_artifacts(tmp_path, monkeypatch):
    initial = _resolved(tmp_path, ",".join(_cli.DEFAULT_STEPS))
    _mock_pipeline(monkeypatch, initial)
    assert pipeline.main() == 0
    first = load_inference_manifest(tmp_path / "bundle", "ascend")

    added = _resolved(tmp_path, ",".join(_cli.DEFAULT_STEPS))
    added.args.deployment = "ascend-copy"
    _mock_pipeline(monkeypatch, added)
    assert pipeline.main() == 0
    second = load_inference_manifest(tmp_path / "bundle", "ascend-copy")

    assert set(second.manifest.deployments) == {"ascend", "ascend-copy"}
    assert second.deployment.uuid != first.deployment.uuid
    assert second.deployment.revision == 1
    assert second.deployment.artifacts == first.deployment.artifacts


@pytest.mark.parametrize(
    ("step", "removed_role", "reused_role"),
    [
        ("ae_om", "vlm", "vlm"),
        ("vlm_om", "action_expert", "action_expert"),
    ],
)
def test_partial_rerun_does_not_require_reused_role_work_files(tmp_path, monkeypatch, step, removed_role, reused_role):
    initial = _resolved(tmp_path, ",".join(_cli.DEFAULT_STEPS))
    _mock_pipeline(monkeypatch, initial)
    assert pipeline.main() == 0
    first = load_inference_manifest(tmp_path / "bundle", "ascend")
    expected = first.deployment.artifacts[reused_role]
    pattern = "pi05-vlm*" if removed_role == "vlm" else "pi05-action_expert*"
    for path in (tmp_path / "work" / "om").glob(pattern):
        path.unlink()

    rerun = _resolved(tmp_path, step)
    _mock_pipeline(monkeypatch, rerun)
    assert pipeline.main() == 0

    second = load_inference_manifest(tmp_path / "bundle", "ascend")
    assert second.deployment.artifacts[reused_role] == expected


def test_convert_om_cli_skips_reused_role_work_files(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _create_pi05_bundle(bundle)
    compiler = tmp_path / "compiler"
    compiler.mkdir()
    vlm_om = compiler / "vlm.om"
    action_om = compiler / "action.om"
    vlm_om.write_bytes(b"vlm")
    action_om.write_bytes(b"velocity")
    vlm_abi = Path(f"{vlm_om}.abi.json")
    action_abi = Path(f"{action_om}.abi.json")
    _write_pi05_abis(vlm_abi, action_abi)
    write_pi05_ascend_deployment(bundle, "ascend", "Ascend310P3", vlm_abi, vlm_om, action_abi, action_om)
    vlm_om.unlink()
    vlm_abi.unlink()
    action_om.write_bytes(b"velocity-v2")
    vlm_onnx = compiler / "unused-vlm.onnx"
    action_onnx = compiler / "action.onnx"
    vlm_onnx.write_bytes(b"unused")
    action_onnx.write_bytes(b"action")

    convert_om = import_module("model_utils.pi05_export.convert_om")
    monkeypatch.setattr(
        "sys.argv",
        [
            "convert_om",
            "--policy-path",
            str(bundle),
            "--soc-version",
            "Ascend310P3",
            "--vlm-onnx",
            str(vlm_onnx),
            "--vlm-om",
            str(vlm_om),
            "--ae-onnx",
            str(action_onnx),
            "--ae-om",
            str(action_om),
            "--ae-abi",
            str(action_abi),
            "--deployment",
            "ascend",
            "--reuse-artifact-role",
            "vlm",
            "--reuse-artifact-role",
            "denoising_schedule",
            "--manifest-only",
            "--no-summary",
        ],
    )

    assert convert_om.main() == 0
    assert load_inference_manifest(bundle, "ascend").deployment.revision == 2


def test_single_om_without_counterpart_keeps_artifact_without_manifest(tmp_path, monkeypatch):
    resolved = _resolved(tmp_path, "ae_quant_om")
    quant_onnx = tmp_path / "work" / "onnx" / "pi05-action_expert_op17_nodyn_fp16_cpu_w8a8.onnx"
    quant_onnx.parent.mkdir(parents=True)
    quant_onnx.write_bytes(b"quantized")
    _mock_pipeline(monkeypatch, resolved)

    assert pipeline.main() == 0

    assert list((tmp_path / "work" / "om").glob("pi05-action_expert*_w8a8.om"))
    assert not (tmp_path / "bundle" / "inference_manifest.json").exists()


@pytest.mark.parametrize(
    "steps",
    [
        "vlm_onnx",
        "ae_onnx",
        "vlm_onnx,ae_onnx",
        "vlm_om",
        "ae_om",
        "vlm_onnx,vlm_om",
        "ae_onnx,ae_om",
    ],
)
def test_individual_onnx_and_om_step_combinations(tmp_path, monkeypatch, steps):
    initial = _resolved(tmp_path, ",".join(_cli.DEFAULT_STEPS))
    _mock_pipeline(monkeypatch, initial)
    assert pipeline.main() == 0

    rerun = _resolved(tmp_path, steps)
    _mock_pipeline(monkeypatch, rerun)
    assert pipeline.main() == 0

    requested = set(steps.split(","))
    if "vlm_onnx" in requested:
        assert list((tmp_path / "work" / "onnx").glob("pi05-vlm*.onnx"))
    if "ae_onnx" in requested:
        assert list((tmp_path / "work" / "onnx").glob("pi05-action_expert*.onnx"))
    if requested & {"vlm_om", "ae_om"}:
        assert load_inference_manifest(tmp_path / "bundle", "ascend").deployment.execution == (
            "vlm",
            "action_expert",
        )


def test_requested_stage_cannot_succeed_from_stale_output(tmp_path, monkeypatch):
    resolved = _resolved(tmp_path, "vlm_onnx")
    stale = tmp_path / "work" / "onnx" / "pi05-vlm_op17_nodyn_fp16_cpu.onnx"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    _mock_pipeline(monkeypatch, resolved, skip_product="vlm_onnx")

    with pytest.raises(RuntimeError, match="completed without producing"):
        pipeline.main()

    assert not stale.exists()


def test_ae_onnx_alone_requires_runtime_handoff_tensors(tmp_path, monkeypatch):
    resolved = _resolved(tmp_path, "ae_onnx")
    ae_onnx = tmp_path / "work" / "onnx" / "pi05-action_expert_op17_nodyn_fp16_cpu.onnx"
    ae_onnx.parent.mkdir(parents=True)
    ae_onnx.write_bytes(b"existing")
    vlm_onnx = tmp_path / "work" / "onnx" / "pi05-vlm_op17_nodyn_fp16_cpu.onnx"
    vlm_onnx.write_bytes(b"existing")
    _mock_pipeline(monkeypatch, resolved)

    with pytest.raises(SystemExit, match="runtime handoff tensors"):
        pipeline.main()


def test_quantized_onnx_and_om_publish_only_requested_deployment(tmp_path, monkeypatch):
    steps = "vlm_onnx,ae_onnx,vlm_quant,ae_quant,vlm_om,ae_om,vlm_quant_om,ae_quant_om"
    resolved = _resolved(tmp_path, steps)
    _mock_pipeline(monkeypatch, resolved)

    assert pipeline.main() == 0

    manifest = load_inference_manifest(tmp_path / "bundle", "ascend").manifest
    assert set(manifest.deployments) == {"ascend"}
    vlm = tmp_path / "bundle" / manifest.deployments["ascend"].artifacts["vlm"].path
    action = tmp_path / "bundle" / manifest.deployments["ascend"].artifacts["action_expert"].path
    assert vlm.read_bytes() == b"vlm_quant_om"
    assert action.read_bytes() == b"ae_quant_om"


@pytest.mark.parametrize("steps", ["vlm_quant", "ae_quant", "vlm_quant_om", "ae_quant_om"])
def test_individual_quantized_steps_can_be_rerun(tmp_path, monkeypatch, steps):
    initial_steps = "vlm_onnx,ae_onnx,vlm_quant,vlm_om,ae_om,ae_quant,vlm_quant_om,ae_quant_om"
    initial = _resolved(tmp_path, initial_steps)
    _mock_pipeline(monkeypatch, initial)
    assert pipeline.main() == 0

    rerun = _resolved(tmp_path, steps)
    _mock_pipeline(monkeypatch, rerun)
    assert pipeline.main() == 0

    if steps.endswith("_om"):
        assert load_inference_manifest(tmp_path / "bundle", "ascend").deployment.execution == (
            "vlm",
            "action_expert",
        )


def test_pi05_export_profile_persists_schedule_file_but_not_steps(tmp_path):
    policy = tmp_path / "bundle"
    policy.mkdir()
    config_path = tmp_path / "pi05-export.yaml"
    schedule_path = tmp_path / "schedule.json"

    resolved = _cli.resolve(
        [
            "--config",
            str(config_path),
            "--policy-path",
            str(policy),
            "--steps",
            "vlm_onnx",
            "--schedule-file",
            str(schedule_path),
            "--save-as",
            "scheduled",
        ]
    )

    saved = _cli.load_config(str(config_path))["profiles"]["scheduled"]
    assert resolved.args.schedule_file == str(schedule_path)
    assert saved["schedule_file"] == str(schedule_path)
    assert "steps" not in saved


def test_pi05_export_pipeline_forwards_explicit_schedule_to_manifest(tmp_path, monkeypatch):
    resolved = _resolved(tmp_path, ",".join(_cli.DEFAULT_STEPS))
    schedule_path = tmp_path / "schedule.json"
    _write_json(
        schedule_path,
        {
            "format": "pi05-denoising-schedule-v1",
            "name": "profile-schedule",
            "algorithm": "euler",
            "model_output": "velocity",
            "timesteps": [1.0, 0.7, 0.0],
        },
    )
    resolved.args.schedule_file = str(schedule_path)
    _mock_pipeline(monkeypatch, resolved)

    assert pipeline.main() == 0

    deployment = load_inference_manifest(tmp_path / "bundle", "ascend").deployment
    artifact = deployment.artifacts["denoising_schedule"]
    assert load_pi05_schedule(tmp_path / "bundle" / artifact.path).name == "profile-schedule"

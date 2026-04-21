import argparse
import json
import subprocess
from pathlib import Path

OM_MANIFEST_BASENAME = "config.om.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, required=True, help="The path of pretrained model")
    parser.add_argument("--soc_version", type=str, required=True, help="The Ascend soc version")
    parser.add_argument("--onnx_model_path", type=str, default=None, help="The path to store onnx model")
    parser.add_argument("--om_model_path", type=str, default=None, help="The path to store om model")
    parser.add_argument(
        "--skip_onnx_export",
        action="store_true",
        help="Convert an existing ONNX file without exporting from model.safetensors first",
    )
    args = parser.parse_args()

    pretrained_model_path = args.pretrained_model
    onnx_model_path = args.onnx_model_path
    om_model_path = args.om_model_path
    soc_version = args.soc_version

    if onnx_model_path is None:
        onnx_model_path = pretrained_model_path + "/model.onnx"
    if om_model_path is None:
        om_model_path = pretrained_model_path + "/model.om"

    config_path = pretrained_model_path + "/config.json"
    with open(config_path) as f:
        config = json.load(f)

    return (
        pretrained_model_path,
        config,
        onnx_model_path,
        om_model_path,
        soc_version,
        args.skip_onnx_export,
    )


def _relative_or_absolute_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def write_om_manifest(pretrained_model_path, config, om_model_path):
    policy_type = str(config.get("type", "")).lower().strip()
    if not policy_type:
        raise ValueError("config.json is missing required policy type metadata")
    if policy_type != "act":
        raise ValueError(f"Policy {policy_type} is not supported currently.")

    policy_dir = Path(pretrained_model_path).expanduser()
    if not policy_dir.is_absolute():
        policy_dir = (Path.cwd() / policy_dir).resolve()
    om_path = Path(om_model_path).expanduser()
    if not om_path.is_absolute():
        om_path = (Path.cwd() / om_path).resolve()

    manifest_path = policy_dir / OM_MANIFEST_BASENAME
    manifest = {
        "schema_version": 1,
        "policy_type": policy_type,
        "backend": "ascend_om",
        "artifacts": {"policy": _relative_or_absolute_path(om_path, policy_dir)},
        "execution": ["policy"],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest_path


def _act_input_shape(config):
    input_shape = []
    for key, value in config["input_features"].items():
        # Only keep keys that ACT policy actually consumes;
        # skip unrelated entries like "observation.state.current".
        if key != "observation.state" and not key.startswith("observation.images."):
            continue
        shape = [1] + list(value["shape"])
        input_shape.append(key + ":" + ",".join(map(str, shape)))
    return ";".join(input_shape)


def convert_onnx_to_om(config, onnx_model_path, om_model_path, soc_version):
    input_shape = _act_input_shape(config)
    transfer_om_command = [
        "atc",
        "--framework=5",
        f"--soc_version={soc_version}",
        f"--model={onnx_model_path}",
        f"--output={om_model_path.removesuffix('.om')}",
        f"--input_shape={input_shape}",
    ]
    return subprocess.run(transfer_om_command, check=False).returncode == 0  # nosec B603


def export_act_model(pretrained_model_path, config, onnx_model_path, om_model_path, soc_version):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    input_features = config["input_features"]

    act_input_batch = {}
    input_names = []

    for key in input_features:
        # Only keep keys that ACT policy actually consumes;
        # skip unrelated entries like "observation.state.current".
        if key != "observation.state" and not key.startswith("observation.images."):
            continue

        shape = [1] + list(input_features[key]["shape"])

        input_names.append(key)

        if key == "observation.state":
            act_input_batch[key] = torch.rand(shape, dtype=torch.float32, device="cpu")
        else:
            if "observation.images" not in act_input_batch:
                act_input_batch["observation.images"] = []
            act_input_batch["observation.images"].append(torch.rand(shape, dtype=torch.float32, device="cpu"))

    act_batch = {
        "batch": act_input_batch,
    }

    # Force-disable OM inference modes so that policy initialization always
    # constructs the standard PyTorch model (`self.model = ACT(config)`),
    # regardless of what config.json says.
    #
    # NOTE:
    # - `cli_overrides` is supported by `PreTrainedConfig.from_pretrained(...)`.
    # - `ACTPolicy.from_pretrained(...)` itself does not consume `cli_overrides`.
    #
    # Exporting ONNX/OM requires the original PyTorch graph, so we first build
    # a config with OM flags disabled, then pass that config into ACTPolicy.
    policy_config = PreTrainedConfig.from_pretrained(
        pretrained_model_path,
        cli_overrides=[
            "--is_ascend_om_enabled=false",
            "--is_ascend_om_3403_enabled=false",
        ],
    )
    act_policy = ACTPolicy.from_pretrained(pretrained_model_path, config=policy_config)
    act_policy.model = act_policy.model.to("cpu")
    act_policy.model.eval()

    torch.onnx.export(
        act_policy.model,
        (act_batch,),
        onnx_model_path,
        input_names=input_names,
        opset_version=14,
        output_names=["action"],
        dynamo=False,
    )

    return convert_onnx_to_om(config, onnx_model_path, om_model_path, soc_version)


if __name__ == "__main__":
    (
        pretrained_model_path,
        config,
        onnx_model_path,
        om_model_path,
        soc_version,
        skip_onnx_export,
    ) = parse_args()
    policy_type = config["type"]

    if policy_type == "act":
        if skip_onnx_export:
            if not convert_onnx_to_om(config, onnx_model_path, om_model_path, soc_version):
                raise ValueError("convert_onnx_to_om failed")
        elif not export_act_model(pretrained_model_path, config, onnx_model_path, om_model_path, soc_version):
            raise ValueError("export_act_model failed")
        write_om_manifest(pretrained_model_path, config, om_model_path)
    else:
        raise ValueError(f"Policy {policy_type} is not supported currently.")

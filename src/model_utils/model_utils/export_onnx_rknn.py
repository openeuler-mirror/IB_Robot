import argparse
import json
import os
import subprocess
import sys

import onnx
from onnxsim import simplify


def logger(msg):
    print(f"[export_onnx_rknn]: {msg}")


def strip_extra_outputs(onnx_path, output_path):
    onnx_model = onnx.load(onnx_path)
    keep = ["action"]
    kept_outputs = [o for o in onnx_model.graph.output if o.name in keep]
    if len(kept_outputs) == len(onnx_model.graph.output):
        logger("No extra outputs to strip")
        os.replace(onnx_path, output_path)
        return
    logger(f"Stripping outputs: {[o.name for o in onnx_model.graph.output if o.name not in keep]}")
    while len(onnx_model.graph.output) > 0:
        onnx_model.graph.output.pop()
    for o in kept_outputs:
        onnx_model.graph.output.append(o)
    onnx.save(onnx_model, output_path)
    logger(f"Saved stripped model to {output_path}")


def process_existing_onnx(onnx_path, output_path=None):
    if output_path is None:
        base, ext = os.path.splitext(onnx_path)
        output_path = f"{base}_rknn{ext}"

    logger(f"Step 1/2: Stripping extra outputs from {onnx_path}")
    stripped_path = output_path.replace(".onnx", "_stripped.onnx")
    strip_extra_outputs(onnx_path, stripped_path)

    logger("Step 2/2: Simplifying with onnxsim")
    onnx_model = onnx.load(stripped_path)
    model_simp, check = simplify(onnx_model)
    if not check:
        raise ValueError("Simplified ONNX model could not be validated")
    onnx.save(model_simp, output_path)

    if os.path.exists(stripped_path) and stripped_path != output_path:
        os.remove(stripped_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger(f"Done: {output_path} ({size_mb:.1f}MB)")
    return output_path


def export_from_safetensors(args, config):
    import torch

    try:
        import lerobot

        logger(f"lerobot path: {lerobot.__file__}")
    except ImportError:
        logger("lerobot not available, cannot export from safetensors")
        logger("Use --onnx to process an existing ONNX model instead")
        return None

    from lerobot.policies.act.modeling_act import ACTPolicy

    model_path = args.policy_path
    onnx_raw = os.path.join(model_path, "act_ros2_rknn_raw.onnx")
    onnx_final = os.path.join(model_path, "act_ros2_rknn.onnx")

    input_features = config["input_features"]
    input_names = []
    image_keys = []
    dummy_tensors = []

    for key in input_features:
        if key != "observation.state" and not key.startswith("observation.images."):
            continue
        shape = [1] + list(input_features[key]["shape"])
        dummy_tensors.append(torch.randn(*shape, dtype=torch.float32, device=args.device))
        input_names.append(key)
        if key.startswith("observation.images."):
            image_keys.append(key)

    class ACTONNXWrapper(torch.nn.Module):
        def __init__(self, model, input_names, image_keys):
            super().__init__()
            self.model = model
            self.input_names = input_names
            self.image_keys = image_keys

        def forward(self, *args):
            batch = {name: tensor for name, tensor in zip(self.input_names, args, strict=False)}
            batch["observation.images"] = [batch[key] for key in self.image_keys]
            output = self.model(batch)
            if isinstance(output, dict):
                return output["action"]
            if isinstance(output, tuple):
                return output[0]
            return output

    act_policy = ACTPolicy.from_pretrained(
        model_path,
        cli_overrides=[
            "--is_ascend_om_enabled=false",
            "--is_ascend_om_3403_enabled=false",
        ],
    )
    act_policy.model = act_policy.model.to(args.device)
    act_policy.model.eval()

    wrapped_model = ACTONNXWrapper(act_policy.model, input_names, image_keys)

    logger("Exporting ONNX (opset=13, action-only output)")
    torch.onnx.export(
        wrapped_model,
        tuple(dummy_tensors),
        onnx_raw,
        input_names=input_names,
        opset_version=13,
        output_names=["action"],
        do_constant_folding=True,
        verbose=False,
    )

    return process_existing_onnx(onnx_raw, onnx_final)


def _ensure_rknn_venv(venv_python):
    venv_dir = os.path.dirname(os.path.dirname(venv_python))
    if os.path.exists(venv_python):
        return True
    logger(f"Creating .venv-rknn at {venv_dir}")
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    subprocess.run([venv_python, "-m", "pip", "install", "-q", "rknn-toolkit2", "onnx", "onnxruntime"], check=True)
    logger(".venv-rknn created and dependencies installed")
    return True


def convert_to_rknn(onnx_path, args):
    if args.rknn_output:
        rknn_output = args.rknn_output
    else:
        onnx_dir = os.path.dirname(onnx_path)
        rknn_output = os.path.join(onnx_dir, "model.rknn")
    venv_python = args.rknn_venv_python

    if not venv_python:
        logger("No rknn venv path configured, skipping RKNN conversion")
        return None

    try:
        _ensure_rknn_venv(venv_python)
    except Exception as e:
        logger(f"Failed to create .venv-rknn: {e}")
        logger(
            "  Create manually: python3 -m venv .venv-rknn && .venv-rknn/bin/pip install rknn-toolkit2 onnx onnxruntime"
        )
        return None

    convert_script = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        ".agents",
        "skills",
        "rknn-convert",
        "convert_to_rknn.py",
    )
    if not os.path.exists(convert_script):
        logger(f"RKNN convert script not found: {convert_script}")
        return None

    cmd = [venv_python, convert_script, "--onnx", onnx_path, "--output", rknn_output, "--mode", args.rknn_mode]
    logger(f"Converting to RKNN: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger("RKNN conversion failed!")
        return None
    logger(f"RKNN model saved to {rknn_output}")
    return rknn_output


def parse_args():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    parser = argparse.ArgumentParser(description="Export ONNX for RKNN (RK3588 NPU)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--policy_path", type=str, help="Path to pretrained policy model directory (export from safetensors)"
    )
    group.add_argument("--onnx", type=str, help="Path to existing ONNX model (strip + simplify only)")

    parser.add_argument(
        "--output", type=str, default=None, help="Output ONNX path (default: auto-named with _rknn suffix)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for export (cpu or cuda, only used with --policy_path)"
    )
    parser.add_argument("--convert_rknn", action="store_true", help="Also convert ONNX to RKNN after export")
    parser.add_argument(
        "--rknn_output", type=str, default=None, help="Output RKNN path (default: same as onnx with .rknn)"
    )
    parser.add_argument(
        "--rknn_mode",
        type=str,
        default="float16",
        choices=["float16", "int8", "hybrid"],
        help="RKNN conversion mode (default: float16)",
    )
    parser.add_argument(
        "--rknn_venv_python", type=str, default=None, help="Path to .venv-rknn python (auto-detected if not set)"
    )

    args = parser.parse_args()

    if args.rknn_venv_python is None:
        candidate = os.path.join(project_root, ".venv-rknn", "bin", "python")
        if os.path.exists(candidate):
            args.rknn_venv_python = candidate

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.onnx:
        if not os.path.exists(args.onnx):
            logger(f"ONNX file not found: {args.onnx}")
            sys.exit(1)
        onnx_result = process_existing_onnx(args.onnx, args.output)
    else:
        config_path = os.path.join(args.policy_path, "config.json")
        if not os.path.exists(config_path):
            logger(f"config.json not found: {args.policy_path}")
            sys.exit(1)
        with open(config_path) as f:
            config = json.load(f)

        onnx_result = export_from_safetensors(args, config)
        if onnx_result is None:
            logger("Export from safetensors failed. Use --onnx with an existing ONNX model.")
            sys.exit(1)

    if args.convert_rknn and onnx_result:
        convert_to_rknn(onnx_result, args)

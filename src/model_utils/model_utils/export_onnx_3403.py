import argparse
import json
import os

import onnx
import torch
from onnxsim import simplify

import lerobot


def logger(msg):
    print(f"[export_onnx]: {msg}")

logger(f"lerobot path: {lerobot.__file__}")

# Wrapper to rebuild observation.images list before calling ACT model,
# matching what ACTPolicy.forward does internally.
# ONNX export requires inputs to be passed as positional tensors, so this wrapper
# receives *args and reconstructs the batch dict internally.
class ACTONNXWrapper(torch.nn.Module):
    def __init__(self, model, input_names, image_keys):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.image_keys = image_keys

    def forward(self, *args):
        batch = {name: tensor for name, tensor in zip(self.input_names, args)}
        batch["observation.images"] = [batch[key] for key in self.image_keys]
        return self.model(batch)


def export_act(args, config):
    from lerobot.policies.act.modeling_act import ACTPolicy
    model_path = args.policy_path
    onnx_path = os.path.join(model_path, "act_ros2.onnx")
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

    policy = ACTPolicy.from_pretrained(model_path)
    policy.model = policy.model.to(args.device)
    policy.model.eval()
    wrapped_model = ACTONNXWrapper(policy.model, input_names, image_keys)

    logger("Exporting onnx")
    torch.onnx.export(
        wrapped_model,
        tuple(dummy_tensors),
        onnx_path,
        input_names=input_names,
        # mindcmd 要求 opset 版本为 13, atc 要求为 11~15
        opset_version=13,
        output_names=["action"],
        external_data=True,
        verbose=False,
        do_constant_folding=False,
    )

    logger("Simplify onnx")
    onnx_model = onnx.load(onnx_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    if not check:
        raise ValueError("Simplified ONNX model could not be validated")
    onnx.save(model_simp, os.path.join(model_path, "act_ros2_simplified.onnx"))
    print("finished exporting onnx")

def parse_args():
    parser = argparse.ArgumentParser(description="export_onnx")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (e.g. cpu, cuda)")
    parser.add_argument(
        "--policy_path", type=str, required=True, help="Path to pretrained policy model directory"
    )
    parser.add_argument("--policy_type", type=str, default="act", help="Type of policy model (e.g. act)")
    args = parser.parse_args()

    config_path = os.path.join(args.policy_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    return args, config

if __name__ == "__main__":
    args, config = parse_args()
    if args.policy_type == "act":
        export_act(args, config)
    else:
        logger(f"Invalid option: {args.policy_type}")
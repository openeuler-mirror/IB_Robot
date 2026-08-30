#!/usr/bin/env python3
"""Generate mock-mode robot YAML for inference verification.

Two modes:
  policy      one policy pipeline (for `First inference received` checks)
  perception  perception_services with typed model service endpoints

The deployment name is auto-resolved from the bundle's inference_manifest.json
so callers only need `--device cpu|cuda` (deployment keys differ per bundle:
cpu / torch-cpu / torch_cpu / torch-cuda / torch_cuda).

Usage:
  python3 generate_verify_yaml.py policy --model models/pi05 --device cuda -o /tmp/pi05_cuda.yaml
  python3 generate_verify_yaml.py perception --device cpu -o /tmp/perception_cpu.yaml
  python3 generate_verify_yaml.py perception --device cpu --only siglip2,ram_plus -o /tmp/p.yaml
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROBOT_YAML = REPO_ROOT / "src/robot_config/config/robots/so101_single_arm.yaml"

# candidate deployment keys, most specific first
DEPLOYMENT_CANDIDATES = {
    "cpu": ["cpu", "torch-cpu", "torch_cpu"],
    "cuda": ["torch-cuda", "torch_cuda"],
}

# default perception service presets (host-runnable bundles; endpoint/node names
# follow references/host.md conventions)
PERCEPTION_PRESETS = {
    "siglip2": {
        "bundle_path": "models/siglip2_so400m_patch14_384",
        "adapter_class": "perception_service.model_service_plugins:SigLIP2EncodeEmbeddingsPlugin",
        "service_type": "ibrobot_msgs/srv/EncodeEmbeddings",
        "endpoint": "/perception/siglip2/encode_embeddings",
        "node_name": "siglip2_image",
    },
    "ram_plus": {
        "bundle_path": "models/ram_plus_swin_large_14m",
        "adapter_class": "perception_service.model_service_plugins:RAMPlusRecognizeTagsPlugin",
        "service_type": "ibrobot_msgs/srv/RecognizeTags",
        "endpoint": "/perception/ram_plus/recognize_tags",
        "node_name": "ram_plus_tags",
    },
    "sam2": {
        "bundle_path": "models/sam2.1_hiera_tiny",
        "adapter_class": "perception_service.model_service_plugins:SAM2GenerateMasksPlugin",
        "service_type": "ibrobot_msgs/srv/GenerateMasks",
        "endpoint": "/perception/sam2/generate_masks",
        "node_name": "sam2_masks",
    },
    "grounding_dino": {
        "bundle_path": "models/grounding_dino_swint_seq8_1280x720",
        "adapter_class": "perception_service.model_service_plugins:GroundingDINORawDetectPlugin",
        "service_type": "ibrobot_msgs/srv/GroundingDetect",
        "endpoint": "/perception/grounding_dino/detect",
        "node_name": "grounding_dino",
    },
}


def resolve_deployment(model_path: Path, device: str) -> str:
    manifest = model_path / "inference_manifest.json"
    available = list(json.loads(manifest.read_text()).get("deployments", {}))
    for key in DEPLOYMENT_CANDIDATES[device]:
        if key in available:
            return key
    raise SystemExit(
        f"ERROR: {model_path} has no {device} deployment; available: {available}. "
        "Ascend-only bundles cannot be verified on host - skip them."
    )


def base_mock_config(robot_yaml: Path) -> dict:
    c = copy.deepcopy(yaml.safe_load(robot_yaml.read_text()))
    c["robot"]["simulation"]["platform"] = "mock"
    c["robot"]["simulation"]["scene"] = None
    c["robot"]["default_control_mode"] = "model_inference"
    mi = c["robot"]["control_modes"]["model_inference"]
    mi["scheduler_enabled"] = False
    c["robot"]["voice_tts"]["enabled"] = False
    return c


def build_policy(args) -> dict:
    c = base_mock_config(Path(args.robot))
    model_path = Path(args.model)
    deployment = args.deployment or resolve_deployment(model_path, args.device)
    c["robot"]["control_modes"]["model_inference"]["inference"]["pipelines"] = {
        "policy": {
            "model_path": str(model_path),
            "deployment": deployment,
            "execution_mode": "monolithic",
            "request_timeout": 300.0,
            "default_task": "pick up the banana",
        }
    }
    return c


def build_perception(args) -> dict:
    c = base_mock_config(Path(args.robot))
    mi = c["robot"]["control_modes"]["model_inference"]
    mi["inference"]["enabled"] = False
    mi["inference"]["pipelines"] = {}
    names = args.only.split(",") if args.only else list(PERCEPTION_PRESETS)
    services = []
    for name in names:
        preset = dict(PERCEPTION_PRESETS[name])
        try:
            deployment = args.deployment or resolve_deployment(Path(preset["bundle_path"]), args.device)
        except SystemExit as exc:
            # an ascend-only (or otherwise host-unrunnable) bundle is skipped
            # with a warning instead of blocking the whole verification
            print(f"WARN: skipping preset '{name}': {exc}", file=sys.stderr)
            continue
        services.append(
            {
                "id": name,
                "enabled": True,
                # required=False so an unavailable bundle degrades instead of
                # blocking the whole launch (report it, don't crash on it)
                "required": False,
                **preset,
                "deployment": deployment,
            }
        )
    c["robot"]["perception_services"] = {"services": services}
    return c


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["policy", "perception"])
    p.add_argument("--robot", default=str(DEFAULT_ROBOT_YAML), help="base robot YAML")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--deployment", default="", help="override deployment key (default: auto from manifest)")
    p.add_argument("--model", default="", help="policy mode: model bundle path")
    p.add_argument("--only", default="", help="perception mode: comma-separated preset names")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    if args.mode == "policy":
        if not args.model:
            p.error("policy mode requires --model")
        c = build_policy(args)
    else:
        c = build_perception(args)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.dump(c, default_flow_style=False, allow_unicode=True, sort_keys=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

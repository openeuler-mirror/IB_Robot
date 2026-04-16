import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


class LossUtils:
    def __init__(self, args):
        self.args = args
        self.policy = self.prepare_policy()

    def run(self):
        if self.args.generate_target:
            self.generate_target()
        else:
            self.compute_loss()

    def compute_loss(self):
        print("computing loss...")
        with open(self.args.target_path, encoding="utf-8") as f:
            targets = json.load(f)

        for i in range(len(targets)):
            targets[i] = torch.tensor(targets[i])

        batches = self.load_batches_as_tensors()
        start_time = time.perf_counter()
        preds = self.forward(batches)
        end_time = time.perf_counter()
        print(end_time - start_time)

        if len(targets) != len(preds):
            raise ValueError(f"Length mismatch: targets {len(targets)} vs preds {len(preds)}")

        arr_loss = []
        for i in range(len(targets)):
            loss = torch.nn.functional.l1_loss(preds[i], targets[i], reduction="mean")
            print(f"loss[{i}]: {loss.mean()}")
            arr_loss.append(loss.item())
        avg_loss = sum(arr_loss) / len(arr_loss)
        print(f"avg loss: {avg_loss}")

    def prepare_policy(self):
        if self.args.policy_type == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy

            policy_path = self.args.policy_path
            policy = ACTPolicy.from_pretrained(policy_path)
        elif self.args.policy_type == "pi05":
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy

            policy_path = self.args.policy_path
            policy = PI05Policy.from_pretrained(policy_path)
        else:
            raise NotImplementedError(f"Policy type {self.args.policy_type} not implemented")

        print(f"model loaded: {policy_path}")
        return policy

    def load_batches_as_tensors(self):
        with open(self.args.batch_path, encoding="utf-8") as f:
            raw_batches = json.load(f)
        processed_batches = []
        for b in raw_batches:
            processed_batch = {}
            for k, v in b.items():
                if "side_view" in k:
                    continue
                elif k == "observation.images.hand_view":
                    processed_batch["observation.images.wrist"] = np.array(v).astype(np.float32)
                elif k == "observation.images.top_view":
                    processed_batch["observation.images.top"] = np.array(v).astype(np.float32)
                else:
                    processed_batch[k] = np.array(v).astype(np.float32)
            processed_batches.append(processed_batch)
        return processed_batches

    def forward(self, batches):
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=self.policy, pretrained_path=self.args.policy_path
        )
        device = get_safe_torch_device(self.policy.config.device)
        outputs = []
        for i in tqdm(range(len(batches)), desc="forwarding"):
            # Fix random seed per batch so that diffusion/flow-matching noise is
            # deterministic across runs.  Without this, PI05's sample_noise()
            # generates different Gaussian noise each time → different actions.
            torch.manual_seed(self.args.seed + i)

            # --- Scheme C: file-based noise transfer for cross-machine comparison ---
            # When --noise-dir is specified:
            #   generate-target (GPU): generate noise → save .npy → inject into policy
            #   compute-loss   (NPU): load .npy → inject into policy → OM uses same noise
            if self.args.noise_dir:
                noise_path = os.path.join(self.args.noise_dir, f"noise_{i:04d}.npy")
                if self.args.generate_target:
                    cfg = self.policy.config
                    noise_shape = (1, cfg.chunk_size, cfg.max_action_dim)
                    noise = torch.normal(
                        mean=0.0, std=1.0, size=noise_shape, dtype=torch.float32
                    )
                    os.makedirs(self.args.noise_dir, exist_ok=True)
                    np.save(noise_path, noise.numpy())
                else:
                    noise = torch.from_numpy(np.load(noise_path)).float()
                self.policy._external_noise = noise.to(device)

            output = predict_action(
                observation=batches[i],
                policy=self.policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=self.policy.config.use_amp,
            )
            outputs.append(output)
        return outputs

    def generate_target(self):
        print("generating target json from batches...")
        if self.args.noise_dir:
            print(f"  noise files will be saved to: {self.args.noise_dir}")

        batches = self.load_batches_as_tensors()
        outputs = self.forward(batches)

        print(f"saving output json: length={len(outputs)}")

        for i in range(len(outputs)):
            outputs[i] = outputs[i].tolist()

        output_path = self.args.target_path
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4)

        print(f"output saved at {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Model Loss Comparison")
    parser.add_argument("--batch_path", type=str, required=True, help="Path to input batches json file")
    parser.add_argument("--target_path", type=str, required=True, help="Path to save target json file")
    parser.add_argument(
        "--policy_path", type=str, required=True, help="Path to pretrained policy model directory"
    )
    parser.add_argument(
        "--policy_type", type=str, default="act", help="Type of policy model (e.g. act, diffuser, ddpg)"
    )
    parser.add_argument(
        "--generate-target", action="store_true", help="Reading batches and generating target json file"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic inference (fixes diffusion noise).",
    )
    parser.add_argument(
        "--noise-dir", type=str, default=None,
        help="Directory for noise file transfer (Scheme C). "
             "generate-target: saves noise_{NNNN}.npy files here. "
             "compute-loss: loads noise files from here to ensure identical "
             "noise across GPU (PyTorch) and NPU (OM) machines.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loss_utils = LossUtils(args)
    loss_utils.run()

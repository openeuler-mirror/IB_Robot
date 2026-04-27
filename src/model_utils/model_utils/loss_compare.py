import argparse
import contextlib
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
        print(f"inference time: {end_time - start_time:.3f}s")

        if len(targets) != len(preds):
            raise ValueError(f"Length mismatch: targets {len(targets)} vs preds {len(preds)}")

        arr_l1 = []
        arr_cos = []
        for i in range(len(targets)):
            l1 = torch.nn.functional.l1_loss(preds[i], targets[i], reduction="mean").item()
            cos = torch.nn.functional.cosine_similarity(
                preds[i].flatten().unsqueeze(0),
                targets[i].flatten().unsqueeze(0),
            ).item()
            arr_l1.append(l1)
            arr_cos.append(cos)

        # Print summary table (unnormalized / physical action space)
        print(f"\n=== Unnormalized action space (post-postprocessor) ===")
        print(f"{'Batch':>6} {'L1 Loss':>12} {'Cosine Sim':>12}")
        print("-" * 32)
        for i in range(len(arr_l1)):
            print(f"{i:>6} {arr_l1[i]:>12.6f} {arr_cos[i]:>12.6f}")
        print("-" * 32)
        avg_l1 = sum(arr_l1) / len(arr_l1)
        avg_cos = sum(arr_cos) / len(arr_cos)
        print(f"{'Avg':>6} {avg_l1:>12.6f} {avg_cos:>12.6f}")

        # ------------------------------------------------------------------
        # Independent sanity check: compare in *normalized* (pre-postprocessor)
        # action space.  ``self._raw_preds`` was populated by the postprocessor
        # hook in ``forward()``.  This isolates the model's true output error
        # from any unnormalization scale-up — useful for diagnosing whether a
        # large unnormalized L1 is real model drift or just dataset stats
        # blowing the number up.
        # ------------------------------------------------------------------
        raw_preds = getattr(self, "_raw_preds", None)
        raw_targets_path = getattr(self.args, "raw_target_path", None)

        if raw_preds is not None and raw_targets_path and os.path.exists(raw_targets_path):
            with open(raw_targets_path, encoding="utf-8") as f:
                raw_targets = json.load(f)
            raw_targets = [torch.tensor(t) for t in raw_targets]

            if len(raw_targets) != len(raw_preds):
                print(f"WARN: raw target/pred length mismatch: "
                      f"{len(raw_targets)} vs {len(raw_preds)}; skipping raw L1")
            else:
                arr_raw_l1 = []
                arr_raw_cos = []
                for i in range(len(raw_targets)):
                    rp = raw_preds[i].detach().cpu().float()
                    rt = raw_targets[i].float()
                    if rp.shape != rt.shape:
                        print(f"WARN: raw shape mismatch on batch {i}: "
                              f"pred={tuple(rp.shape)} target={tuple(rt.shape)}")
                        continue
                    arr_raw_l1.append(torch.nn.functional.l1_loss(rp, rt, reduction="mean").item())
                    arr_raw_cos.append(torch.nn.functional.cosine_similarity(
                        rp.flatten().unsqueeze(0), rt.flatten().unsqueeze(0)
                    ).item())

                if arr_raw_l1:
                    print(f"\n=== Normalized action space (pre-postprocessor) ===")
                    print(f"{'Batch':>6} {'raw L1':>12} {'raw Cos':>12}")
                    print("-" * 32)
                    for i in range(len(arr_raw_l1)):
                        print(f"{i:>6} {arr_raw_l1[i]:>12.6f} {arr_raw_cos[i]:>12.6f}")
                    print("-" * 32)
                    print(f"{'Avg':>6} {sum(arr_raw_l1)/len(arr_raw_l1):>12.6f} "
                          f"{sum(arr_raw_cos)/len(arr_raw_cos):>12.6f}")
        elif raw_preds is not None and raw_targets_path:
            print(f"\nNOTE: raw target file not found at {raw_targets_path}; "
                  f"run --generate-target with --raw-target-path to create it.")

        # Diagnostic dump for batch 0 — physical-space numbers so you can judge
        # whether the unnormalized L1 is "small" (e.g. mm in cartesian) or
        # "large" (e.g. radians in joint space).
        if len(preds) > 0 and len(targets) > 0:
            p, t = preds[0], targets[0]
            print(f"\n=== batch 0 diagnostic (unnormalized) ===")
            print(f"  pred   shape : {tuple(p.shape)}  dtype={p.dtype}")
            print(f"  target shape : {tuple(t.shape)}  dtype={t.dtype}")
            print(f"  pred   range : [{p.min().item():+.4f}, {p.max().item():+.4f}]  "
                  f"mean={p.mean().item():+.4f}  std={p.std().item():.4f}")
            print(f"  target range : [{t.min().item():+.4f}, {t.max().item():+.4f}]  "
                  f"mean={t.mean().item():+.4f}  std={t.std().item():.4f}")
            diff = (p - t).abs()
            if diff.ndim >= 2:
                # Per-action-dim stats (last axis is action_dim)
                reduce_dims = tuple(range(diff.ndim - 1))
                per_dim_l1 = diff.mean(dim=reduce_dims)
                per_dim_max = diff.amax(dim=reduce_dims)
                np.set_printoptions(precision=4, suppress=True, linewidth=160)
                print(f"  per-dim L1   : {per_dim_l1.cpu().numpy()}")
                print(f"  per-dim Linf : {per_dim_max.cpu().numpy()}")
            print(f"  pred   first row: {p.flatten()[:p.shape[-1]].cpu().numpy()}")
            print(f"  target first row: {t.flatten()[:t.shape[-1]].cpu().numpy()}")

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

        # Optional dtype cast — defaults to whatever the checkpoint provides
        # (BF16 for PI05).  Use ``--model_dtype fp16`` to match the OM/ORT
        # deployment dtype and isolate BF16↔FP16 conversion error from any
        # real ONNX-export error.
        model_dtype = getattr(self.args, "model_dtype", "native")
        if model_dtype == "fp16":
            policy.model = policy.model.half()
            print("  Cast policy.model to float16")
        elif model_dtype == "bf16":
            policy.model = policy.model.bfloat16()
            print("  Cast policy.model to bfloat16")
        elif model_dtype == "fp32":
            policy.model = policy.model.float()
            print("  Cast policy.model to float32")
        elif model_dtype != "native":
            raise ValueError(f"unknown --model_dtype: {model_dtype}")

        # Log actual running dtype for clarity.
        try:
            sample_param = next(policy.model.parameters())
            print(f"  Running PT policy in dtype={sample_param.dtype}")
        except (StopIteration, AttributeError):
            pass

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

        # ------------------------------------------------------------------
        # Capture raw (pre-postprocessor) action by wrapping postprocessor.
        # The postprocessor takes the policy's raw normalized action and
        # un-normalizes it; by intercepting its input we get the model's
        # actual output without any dataset-stats scale-up.
        # ------------------------------------------------------------------
        raw_preds: list[torch.Tensor] = []
        original_postprocessor = postprocessor

        def _wrapped_postprocessor(action, *args, **kwargs):
            # Be defensive — never break the inference pipeline because
            # of a diagnostic hook.
            with contextlib.suppress(Exception):
                raw_preds.append(action.detach().cpu().clone())
            return original_postprocessor(action, *args, **kwargs)

        postprocessor = _wrapped_postprocessor

        device = get_safe_torch_device(self.policy.config.device)
        # Resolve model dtype once — noise has to match the action_expert
        # weight dtype, otherwise action_in_proj fails with
        # ``mat1 and mat2 must have the same dtype``.
        try:
            model_dtype = next(self.policy.model.parameters()).dtype
        except (StopIteration, AttributeError):
            model_dtype = torch.float32
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
                # Cast noise to model dtype so action_in_proj matmul matches
                # (noise files stay fp32 on disk for portability across runs
                # with different model dtypes).
                self.policy._external_noise = noise.to(device=device, dtype=model_dtype)

            output = predict_action(
                observation=batches[i],
                policy=self.policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=self.policy.config.use_amp,
            )
            outputs.append(output)

        # Stash raw preds for compute_loss / generate_target to use.
        self._raw_preds = raw_preds
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

        # Also dump raw (pre-postprocessor / normalized-space) targets so the
        # NPU side can perform an apples-to-apples comparison without the
        # unnormalization scale-up.
        raw_target_path = getattr(self.args, "raw_target_path", None)
        raw_preds = getattr(self, "_raw_preds", None)
        if raw_target_path and raw_preds:
            raw_dump = [t.tolist() for t in raw_preds]
            with open(raw_target_path, "w", encoding="utf-8") as f:
                json.dump(raw_dump, f, indent=4)
            print(f"raw (normalized) target saved at {raw_target_path}")


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
        "--model_dtype", type=str, default="native",
        choices=["native", "fp16", "bf16", "fp32"],
        help="Cast PT model to this dtype before forward. "
             "'native' (default) keeps the checkpoint dtype "
             "(BF16 for PI05). Use 'fp16' for apples-to-apples "
             "comparison with OM/ORT deployment.",
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
    parser.add_argument(
        "--raw-target-path", type=str, default=None,
        help="Optional path to dump/read normalized-space (pre-postprocessor) "
             "actions. generate-target: writes raw target JSON next to the "
             "regular target. compute-loss: reads it and prints an extra L1 / "
             "Cosine table in normalized action space — useful for separating "
             "real model drift from unnormalization scale-up.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loss_utils = LossUtils(args)
    loss_utils.run()

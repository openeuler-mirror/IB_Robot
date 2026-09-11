"""No-ROS CPU microbenchmark of baseline versus working-tree plan storage.

Run from the implementation checkout after sourcing .shrc_local. Baseline
modules are loaded from git show into isolated in-memory package namespaces;
the index, checkout and submodules are never changed.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
import types
from collections import deque
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="5e38a8d3")
    parser.add_argument("--cycles", type=int, default=300)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("cycles must be positive")
    root = Path(__file__).resolve().parents[3]

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root)

    baseline = git("rev-parse", "--verify", f"{args.baseline}^{{commit}}").decode().strip()
    hashes = {}
    modules = {}
    for version in ("baseline", "refactor"):
        package = types.ModuleType(f"plan_bench_{version}")
        package.__path__ = []
        sys.modules[package.__name__] = package
        for name in ("chunk_planning", "temporal_smoother", "action_blending", "active_plan"):
            if version == "baseline" and name == "active_plan":
                continue
            relative = f"src/action_dispatch/action_dispatch/{name}.py"
            source = git("show", f"{baseline}:{relative}") if version == "baseline" else (root / relative).read_bytes()
            hashes[f"{version}/{name}"] = hashlib.sha256(source).hexdigest()
            qualified = f"{package.__name__}.{name}"
            module = importlib.util.module_from_spec(importlib.util.spec_from_loader(qualified, loader=None))
            sys.modules[qualified] = module
            exec(compile(source, f"{version}:{relative}", "exec"), module.__dict__)
            modules[version, name] = module

    torch.set_num_threads(1)
    print(
        json.dumps(
            {
                "baseline": baseline,
                "head": git("rev-parse", "HEAD").decode().strip(),
                "head_tree": git("rev-parse", "HEAD^{tree}").decode().strip(),
                "working_sources_sha256": hashes,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
                "cycles": args.cycles,
                "warmup_cycles": 20,
                "scope": "post-decode refill and snapshot/consume tick; no ROS, executor, inference or contention",
            },
            sort_keys=True,
        )
    )
    for steps, dimension in ((50, 6), (100, 14), (1000, 32)):
        actions = np.random.default_rng(42).normal(size=(steps, dimension)).astype(np.float32)
        skip = 5
        ticks = (steps - skip) // 2
        for storage in ("queue", "smoother"):
            outputs = {}
            for version in ("baseline", "refactor"):
                planner = modules[version, "chunk_planning"].FullChunkPlanner()
                smoother = (
                    modules[version, "temporal_smoother"].TemporalSmootherManager(
                        enabled=True, chunk_size=steps, device="cpu"
                    )
                    if storage == "smoother"
                    else None
                )
                queue = deque(maxlen=steps)
                if version == "baseline":
                    blending = modules[version, "action_blending"]
                    consumer = (
                        blending.TemporalEnsembleBlender(smoother) if smoother else blending.PassthroughBlender(queue)
                    )
                else:
                    owner_module = modules[version, "active_plan"]
                    consumer = owner_module.ActivePlan(capacity=steps, watermark=20, smoother=smoother)
                    source = owner_module.PlanSource("microbenchmark", request_generation=1)
                refill_ns, tick_ns = [], []
                for cycle in range(args.cycles + 20):
                    start = time.perf_counter_ns()
                    if version == "baseline":
                        if smoother:
                            smoother.update(actions, skip)
                        else:
                            candidate = planner.plan(actions, actions_executed=skip)
                            modules[version, "chunk_planning"].apply_bounded(queue, candidate)
                    else:
                        candidate = planner.plan(actions, actions_executed=skip)
                        consumer.accept(candidate, source, action_dimension=dimension)
                    elapsed = time.perf_counter_ns() - start
                    if cycle >= 20:
                        refill_ns.append(elapsed)
                    values = []
                    for _ in range(ticks):
                        start = time.perf_counter_ns()
                        if version == "baseline":
                            remaining = smoother.plan_length if smoother else len(queue)
                        else:
                            remaining = consumer.snapshot().remaining
                        value = consumer.take_action().action
                        elapsed = time.perf_counter_ns() - start
                        assert remaining > 0 and value is not None
                        if cycle >= 20:
                            tick_ns.append(elapsed)
                        if cycle == args.cycles + 19:
                            values.append(value.copy())
                    if values:
                        outputs[version] = np.asarray(values)
                result = {
                    "shape": [steps, dimension],
                    "skip": skip,
                    "ticks_per_refill": ticks,
                    "storage": storage,
                    "version": version,
                }
                for label, samples in (("refill", refill_ns), ("tick", tick_ns)):
                    result[label] = {
                        "samples": len(samples),
                        **{
                            key: round(float(value) / 1000, 3)
                            for key, value in zip(
                                ("p50_us", "p95_us", "p99_us", "max_us"),
                                (*np.percentile(samples, [50, 95, 99]), max(samples)),
                                strict=True,
                            )
                        },
                    }
                print(json.dumps(result, sort_keys=True))
            np.testing.assert_allclose(outputs["baseline"], outputs["refactor"], rtol=1e-6, atol=1e-6)
    print("PASS: all six final-cycle baseline/refactor output arrays agree (rtol=atol=1e-6)")


if __name__ == "__main__":
    main()

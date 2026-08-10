"""Lazy pinned-LIBERO provider seam for plan metadata and init-state counts."""

from __future__ import annotations

from benchmark_libero.init_state_loader import load_trusted_init_states, resolve_init_states_path
from benchmark_libero.plan_resolver import ProviderTask


class NativeLiberoPlanProvider:
    """Read task snapshots from the pinned provider without creating an environment."""

    def resolve_suite(self, suite: str, task_order_index: int) -> tuple[ProviderTask, ...]:
        if suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}:
            raise ValueError(f"native provider suite {suite!r} is unsupported; libero_100 is composed by the resolver")

        from libero.libero import benchmark as libero_benchmark  # noqa: PLC0415
        from libero.libero import get_libero_path  # noqa: PLC0415

        try:
            suite_factory = libero_benchmark.get_benchmark_dict()[suite]
        except KeyError as exc:
            raise ValueError(f"pinned LIBERO provider does not expose suite {suite!r}") from exc
        native_suite = suite_factory(task_order_index=task_order_index)
        tasks: list[ProviderTask] = []
        for index in range(native_suite.get_num_tasks()):
            native_task = native_suite.get_task(index)
            init_states_path = resolve_init_states_path(native_task, get_libero_path)
            init_states = load_trusted_init_states(init_states_path)
            init_state_count = int(len(init_states)) if init_states is not None else 0
            if init_state_count <= 0:
                raise ValueError(f"suite {suite!r} task {index} has no native init states")
            tasks.append(
                ProviderTask(
                    name=str(native_task.name),
                    prompt=str(native_task.language),
                    problem_folder=str(native_task.problem_folder),
                    bddl_file=str(native_task.bddl_file),
                    init_states_file=str(native_task.init_states_file),
                    init_state_count=init_state_count,
                )
            )
        return tuple(tasks)

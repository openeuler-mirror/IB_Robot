"""Manifest-validated OM model resources and execution through Ascend ACL."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

import numpy as np

from inference_manifest import ArtifactBindings, TensorBinding
from inference_service.backends.ascend.acl_runtime import AclRuntimeLease, check_acl_ret
from inference_service.backends.errors import BackendInferenceError, BackendLoadError

if TYPE_CHECKING:
    from inference_service.codecs import BoundInputs
else:
    BoundInputs = Any

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

_ACL_DTYPES = {
    0: np.dtype("float32"),
    1: np.dtype("float16"),
    2: np.dtype("int8"),
    3: np.dtype("int32"),
    4: np.dtype("uint8"),
    6: np.dtype("int16"),
    9: np.dtype("int64"),
    11: np.dtype("float64"),
    12: np.dtype("bool"),
}


def numpy_dtype(dtype: str) -> np.dtype:
    if dtype != "bfloat16":
        return np.dtype(dtype)
    try:
        return np.dtype(dtype)
    except TypeError:
        try:
            import ml_dtypes
        except ImportError as exc:
            raise BackendLoadError(
                "Ascend bfloat16 bindings require NumPy bfloat16 support or ml_dtypes",
                code="unsupported_runtime_dtype",
            ) from exc
        return np.dtype(ml_dtypes.bfloat16)


def _result_value(value: object, operation: str) -> object:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], int):
        result, ret = value
        check_acl_ret(ret, operation)
        return result
    return value


def _binding_by_index(bindings: tuple[TensorBinding, ...], direction: str) -> dict[int, TensorBinding]:
    result: dict[int, TensorBinding] = {}
    for binding in bindings:
        if binding.index is None:
            raise BackendLoadError(
                f"Ascend {direction} binding {binding.semantic!r} requires an explicit runtime index",
                code=f"invalid_{direction}_bindings",
            )
        result[int(binding.index)] = binding
    return result


@dataclass(frozen=True)
class AclTensorDescriptor:
    index: int
    name: str | None
    dtype: np.dtype | None
    shape: tuple[int, ...] | None
    size: int


@dataclass(frozen=True)
class AclDeviceBuffer:
    pointer: object
    size: int


@dataclass(frozen=True)
class AclAsyncExecution:
    """Submitted ACL execution whose output becomes readable after completion."""

    stream: object
    completion: Future[dict[int, np.ndarray]]


@dataclass
class _DatasetBuffer:
    pointer: object
    data_buffer: object
    size: int
    owned: bool


class AclModel:
    """One loaded OM model with deterministic datasets, buffers, and host staging.

    并发契约:

    同一个 ``AclModel`` 实例的 ``execute()`` 与 ``execute_bank()`` **不可并发调用**
    ——两者共享 ``self.model_id`` 与 ACL 上下文，并发会破坏 dataset 状态。
    调用方负责串行化：legacy 单 dataset 路径由 ``AscendOmModelSession._run_role``
    经 admission control 串行；stateful 双 bank 路径由 ``SileroVadAclRunner`` /
    ``StatefulAclFullSubNetRunner`` 持 ``self._lock`` 保证 ``execute_bank`` 串行。

    ``prepare_dataset_banks`` 与 legacy ``prepare_datasets`` 互斥：前者构建
    ``_dataset_banks`` 后，``execute()``（走 legacy ``input_dataset``）不再可用，
    调用方须全程用 ``execute_bank``。两条路径不会在同一实例上混用。
    """

    def __init__(
        self,
        lease: AclRuntimeLease,
        role: str,
        path: Path,
        bindings: ArtifactBindings,
        *,
        async_execution: bool = False,
    ) -> None:
        self._lease = lease
        self._acl = lease.acl
        self.role = role
        self.path = path
        self.bindings = bindings
        self.model_id: object | None = None
        self.model_desc: object | None = None
        self.input_descriptors: tuple[AclTensorDescriptor, ...] = ()
        self.output_descriptors: tuple[AclTensorDescriptor, ...] = ()
        self.input_dataset: object | None = None
        self.output_dataset: object | None = None
        self.input_buffers: list[_DatasetBuffer] = []
        self.output_buffers: list[_DatasetBuffer] = []
        self.output_host_buffers: list[object | None] = []
        # Stateful 推理路径使用预创建的多 dataset 做 hidden/cell ping-pong，
        # 避免 recurrent state 在 Host/Device 间反复拷贝。
        self._dataset_banks: list[tuple[object, list[_DatasetBuffer], object, list[_DatasetBuffer]]] = []
        self._dataset_bank_host_buffers: list[list[object | None]] = []
        self._host_output_indices: set[int] = set()
        self._owned_device_buffers: list[_DatasetBuffer] = []
        self._closed = False
        self._closing = False
        # Legacy execution keeps the pre-scheduler runtime surface: no lock or
        # completion worker is created. Scheduled execution owns the async
        # gate and worker used by stream completion callbacks.
        self._execution_lock = RLock() if async_execution else None
        self._completion_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"acl-{role}-complete") if async_execution else None
        )
        self._quarantined_datasets: list[tuple[object, object, list[_DatasetBuffer], object, list[_DatasetBuffer]]] = []
        self._quarantined_shared_stream: object | None = None

    def load_descriptor(self) -> None:
        try:
            self._lease.bind_current_thread()
            self.model_id, ret = self._acl.mdl.load_from_file(str(self.path))
            check_acl_ret(ret, f"acl.mdl.load_from_file({self.role})")
            self.model_desc = self._acl.mdl.create_desc()
            if self.model_desc is None:
                raise RuntimeError(f"acl.mdl.create_desc({self.role}) returned no descriptor")
            check_acl_ret(
                self._acl.mdl.get_desc(self.model_desc, self.model_id),
                f"acl.mdl.get_desc({self.role})",
            )
            self.input_descriptors = self._describe("input")
            self.output_descriptors = self._describe("output")
            self.input_descriptors = self._resolve_zero_sizes("input", self.bindings.inputs, self.input_descriptors)
            self.output_descriptors = self._resolve_zero_sizes("output", self.bindings.outputs, self.output_descriptors)
            self._validate_bindings()
        except Exception:
            self.close()
            raise

    def prepare_datasets(
        self,
        *,
        input_overrides: dict[int, AclDeviceBuffer] | None = None,
        output_overrides: dict[int, AclDeviceBuffer] | None = None,
    ) -> None:
        try:
            self.input_dataset, self.input_buffers = self._create_dataset(self.input_descriptors, input_overrides or {})
            self.output_dataset, self.output_buffers = self._create_dataset(
                self.output_descriptors, output_overrides or {}
            )
            self.output_host_buffers = [None] * len(self.output_descriptors)
        except Exception:
            self.close()
            raise

    def prepare_dataset_banks(
        self,
        input_overrides: tuple[dict[int, AclDeviceBuffer], ...],
        output_overrides: tuple[dict[int, AclDeviceBuffer], ...],
        *,
        host_output_indices: set[int],
    ) -> None:
        """一次创建多套静态dataset，供stateful模型轮换Device状态。"""
        if not self.input_descriptors or not self.output_descriptors:
            raise ValueError("prepare_dataset_banks 前必须先调用 load_descriptor() 完成 IO 描述")
        if len(input_overrides) != len(output_overrides) or not input_overrides:
            raise ValueError("dataset bank的输入/输出override数量必须一致且非空")
        try:
            self._host_output_indices = set(host_output_indices)
            for input_bank, output_bank in zip(input_overrides, output_overrides, strict=True):
                input_dataset, input_buffers = self._create_dataset(self.input_descriptors, input_bank)
                output_dataset, output_buffers = self._create_dataset(self.output_descriptors, output_bank)
                host_buffers: list[object | None] = []
                for descriptor in self.output_descriptors:
                    if descriptor.index in self._host_output_indices:
                        host_buffer, ret = self._acl.rt.malloc_host(descriptor.size)
                        check_acl_ret(ret, f"acl.rt.malloc_host({self.role} output {descriptor.index})")
                        host_buffers.append(host_buffer)
                    else:
                        host_buffers.append(None)
                self._dataset_banks.append((input_dataset, input_buffers, output_dataset, output_buffers))
                self._dataset_bank_host_buffers.append(host_buffers)
        except Exception:
            self._close_dataset_banks()
            raise

    def execute_bank(self, bank: int, inputs: BoundInputs | dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """执行指定静态dataset；非owned输入（包括hidden/cell）绝不H2D。"""
        if self._execution_lock is None:
            return self._execute_bank_unlocked(bank, inputs)
        with self._execution_lock:
            return self._execute_bank_unlocked(bank, inputs)

    def _execute_bank_unlocked(self, bank: int, inputs: BoundInputs | dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        if not self._dataset_banks:
            raise BackendInferenceError(f"Ascend role {self.role!r}没有dataset bank", code="runtime_not_loaded")
        try:
            input_dataset, input_buffers, output_dataset, output_buffers = self._dataset_banks[bank]
            host_buffers = self._dataset_bank_host_buffers[bank]
        except IndexError as exc:
            raise BackendInferenceError(f"Ascend role {self.role!r} bank索引无效", code="invalid_bank") from exc
        self._lease.bind_current_thread()
        values = self._indexed_inputs(inputs)
        for descriptor, buffer in zip(self.input_descriptors, input_buffers, strict=True):
            if not buffer.owned:
                continue
            if descriptor.index not in values:
                raise BackendInferenceError(f"缺少{self.role}输入{descriptor.index}", code="missing_runtime_input")
            payload = np.ascontiguousarray(values[descriptor.index]).tobytes()
            if len(payload) != buffer.size:
                raise BackendInferenceError(
                    f"{self.role}输入{descriptor.index}字节数不匹配", code="input_size_mismatch"
                )
            source = self._acl.util.bytes_to_ptr(payload)
            check_acl_ret(
                self._acl.rt.memcpy(buffer.pointer, buffer.size, source, len(payload), ACL_MEMCPY_HOST_TO_DEVICE),
                f"acl.mdl.execute H2D({self.role})",
            )
        check_acl_ret(
            self._acl.mdl.execute(self.model_id, input_dataset, output_dataset), f"acl.mdl.execute({self.role})"
        )
        result: dict[int, np.ndarray] = {}
        for descriptor, buffer, host_buffer in zip(self.output_descriptors, output_buffers, host_buffers, strict=True):
            if descriptor.index not in self._host_output_indices:
                continue
            if host_buffer is None:
                raise BackendInferenceError(
                    f"{self.role}输出{descriptor.index}没有Host staging", code="missing_host_staging"
                )
            check_acl_ret(
                self._acl.rt.memcpy(host_buffer, buffer.size, buffer.pointer, buffer.size, ACL_MEMCPY_DEVICE_TO_HOST),
                f"acl.mdl.execute D2H({self.role})",
            )
            dtype = descriptor.dtype or np.dtype("float32")
            value = np.frombuffer(self._acl.util.ptr_to_bytes(host_buffer, buffer.size), dtype=dtype).copy()
            if descriptor.shape is not None and all(d > 0 for d in descriptor.shape):
                value = value.reshape(descriptor.shape)
            result[descriptor.index] = value
        return result

    def _close_dataset_banks(self) -> None:
        for host_buffers in reversed(self._dataset_bank_host_buffers):
            for host_buffer in reversed(host_buffers):
                if host_buffer is not None:
                    self._acl.rt.free_host(host_buffer)
        self._dataset_bank_host_buffers.clear()
        for input_dataset, input_buffers, output_dataset, output_buffers in reversed(self._dataset_banks):
            self._destroy_dataset(output_dataset, output_buffers)
            self._destroy_dataset(input_dataset, input_buffers)
        self._dataset_banks.clear()

    def execute(
        self,
        inputs: BoundInputs | dict[int, np.ndarray],
        *,
        read_outputs: set[int] | None = None,
        stream: object | None = None,
    ) -> dict[int, np.ndarray]:
        """Execute one request while exclusively owning this model's buffers."""

        if self._execution_lock is None:
            return self._execute_unlocked(inputs, read_outputs=read_outputs, stream=stream)
        with self._execution_lock:
            return self._execute_unlocked(inputs, read_outputs=read_outputs, stream=stream)

    def _execute_unlocked(
        self,
        inputs: BoundInputs | dict[int, np.ndarray],
        *,
        read_outputs: set[int] | None = None,
        stream: object | None = None,
    ) -> dict[int, np.ndarray]:
        if self._execution_lock is not None and (
            self._closing or self._closed or self._quarantined_shared_stream is not None or self._quarantined_datasets
        ):
            raise BackendInferenceError(
                "Ascend model is closing or quarantined",
                code="runtime_not_ready",
                operation_started=False,
                outcome_known=True,
            )
        if self.input_dataset is None or self.output_dataset is None or self.model_id is None:
            raise BackendInferenceError(f"Ascend role {self.role!r} is not fully loaded", code="runtime_not_loaded")
        self._lease.bind_current_thread()
        values = self._indexed_inputs(inputs)
        for descriptor, buffer in zip(self.input_descriptors, self.input_buffers, strict=True):
            if not buffer.owned:
                continue
            try:
                value = values[descriptor.index]
            except KeyError as exc:
                raise BackendInferenceError(
                    f"Ascend role {self.role!r} is missing runtime input index {descriptor.index}",
                    code="missing_runtime_input",
                ) from exc
            payload = np.ascontiguousarray(value).tobytes()
            if len(payload) != buffer.size:
                raise BackendInferenceError(
                    f"Ascend role {self.role!r} input {descriptor.index} has {len(payload)} bytes, "
                    f"runtime requires {buffer.size}",
                    code="input_size_mismatch",
                )
            source = self._acl.util.bytes_to_ptr(payload)
            check_acl_ret(
                self._acl.rt.memcpy(buffer.pointer, buffer.size, source, len(payload), ACL_MEMCPY_HOST_TO_DEVICE),
                f"acl.rt.memcpy H2D({self.role} input {descriptor.index})",
            )

        if stream is None:
            check_acl_ret(
                self._acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset),
                f"acl.mdl.execute({self.role})",
            )
        else:
            try:
                check_acl_ret(
                    self._acl.mdl.execute_async(self.model_id, self.input_dataset, self.output_dataset, stream),
                    f"acl.mdl.execute_async({self.role})",
                )
                check_acl_ret(
                    self._acl.rt.synchronize_stream(stream),
                    f"acl.rt.synchronize_stream({self.role})",
                )
            except Exception as exc:
                self._quarantined_shared_stream = stream
                raise BackendInferenceError(
                    str(exc),
                    code="async_execution_uncertain",
                    operation_started=True,
                    outcome_known=False,
                ) from exc
        selected = read_outputs if read_outputs is not None else set(range(len(self.output_descriptors)))
        outputs: dict[int, np.ndarray] = {}
        for descriptor, buffer, host_buffer in zip(
            self.output_descriptors, self.output_buffers, self.output_host_buffers, strict=True
        ):
            if descriptor.index not in selected:
                continue
            if host_buffer is None:
                host_buffer, ret = self._acl.rt.malloc_host(descriptor.size)
                check_acl_ret(ret, f"acl.rt.malloc_host({self.role} output {descriptor.index})")
                self.output_host_buffers[descriptor.index] = host_buffer
            check_acl_ret(
                self._acl.rt.memcpy(host_buffer, buffer.size, buffer.pointer, buffer.size, ACL_MEMCPY_DEVICE_TO_HOST),
                f"acl.rt.memcpy D2H({self.role} output {descriptor.index})",
            )
            payload = self._acl.util.ptr_to_bytes(host_buffer, buffer.size)
            dtype = descriptor.dtype or np.dtype("float32")
            value = np.frombuffer(payload, dtype=dtype).copy()
            if descriptor.shape is not None and all(dimension > 0 for dimension in descriptor.shape):
                value = value.reshape(descriptor.shape)
            outputs[descriptor.index] = value
        return outputs

    def _prepare_async_inputs(
        self,
        inputs: BoundInputs | dict[int, np.ndarray],
        buffers: list[_DatasetBuffer],
        *,
        operation_name: str,
    ) -> None:
        """Copy request inputs before ACL submission, with known outcomes."""

        try:
            values = self._indexed_inputs(inputs)
            for descriptor, buffer in zip(self.input_descriptors, buffers, strict=True):
                if not buffer.owned:
                    continue
                try:
                    payload = np.ascontiguousarray(values[descriptor.index]).tobytes()
                except KeyError as exc:
                    raise BackendInferenceError(
                        f"Ascend role {self.role!r} is missing runtime input index {descriptor.index}",
                        code="missing_runtime_input",
                        operation_started=False,
                        outcome_known=True,
                    ) from exc
                if len(payload) != buffer.size:
                    raise BackendInferenceError(
                        "input size mismatch",
                        code="input_size_mismatch",
                        operation_started=False,
                        outcome_known=True,
                    )
                source = self._acl.util.bytes_to_ptr(payload)
                check_acl_ret(
                    self._acl.rt.memcpy(
                        buffer.pointer,
                        buffer.size,
                        source,
                        len(payload),
                        ACL_MEMCPY_HOST_TO_DEVICE,
                    ),
                    f"acl.rt.memcpy {operation_name}({self.role} input {descriptor.index})",
                )
        except BackendInferenceError:
            raise
        except Exception as exc:
            raise BackendInferenceError(
                str(exc),
                code="async_input_preparation_failed",
                operation_started=False,
                outcome_known=True,
            ) from exc

    def submit_async_isolated(
        self,
        inputs: BoundInputs | dict[int, np.ndarray],
        *,
        stream: object,
        read_outputs: set[int] | None = None,
    ) -> AclAsyncExecution:
        """Submit with request-owned datasets so another stage model may overlap."""

        if self._execution_lock is None or self._completion_executor is None:
            raise BackendInferenceError("async execution is disabled", code="async_execution_not_enabled")
        with self._execution_lock:
            if (
                self._closed
                or self._closing
                or self._quarantined_datasets
                or self._quarantined_shared_stream is not None
            ):
                raise BackendInferenceError("model is closed or quarantined", code="runtime_unavailable")
            self._lease.bind_current_thread()
            input_dataset, input_buffers = self._create_dataset(self.input_descriptors, {})
            try:
                output_dataset, output_buffers = self._create_dataset(self.output_descriptors, {})
            except Exception:
                self._destroy_dataset(input_dataset, input_buffers)
                raise
            try:
                self._prepare_async_inputs(inputs, input_buffers, operation_name="isolated H2D")
            except Exception:
                self._destroy_dataset(output_dataset, output_buffers)
                self._destroy_dataset(input_dataset, input_buffers)
                raise
            try:
                check_acl_ret(
                    self._acl.mdl.execute_async(self.model_id, input_dataset, output_dataset, stream),
                    f"acl.mdl.execute_async({self.role})",
                )
            except Exception as exc:
                self._quarantined_datasets.append(
                    (stream, input_dataset, input_buffers, output_dataset, output_buffers)
                )
                raise BackendInferenceError(
                    str(exc), code="async_execution_uncertain", operation_started=True, outcome_known=False
                ) from exc

            def complete() -> dict[int, np.ndarray]:
                synchronized = False
                try:
                    self._lease.bind_current_thread()
                    check_acl_ret(self._acl.rt.synchronize_stream(stream), f"acl.rt.synchronize_stream({self.role})")
                    synchronized = True
                    return self._read_dataset_outputs(output_buffers, read_outputs)
                except Exception as exc:
                    raise BackendInferenceError(
                        str(exc),
                        code="async_output_failed" if synchronized else "async_execution_uncertain",
                        operation_started=True,
                        outcome_known=synchronized,
                    ) from exc
                finally:
                    if synchronized:
                        self._destroy_dataset(output_dataset, output_buffers)
                        self._destroy_dataset(input_dataset, input_buffers)
                    else:
                        with self._execution_lock:
                            self._quarantined_datasets.append(
                                (stream, input_dataset, input_buffers, output_dataset, output_buffers)
                            )

            try:
                completion = self._completion_executor.submit(complete)
            except Exception as exc:
                self._quarantined_datasets.append(
                    (stream, input_dataset, input_buffers, output_dataset, output_buffers)
                )
                raise BackendInferenceError(
                    str(exc), code="async_execution_uncertain", operation_started=True, outcome_known=False
                ) from exc
        return AclAsyncExecution(stream, completion)

    def _read_dataset_outputs(
        self, output_buffers: list[_DatasetBuffer], read_outputs: set[int] | None
    ) -> dict[int, np.ndarray]:
        selected = read_outputs if read_outputs is not None else set(range(len(self.output_descriptors)))
        outputs: dict[int, np.ndarray] = {}
        for descriptor, buffer in zip(self.output_descriptors, output_buffers, strict=True):
            if descriptor.index not in selected:
                continue
            host_buffer, ret = self._acl.rt.malloc_host(descriptor.size)
            check_acl_ret(ret, f"acl.rt.malloc_host({self.role} isolated output {descriptor.index})")
            try:
                check_acl_ret(
                    self._acl.rt.memcpy(
                        host_buffer, buffer.size, buffer.pointer, buffer.size, ACL_MEMCPY_DEVICE_TO_HOST
                    ),
                    f"acl.rt.memcpy isolated D2H({self.role} output {descriptor.index})",
                )
                dtype = descriptor.dtype or np.dtype("float32")
                value = np.frombuffer(self._acl.util.ptr_to_bytes(host_buffer, buffer.size), dtype=dtype).copy()
                if descriptor.shape is not None and all(dimension > 0 for dimension in descriptor.shape):
                    value = value.reshape(descriptor.shape)
                outputs[descriptor.index] = value
            finally:
                self._acl.rt.free_host(host_buffer)
        return outputs

    def allocate_device_buffer(self, size: int) -> AclDeviceBuffer:
        """申请由模型持有的常驻Device buffer，供dataset bank复用。"""
        pointer, ret = self._acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
        check_acl_ret(ret, f"acl.rt.malloc({self.role} shared state)")
        buffer = _DatasetBuffer(pointer, None, size, True)
        self._owned_device_buffers.append(buffer)
        return AclDeviceBuffer(pointer=pointer, size=size)

    def zero_device_buffer(self, buffer: AclDeviceBuffer) -> None:
        """清零常驻状态；优先使用ACL memset，兼容测试runtime则一次H2D零值。"""
        memset = getattr(self._acl.rt, "memset", None)
        if callable(memset):
            check_acl_ret(memset(buffer.pointer, buffer.size, 0, buffer.size), f"acl.rt.memset({self.role})")
            return
        payload = bytes(buffer.size)
        source = self._acl.util.bytes_to_ptr(payload)
        check_acl_ret(
            self._acl.rt.memcpy(buffer.pointer, buffer.size, source, buffer.size, ACL_MEMCPY_HOST_TO_DEVICE),
            f"acl.rt.memcpy zero H2D({self.role})",
        )

    def output_buffer(self, index: int) -> AclDeviceBuffer:
        try:
            buffer = self.output_buffers[index]
        except IndexError as exc:
            raise BackendLoadError(
                f"Ascend role {self.role!r} has no output buffer at index {index}",
                code="invalid_device_link",
            ) from exc
        return AclDeviceBuffer(pointer=buffer.pointer, size=buffer.size)

    def drain_quarantined(self) -> None:
        """排空隔离中的设备资源：同步隔离流并销毁隔离数据集。

        reset 复用 close 的排空语义在不卸载模型的前提下恢复确定性：
        失败时保留隔离资源以便重试（与 close 一致，失败不是完成栅栏）。
        """
        if self._quarantined_shared_stream is not None:
            self._lease.bind_current_thread()
            check_acl_ret(
                self._acl.rt.synchronize_stream(self._quarantined_shared_stream),
                f"acl.rt.synchronize_stream({self.role} shared drain)",
            )
            self._quarantined_shared_stream = None
        if self._quarantined_datasets:
            self._lease.bind_current_thread()
            for stream, *_datasets in self._quarantined_datasets:
                check_acl_ret(self._acl.rt.synchronize_stream(stream), f"acl.rt.synchronize_stream({self.role} drain)")
            for _stream, inputs, input_buffers, outputs, output_buffers in self._quarantined_datasets:
                self._destroy_dataset(outputs, output_buffers)
                self._destroy_dataset(inputs, input_buffers)
            self._quarantined_datasets.clear()

    def close(self) -> None:
        if self._closed:
            return
        if self._completion_executor is not None:
            with self._execution_lock:
                self._closing = True
            self._completion_executor.shutdown(wait=True, cancel_futures=False)
        # An unsuccessful synchronization is not a device completion fence.
        # Retain the model, context and datasets so close can be retried.
        self.drain_quarantined()
        self._closed = True
        if self._execution_lock is None:
            acl = self._acl
            self._lease.bind_current_thread()
            self._close_dataset_banks()
        else:
            with self._execution_lock:
                acl = self._acl
                self._lease.bind_current_thread()
                self._close_dataset_banks()
        for host_buffer in reversed(self.output_host_buffers):
            if host_buffer is not None:
                acl.rt.free_host(host_buffer)
        self.output_host_buffers.clear()
        self._destroy_dataset(self.output_dataset, self.output_buffers)
        self._destroy_dataset(self.input_dataset, self.input_buffers)
        self.output_dataset = None
        self.input_dataset = None
        for buffer in reversed(self._owned_device_buffers):
            acl.rt.free(buffer.pointer)
        self._owned_device_buffers.clear()
        if self.model_desc is not None:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
            self.model_id = None

    def _describe(self, direction: str) -> tuple[AclTensorDescriptor, ...]:
        mdl = self._acl.mdl
        if direction == "input":
            count = int(mdl.get_num_inputs(self.model_desc))
            size_fn = mdl.get_input_size_by_index
        else:
            count = int(mdl.get_num_outputs(self.model_desc))
            size_fn = mdl.get_output_size_by_index
        return tuple(
            AclTensorDescriptor(
                index=index,
                name=self._optional_name(direction, index),
                dtype=self._optional_dtype(direction, index),
                shape=self._optional_shape(direction, index),
                size=int(size_fn(self.model_desc, index)),
            )
            for index in range(count)
        )

    def _optional_name(self, direction: str, index: int) -> str | None:
        callback = getattr(self._acl.mdl, f"get_{direction}_name_by_index", None)
        if not callable(callback):
            return None
        value = _result_value(callback(self.model_desc, index), f"ACL {direction} name query")
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value) if value is not None else None

    def _optional_dtype(self, direction: str, index: int) -> np.dtype | None:
        callback = getattr(self._acl.mdl, f"get_{direction}_data_type", None)
        if not callable(callback):
            return None
        value = int(_result_value(callback(self.model_desc, index), f"ACL {direction} dtype query"))
        try:
            return _ACL_DTYPES[value]
        except KeyError as exc:
            raise BackendLoadError(
                f"Ascend role {self.role!r} {direction} {index} exposes unsupported ACL dtype code {value}",
                code="unsupported_runtime_dtype",
            ) from exc

    def _optional_shape(self, direction: str, index: int) -> tuple[int, ...] | None:
        callback = getattr(self._acl.mdl, f"get_{direction}_dims", None)
        if not callable(callback):
            return None
        value = _result_value(callback(self.model_desc, index), f"ACL {direction} shape query")
        if isinstance(value, dict):
            value = value.get("dims")
        if not isinstance(value, list | tuple):
            return None
        return tuple(int(dimension) for dimension in value)

    def _validate_bindings(self) -> None:
        self._validate_direction("input", self.bindings.inputs, self.input_descriptors)
        self._validate_direction("output", self.bindings.outputs, self.output_descriptors)

    def _validate_direction(
        self,
        direction: str,
        bindings: tuple[TensorBinding, ...],
        descriptors: tuple[AclTensorDescriptor, ...],
    ) -> None:
        indexed = _binding_by_index(bindings, direction)
        if len(indexed) != len(descriptors):
            raise BackendLoadError(
                f"Ascend role {self.role!r} declares {len(indexed)} {direction} bindings but runtime exposes "
                f"{len(descriptors)}",
                code=f"{direction}_binding_count_mismatch",
            )
        for descriptor in descriptors:
            try:
                binding = indexed[descriptor.index]
            except KeyError as exc:
                raise BackendLoadError(
                    f"Ascend role {self.role!r} has no manifest {direction} binding for runtime index "
                    f"{descriptor.index}",
                    code=f"missing_{direction}_binding",
                ) from exc
            if (
                descriptor.name is not None
                and binding.runtime_name is not None
                and descriptor.name != binding.runtime_name
            ):
                raise BackendLoadError(
                    f"Ascend role {self.role!r} {direction} {descriptor.index} runtime name "
                    f"{descriptor.name!r} does not match manifest name {binding.runtime_name!r}",
                    code="runtime_name_mismatch",
                )
            binding_dtype = numpy_dtype(binding.dtype)
            if descriptor.dtype is not None and descriptor.dtype != binding_dtype:
                raise BackendLoadError(
                    f"Ascend role {self.role!r} {direction} {descriptor.index} runtime dtype "
                    f"{descriptor.dtype.name!r} does not match manifest dtype {binding.dtype!r}",
                    code="runtime_dtype_mismatch",
                )
            if descriptor.shape is not None and not self._shapes_compatible(binding.shape, descriptor.shape):
                raise BackendLoadError(
                    f"Ascend role {self.role!r} {direction} {descriptor.index} runtime shape "
                    f"{descriptor.shape} does not match manifest shape {binding.shape}",
                    code="runtime_shape_mismatch",
                )
            if all(dimension > 0 for dimension in binding.shape):
                expected_size = int(np.prod(binding.shape, dtype=np.int64)) * binding_dtype.itemsize
                if expected_size != descriptor.size:
                    raise BackendLoadError(
                        f"Ascend role {self.role!r} {direction} {descriptor.index} runtime buffer size "
                        f"{descriptor.size} does not match manifest size {expected_size}",
                        code="runtime_size_mismatch",
                    )

    def _resolve_zero_sizes(
        self,
        direction: str,
        bindings: tuple[TensorBinding, ...],
        descriptors: tuple[AclTensorDescriptor, ...],
    ) -> tuple[AclTensorDescriptor, ...]:
        """Use a fixed manifest ABI when ACL reports zero for symbolic outputs."""

        indexed = _binding_by_index(bindings, direction)
        resolved = []
        for descriptor in descriptors:
            if descriptor.size > 0:
                resolved.append(descriptor)
                continue
            binding = indexed.get(descriptor.index)
            if binding is None or any(dimension <= 0 for dimension in binding.shape):
                raise BackendLoadError(
                    f"Ascend role {self.role!r} {direction} {descriptor.index} reports zero buffer size; "
                    "the manifest must declare a fully fixed shape",
                    code="runtime_size_unresolved",
                )
            size = int(np.prod(binding.shape, dtype=np.int64)) * numpy_dtype(binding.dtype).itemsize
            resolved.append(
                AclTensorDescriptor(
                    index=descriptor.index,
                    name=descriptor.name,
                    dtype=descriptor.dtype,
                    shape=descriptor.shape,
                    size=size,
                )
            )
        return tuple(resolved)

    def _create_dataset(
        self,
        descriptors: tuple[AclTensorDescriptor, ...],
        overrides: dict[int, AclDeviceBuffer],
    ) -> tuple[object, list[_DatasetBuffer]]:
        dataset = self._acl.mdl.create_dataset()
        if dataset is None:
            raise RuntimeError(f"acl.mdl.create_dataset({self.role}) returned no dataset")
        buffers: list[_DatasetBuffer] = []
        try:
            for descriptor in descriptors:
                override = overrides.get(descriptor.index)
                if override is not None:
                    if override.size != descriptor.size:
                        raise BackendLoadError(
                            f"Ascend device link for role {self.role!r} index {descriptor.index} has size "
                            f"{override.size}, expected {descriptor.size}",
                            code="device_link_size_mismatch",
                        )
                    pointer = override.pointer
                    owned = False
                else:
                    pointer, ret = self._acl.rt.malloc(descriptor.size, ACL_MEM_MALLOC_HUGE_FIRST)
                    check_acl_ret(ret, f"acl.rt.malloc({self.role} index {descriptor.index})")
                    owned = True
                data_buffer = self._acl.create_data_buffer(pointer, descriptor.size)
                if data_buffer is None:
                    if owned:
                        self._acl.rt.free(pointer)
                    raise RuntimeError(f"acl.create_data_buffer({self.role} index {descriptor.index}) failed")
                result = self._acl.mdl.add_dataset_buffer(dataset, data_buffer)
                if isinstance(result, tuple):
                    check_acl_ret(result[-1], f"acl.mdl.add_dataset_buffer({self.role})")
                else:
                    check_acl_ret(result, f"acl.mdl.add_dataset_buffer({self.role})")
                buffers.append(_DatasetBuffer(pointer, data_buffer, descriptor.size, owned))
        except Exception:
            self._destroy_dataset(dataset, buffers)
            raise
        return dataset, buffers

    def _destroy_dataset(self, dataset: object | None, buffers: list[_DatasetBuffer]) -> None:
        for buffer in reversed(buffers):
            self._acl.destroy_data_buffer(buffer.data_buffer)
            if buffer.owned:
                self._acl.rt.free(buffer.pointer)
        buffers.clear()
        if dataset is not None:
            self._acl.mdl.destroy_dataset(dataset)

    @staticmethod
    def _indexed_inputs(inputs: object | dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        # BoundInputs 在运行时因避免循环导入退化为 Any（见文件头 TYPE_CHECKING
        # 块），这里在入口守住契约边界——dict 走 legacy 索引化，否则要求对象有
        # .tensors 属性（BoundInputs 协议）；传错在此早报错而非深入 ACL
        # memcpy 时才抛 AttributeError。
        if isinstance(inputs, dict):
            return inputs
        tensors = getattr(inputs, "tensors", None)
        if tensors is None:
            raise TypeError(
                f"inputs 必须是 dict[int, np.ndarray] 或 BoundInputs（带 .tensors 属性），得到 {type(inputs).__name__}"
            )
        return {int(tensor.index): tensor.value for tensor in tensors if tensor.index is not None}

    @staticmethod
    def _shapes_compatible(manifest_shape: tuple[int, ...], runtime_shape: tuple[int, ...]) -> bool:
        return len(manifest_shape) == len(runtime_shape) and all(
            declared == -1 or actual == -1 or declared == actual
            for declared, actual in zip(manifest_shape, runtime_shape, strict=True)
        )

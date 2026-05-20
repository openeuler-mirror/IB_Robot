"""SD3403 ACT worker wrapper (persistent worker + binary protocol)."""

import contextlib
import os
import re
import struct
import subprocess
import threading
import time
from collections import deque
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

PROTOCOL_MAGIC = 0x53565031
PROTOCOL_VERSION = 1
WORKER_STATUS_OK = 0

WORKER_ELEM_FLOAT32 = 1
WORKER_ELEM_FLOAT16 = 2
WORKER_ELEM_INT8 = 3
WORKER_ELEM_UINT8 = 4
WORKER_ELEM_INT32 = 5
WORKER_ELEM_INT64 = 6

REQUEST_HEADER_STRUCT = struct.Struct("<IHHII")
INPUT_ENTRY_STRUCT = struct.Struct("<III")
RESPONSE_HEADER_STRUCT = struct.Struct("<IHHIIIiI")
OUTPUT_ENTRY_STRUCT = struct.Struct("<IIIIII")
DIM_STRUCT = struct.Struct("<Q")
DEFAULT_OM_BASENAME = "act_distill_fp32_for_mindcmd_simp_release.om"
MODEL_LOAD_MS_RE = re.compile(r"model_load_ms=([0-9]+(?:\.[0-9]+)?)")


def _is_om_path(path: str) -> bool:
    return path.lower().endswith(".om")


def _guess_worker_path_from_model(model_path: str) -> str:
    model_dir = os.path.dirname(model_path)
    if os.path.basename(model_dir) == "model":
        return os.path.normpath(os.path.join(model_dir, "../out/main"))
    return os.path.normpath(os.path.join(model_dir, "main"))


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("worker stream closed unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _dtype_from_elem_type(elem_type: int):
    if elem_type == WORKER_ELEM_FLOAT32:
        return np.float32
    if elem_type == WORKER_ELEM_FLOAT16:
        return np.float16
    if elem_type == WORKER_ELEM_INT8:
        return np.int8
    if elem_type == WORKER_ELEM_UINT8:
        return np.uint8
    if elem_type == WORKER_ELEM_INT32:
        return np.int32
    if elem_type == WORKER_ELEM_INT64:
        return np.int64
    raise RuntimeError(f"unsupported worker element type: {elem_type}")


def _to_numpy_float32(t: Tensor) -> np.ndarray:
    # Fast path: no copy when already CPU float32 contiguous tensor.
    if t.device.type == "cpu" and t.dtype == torch.float32 and t.is_contiguous():
        return t.detach().numpy()
    return np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32, copy=False))


class ACT3403Policy:
    def __init__(self, cpp_executable: str, model_path: str | None = None):
        super().__init__()
        self.cpp_executable, resolved_model_path = self._resolve_paths(cpp_executable, model_path)
        self.cpp_dir = os.path.dirname(self.cpp_executable)

        self._worker_env = os.environ.copy()
        self._worker_env["SVP_MODEL_PATH"] = resolved_model_path
        self._image_height = int(os.environ.get("SVP_IMAGE_HEIGHT", "240"))
        self._image_width = int(os.environ.get("SVP_IMAGE_WIDTH", "320"))
        self._resize_warned = False
        self._perf_enabled = os.environ.get("SVP_PERF_LOG", "1") != "0"
        self._perf_log_every = max(1, int(os.environ.get("SVP_PERF_LOG_EVERY", "1")))
        self._predict_count = 0
        self._process_start_ts = 0.0
        self._model_load_ms: float | None = None
        self._model_load_logged = False
        self._model_load_reported_unavailable = False

        self._request_id = 0
        self._process: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._io_lock = threading.Lock()
        self._stderr_tail = deque(maxlen=80)
        self._start_process()

    def _resolve_paths(self, cpp_executable: str, model_path: str | None) -> tuple[str, str]:
        env_worker = (
            os.environ.get("SVP_WORKER_EXECUTABLE", "").strip() or os.environ.get("SVP_CPP_EXECUTABLE", "").strip()
        )
        env_model = os.environ.get("SVP_MODEL_PATH", "").strip()
        arg = cpp_executable.strip()

        resolved_worker = ""
        resolved_model = ""
        if model_path:
            resolved_worker = arg
            resolved_model = model_path
        elif _is_om_path(arg):
            resolved_model = arg
            resolved_worker = env_worker or _guess_worker_path_from_model(resolved_model)
        else:
            resolved_worker = arg or env_worker
            if env_model:
                resolved_model = env_model
                if not resolved_worker:
                    resolved_worker = _guess_worker_path_from_model(env_model)
            elif resolved_worker:
                resolved_model = os.path.normpath(
                    os.path.join(os.path.dirname(resolved_worker), f"../model/{DEFAULT_OM_BASENAME}")
                )

        if not resolved_worker:
            raise RuntimeError(
                "missing worker executable path: pass binary path to ACT3403Policy(...) or set SVP_WORKER_EXECUTABLE"
            )
        if not resolved_model:
            raise RuntimeError("missing model path: pass model_path to ACT3403Policy(...) or set SVP_MODEL_PATH")

        resolved_worker = os.path.abspath(resolved_worker)
        resolved_model = os.path.abspath(resolved_model)

        if not os.path.isfile(resolved_worker):
            raise FileNotFoundError(f"worker executable not found: {resolved_worker}")
        if not os.access(resolved_worker, os.X_OK):
            raise PermissionError(f"worker executable is not executable: {resolved_worker}")
        if not os.path.isfile(resolved_model):
            raise FileNotFoundError(f"OM model file not found: {resolved_model}")
        return resolved_worker, resolved_model

    def _start_process(self):
        self._process_start_ts = time.perf_counter()
        self._model_load_ms = None
        self._model_load_logged = False
        self._model_load_reported_unavailable = False
        self._process = subprocess.Popen(
            [self.cpp_executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cpp_dir,
            env=self._worker_env,
            text=False,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self, pipe):
        with contextlib.suppress(Exception):
            while True:
                line = pipe.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    self._stderr_tail.append(decoded)
                    match = MODEL_LOAD_MS_RE.search(decoded)
                    if match is not None:
                        with contextlib.suppress(ValueError):
                            self._model_load_ms = float(match.group(1))

    def close(self):
        if self._process is None:
            return
        with contextlib.suppress(Exception):
            if self._process.stdin:
                self._process.stdin.close()
        with contextlib.suppress(Exception):
            self._process.terminate()
            self._process.wait(timeout=1.0)
        if self._process.poll() is None:
            with contextlib.suppress(Exception):
                self._process.kill()
        thread = self._stderr_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)
        self._stderr_thread = None
        self._process = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _worker_exit_message(self) -> str:
        if self._process is None:
            return "worker process is not running"
        return_code = self._process.poll()
        msg = "worker exited unexpectedly"
        if return_code is not None:
            msg += f" (returncode={return_code})"
        if self._stderr_tail:
            msg += "\nworker stderr tail:\n" + "\n".join(self._stderr_tail)
        return msg

    def _ensure_process(self):
        if self._process is None:
            self._start_process()
            return
        if self._process.poll() is not None:
            self.close()
            self._start_process()

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _normalize_image_tensor(self, image: Tensor, name: str) -> Tensor:
        if image.ndim != 4:
            raise RuntimeError(f"{name} must be NCHW tensor, got shape={tuple(image.shape)}")

        if image.dtype != torch.float32:
            image = image.to(dtype=torch.float32)

        height, width = int(image.shape[-2]), int(image.shape[-1])
        if height != self._image_height or width != self._image_width:
            image = functional.interpolate(
                image,
                size=(self._image_height, self._image_width),
                mode="bilinear",
                align_corners=False,
            )
            if not self._resize_warned:
                print(
                    "[ACT3403Policy] resize image inputs to "
                    f"{self._image_height}x{self._image_width} before worker inference"
                )
                self._resize_warned = True

        return image.contiguous()

    def _build_inputs(self, batch: dict[str, Tensor]) -> list[np.ndarray]:
        # Prefer explicit keys expected by exported ACT model.
        state = batch.get("observation.state")
        top = batch.get("observation.images.top")
        wrist = batch.get("observation.images.wrist")

        if isinstance(state, Tensor) and isinstance(top, Tensor) and isinstance(wrist, Tensor):
            top = self._normalize_image_tensor(top, "observation.images.top")
            wrist = self._normalize_image_tensor(wrist, "observation.images.wrist")
            return [_to_numpy_float32(state), _to_numpy_float32(top), _to_numpy_float32(wrist)]

        # Fallback 1: merged image tensor in observation.images (camera order assumed top,wrist).
        merged_images = batch.get("observation.images")
        if isinstance(state, Tensor) and isinstance(merged_images, Tensor) and merged_images.ndim >= 2:
            if merged_images.shape[1] < 2:
                raise RuntimeError("observation.images must contain at least 2 cameras for SD3403")
            top_tensor = self._normalize_image_tensor(merged_images[:, 0, ...], "observation.images[0]")
            wrist_tensor = self._normalize_image_tensor(merged_images[:, 1, ...], "observation.images[1]")
            top_arr = _to_numpy_float32(top_tensor)
            wrist_arr = _to_numpy_float32(wrist_tensor)
            return [_to_numpy_float32(state), top_arr, wrist_arr]

        # Fallback 2: observation.images can be a list/tuple of image tensors in ACT pipeline.
        if (
            isinstance(state, Tensor)
            and isinstance(merged_images, list | tuple)
            and len(merged_images) >= 2
            and isinstance(merged_images[0], Tensor)
            and isinstance(merged_images[1], Tensor)
        ):
            top_tensor = self._normalize_image_tensor(merged_images[0], "observation.images[0]")
            wrist_tensor = self._normalize_image_tensor(merged_images[1], "observation.images[1]")
            return [
                _to_numpy_float32(state),
                _to_numpy_float32(top_tensor),
                _to_numpy_float32(wrist_tensor),
            ]

        raise RuntimeError(
            "missing required inputs: need observation.state + "
            "(observation.images.top & observation.images.wrist or observation.images)"
        )

    def _write_request(self, input_arrays: Sequence[np.ndarray]) -> int:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("worker process is not running")
        if self._process.poll() is not None:
            raise RuntimeError(self._worker_exit_message())

        request_id = self._next_request_id()
        header = REQUEST_HEADER_STRUCT.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            len(input_arrays),
            request_id,
            0,
        )
        try:
            self._process.stdin.write(header)
            for idx, arr in enumerate(input_arrays):
                contiguous = arr
                if contiguous.dtype != np.float32 or not contiguous.flags["C_CONTIGUOUS"]:
                    contiguous = np.ascontiguousarray(contiguous, dtype=np.float32)
                payload_size = int(contiguous.nbytes)
                self._process.stdin.write(INPUT_ENTRY_STRUCT.pack(idx, payload_size, 0))
                # Write directly from numpy memory to avoid an extra tobytes() copy.
                self._process.stdin.write(memoryview(contiguous).cast("B"))
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(self._worker_exit_message()) from exc
        return request_id

    def _read_response(self, expected_request_id: int) -> tuple[np.ndarray, int]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("worker process is not running")
        if self._process.poll() is not None:
            raise RuntimeError(self._worker_exit_message())

        header_bytes = _read_exact(self._process.stdout, RESPONSE_HEADER_STRUCT.size)
        magic, version, status, request_id, output_count, latency_us, error_code, error_msg_size = (
            RESPONSE_HEADER_STRUCT.unpack(header_bytes)
        )
        if magic != PROTOCOL_MAGIC or version != PROTOCOL_VERSION:
            raise RuntimeError(f"unexpected response header: magic=0x{magic:x}, version={version}")
        if request_id != expected_request_id:
            raise RuntimeError(f"mismatched response id: expected {expected_request_id}, got {request_id}")

        target_output: np.ndarray | None = None
        for _ in range(output_count):
            entry = _read_exact(self._process.stdout, OUTPUT_ENTRY_STRUCT.size)
            output_index, elem_type, elem_count, byte_size, dim_count, _reserved = OUTPUT_ENTRY_STRUCT.unpack(entry)
            dims = [DIM_STRUCT.unpack(_read_exact(self._process.stdout, DIM_STRUCT.size))[0] for _ in range(dim_count)]
            payload = _read_exact(self._process.stdout, byte_size)
            if output_index == 2:
                data = np.frombuffer(payload, dtype=_dtype_from_elem_type(elem_type), count=elem_count)
                if dims:
                    data = data.reshape(tuple(int(d) for d in dims))
                target_output = data

        error_msg = ""
        if error_msg_size:
            error_msg = _read_exact(self._process.stdout, error_msg_size).decode("utf-8", errors="replace")
        if status != WORKER_STATUS_OK:
            raise RuntimeError(f"worker inference failed (error_code={error_code}): {error_msg or 'unknown error'}")
        if target_output is None:
            raise RuntimeError("worker response does not contain output index 2")
        return target_output, int(latency_us)

    def _execute_prepared_arrays(self, input_arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, int, int]:
        request_id = self._write_request(input_arrays)
        data, worker_latency_us = self._read_response(request_id)
        return data, worker_latency_us, request_id

    def execute_arrays(self, input_arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Run worker inference for already prepared arrays."""
        with self._io_lock:
            self._ensure_process()
            data, _worker_latency_us, _request_id = self._execute_prepared_arrays(input_arrays)
        return data

    def predict(self, batch: dict[str, Tensor]) -> tuple[Tensor] | None:
        t0 = time.perf_counter()
        with self._io_lock:
            self._ensure_process()
            t1 = time.perf_counter()
            input_arrays = self._build_inputs(batch)
            t2 = time.perf_counter()
            data, worker_latency_us, request_id = self._execute_prepared_arrays(input_arrays)
            t3 = time.perf_counter()
            t4 = time.perf_counter()

        flat = data.astype(np.float32, copy=False).reshape(-1)
        if flat.size % 8 != 0:
            raise RuntimeError(f"unexpected action tensor size={flat.size}, not divisible by 8")
        action = flat.reshape(-1, 8)[:, :6]
        action_tensor = torch.from_numpy(action.reshape(1, -1, 6))

        t5 = time.perf_counter()
        self._predict_count += 1
        if self._perf_enabled and (self._predict_count % self._perf_log_every == 0):
            e2e_ms = (t5 - t0) * 1000.0
            worker_infer_ms = worker_latency_us / 1000.0
            prepare_ms = (t2 - t1) * 1000.0
            write_ms = (t3 - t2) * 1000.0
            wait_resp_ms = (t4 - t3) * 1000.0
            post_ms = (t5 - t4) * 1000.0
            print(
                "[ACT3403Policy][PERF] "
                f"request_id={request_id} "
                f"e2e_ms={e2e_ms:.3f} "
                f"worker_infer_ms={worker_infer_ms:.3f} "
                f"prepare_ms={prepare_ms:.3f} "
                f"ipc_write_ms={write_ms:.3f} "
                f"wait_response_ms={wait_resp_ms:.3f} "
                f"post_ms={post_ms:.3f} "
                f"non_worker_ms={max(0.0, e2e_ms - worker_infer_ms):.3f}"
            )

        if self._perf_enabled and not self._model_load_logged:
            if self._model_load_ms is not None:
                print(f"[ACT3403Policy][PERF] model_load_ms={self._model_load_ms:.3f} (from worker)")
                self._model_load_logged = True
            elif self._predict_count == 1 and not self._model_load_reported_unavailable:
                print(
                    "[ACT3403Policy][PERF] precise model_load_ms not available from worker log yet; "
                    "please ensure SVP_NNN worker binary includes [PERF] model_load_ms logging"
                )
                self._model_load_reported_unavailable = True

        return (action_tensor,)

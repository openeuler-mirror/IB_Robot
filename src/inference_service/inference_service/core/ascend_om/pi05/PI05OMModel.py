"""
PI05OMModel.py

PI05 专用的融合 OM 模型管理类
实现 VLM 和 Action Expert 之间的 Device 端 buffer 复用，减少数据搬运开销
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
import torch
from torch import Tensor

try:
    import acl
except ImportError:  # pragma: no cover - only available on Ascend runtime hosts
    acl = None  # type: ignore[assignment]

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

# Debug flag: set ``PI05_OM_DEBUG=1`` to enable verbose per-step diagnostics.
_DEBUG = os.environ.get("PI05_OM_DEBUG", "").lower() in ("1", "true", "yes")

# Optional VLM output dump: set ``PI05_OM_DUMP_VLM=<dir>`` and ONLY the first
# forward() call will write each VLM output as ``<name>.npy`` under that
# directory.  Meant to pair with the companion CPU ONNX reference dumper.
_DUMP_VLM_DIR = os.environ.get("PI05_OM_DUMP_VLM", "").strip() or None

# Optional AE first-step dump: set ``PI05_OM_DUMP_AE=<dir>`` and ONLY the
# first forward() call's first denoise step (time=1.0) will save
# ``x_t_step0.npy`` under that directory.  Pairs with ``dump_ae_pt.py``
# for AE isolation testing.
_DUMP_AE_DIR = os.environ.get("PI05_OM_DUMP_AE", "").strip() or None


def _tensor_stats(arr: np.ndarray) -> str:
    """Return a compact string with shape, dtype and basic value stats."""
    flat = arr.reshape(-1)
    finite = np.isfinite(flat).all()
    if flat.size == 0:
        return f"shape={tuple(arr.shape)} dtype={arr.dtype} (empty)"
    a32 = flat.astype(np.float32, copy=False) if flat.dtype.kind == "f" else flat
    return (
        f"shape={tuple(arr.shape)} dtype={arr.dtype} "
        f"min={float(a32.min()):+.4g} max={float(a32.max()):+.4g} "
        f"mean={float(a32.mean()):+.4g} std={float(a32.astype(np.float32).std()):+.4g} "
        f"finite={bool(finite)}"
    )


# ACL data-type codes (aclDataType enum)
ACL_DT_FLOAT = 0  # float32
ACL_DT_FLOAT16 = 1  # float16

_ACL_DTYPE_TO_NP = {
    ACL_DT_FLOAT: np.float32,
    ACL_DT_FLOAT16: np.float16,
}


def logger(msg: str):
    print(f"[PI05OMModel]: {msg}")


class PI05OMModel:
    """
    PI05 专用的融合 OM 模型类.

    实现 VLM 和 Action Expert 之间的 buffer 共享:
    - VLM output buffer (kv_cache, pad_masks) 直接作为 Action Expert input buffer
    - 减少 D2H + H2D 数据搬运
    """

    def __init__(
        self,
        vlm_model_path: str,
        action_expert_model_path: str,
        config: Any,
    ):
        """
        初始化 PI05OMModel.

        Args:
            vlm_model_path: VLM OM 模型文件路径
            action_expert_model_path: Action Expert OM 模型文件路径
            config: duck-typed config exposing
                ``chunk_size``, ``max_action_dim``, ``num_inference_steps``.
        """
        if acl is None:
            raise RuntimeError(
                "Ascend ACL runtime is required for PI05 OM inference but the "
                "``acl`` module is not importable on this host."
            )

        self.config = config
        self.chunk_size = config.chunk_size
        self.max_action_dim = config.max_action_dim
        self.num_inference_steps = config.num_inference_steps
        self.device_id = 0

        # ACL 初始化
        ret = acl.init()
        self._check_ret(ret, "Failed to init ACL")

        ret = acl.rt.set_device(self.device_id)
        self._check_ret(ret, "Failed to set device")
        logger(f"Set device id {self.device_id}")

        self._context, ret = acl.rt.create_context(self.device_id)
        self._check_ret(ret, "Failed to create ACL context")
        self._tls = threading.local()
        self._tls.bound = True  # 当前 (主) 线程已被 create_context 绑上

        # 加载 VLM 模型
        logger(f"Loading VLM model from {vlm_model_path}")
        self.vlm_model_id, ret = acl.mdl.load_from_file(vlm_model_path)
        self._check_ret(ret, "Failed to load VLM model")

        self.vlm_model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.vlm_model_desc, self.vlm_model_id)
        self._check_ret(ret, "Failed to get VLM model desc")
        logger("VLM model loaded successfully")

        # 加载 Action Expert 模型
        logger(f"Loading Action Expert model from {action_expert_model_path}")
        self.ae_model_id, ret = acl.mdl.load_from_file(action_expert_model_path)
        self._check_ret(ret, "Failed to load Action Expert model")

        self.ae_model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.ae_model_desc, self.ae_model_id)
        self._check_ret(ret, "Failed to get Action Expert model desc")
        logger("Action Expert model loaded successfully")

        # 创建 datasets 和 buffers
        self._setup_datasets()

        # 验证 buffer 大小匹配
        self._validate_buffer_sizes()

        # 从 AE 模型描述符自动推断浮点精度 (fp16 / fp32)
        self._float_np_dtype = self._detect_float_dtype()

        # Stage C — Plan A: discover prefix sequence length from the
        # last VLM input slot (``prefix_att_2d_masks_4d`` of shape
        # ``(B, 1, S, S)``).
        self.prefix_seq_len = self._detect_prefix_seq_len()
        logger(f"Detected prefix_seq_len = {self.prefix_seq_len}")

        logger(f"PI05OMModel initialized (dtype={self._float_np_dtype.__name__}, buffer sharing enabled)")

    def _detect_prefix_seq_len(self) -> int:
        """Read S from the VLM input slot for ``prefix_att_2d_masks_4d``."""
        last_idx = len(self.vlm_input_data) - 1
        try:
            dims = acl.mdl.get_input_dims(self.vlm_model_desc, last_idx)[0]["dims"]
        except Exception as exc:
            raise RuntimeError(f"Failed to query VLM input[{last_idx}] dims for prefix_att_2d_masks_4d: {exc}") from exc
        if len(dims) != 4 or dims[-1] != dims[-2]:
            raise RuntimeError(
                f"Unexpected VLM input[{last_idx}] dims {dims}; expected (B, 1, S, S) for prefix_att_2d_masks_4d"
            )
        return int(dims[-1])

    def _setup_datasets(self):
        """创建所有 datasets 和 buffers，实现共享机制."""

        # === VLM Input Dataset ===
        self.vlm_input_dataset = acl.mdl.create_dataset()
        self.vlm_input_data = []
        vlm_input_num = acl.mdl.get_num_inputs(self.vlm_model_desc)

        for i in range(vlm_input_num):
            buffer_size = acl.mdl.get_input_size_by_index(self.vlm_model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            self._check_ret(ret, f"Failed to malloc VLM input buffer {i}")

            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.vlm_input_dataset, data_buffer)
            self._check_ret(ret, f"Failed to add VLM input buffer {i}")

            self.vlm_input_data.append(
                {
                    "buffer": buffer,
                    "data": data_buffer,
                    "size": buffer_size,
                    "owned": True,
                }
            )
        logger(f"VLM input dataset created with {vlm_input_num} buffers")

        # === VLM Output Dataset (将被 AE Input 共享) ===
        self.vlm_output_dataset = acl.mdl.create_dataset()
        self.vlm_output_data = []
        vlm_output_num = acl.mdl.get_num_outputs(self.vlm_model_desc)

        for i in range(vlm_output_num):
            buffer_size = acl.mdl.get_output_size_by_index(self.vlm_model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            self._check_ret(ret, f"Failed to malloc VLM output buffer {i}")

            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.vlm_output_dataset, data_buffer)
            self._check_ret(ret, f"Failed to add VLM output buffer {i}")

            self.vlm_output_data.append(
                {
                    "buffer": buffer,
                    "data": data_buffer,
                    "size": buffer_size,
                    "owned": True,
                }
            )
        logger(f"VLM output dataset created with {vlm_output_num} buffers")

        # === Action Expert Input Dataset (部分共享 VLM output) ===
        self.ae_input_dataset = acl.mdl.create_dataset()
        self.ae_input_data = []
        ae_input_num = acl.mdl.get_num_inputs(self.ae_model_desc)

        for i in range(ae_input_num):
            buffer_size = acl.mdl.get_input_size_by_index(self.ae_model_desc, i)

            if i < len(self.vlm_output_data):
                # 共享 VLM output buffer (kv_cache, pad_masks)
                shared_buffer = self.vlm_output_data[i]["buffer"]
                data_buffer = acl.create_data_buffer(shared_buffer, buffer_size)
                _, ret = acl.mdl.add_dataset_buffer(self.ae_input_dataset, data_buffer)
                self._check_ret(ret, f"Failed to add shared AE input buffer {i}")

                self.ae_input_data.append(
                    {
                        "buffer": shared_buffer,
                        "data": data_buffer,
                        "size": buffer_size,
                        "owned": False,
                    }
                )
            else:
                # 独立分配 (time, x_t)
                buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
                self._check_ret(ret, f"Failed to malloc AE input buffer {i}")

                data_buffer = acl.create_data_buffer(buffer, buffer_size)
                _, ret = acl.mdl.add_dataset_buffer(self.ae_input_dataset, data_buffer)
                self._check_ret(ret, f"Failed to add AE input buffer {i}")

                self.ae_input_data.append(
                    {
                        "buffer": buffer,
                        "data": data_buffer,
                        "size": buffer_size,
                        "owned": True,
                    }
                )

        shared_count = min(len(self.vlm_output_data), ae_input_num)
        independent_count = ae_input_num - shared_count
        logger(f"AE input dataset: {shared_count} shared + {independent_count} independent buffers")

        # === Action Expert Output Dataset ===
        self.ae_output_dataset = acl.mdl.create_dataset()
        self.ae_output_data = []
        ae_output_num = acl.mdl.get_num_outputs(self.ae_model_desc)

        for i in range(ae_output_num):
            buffer_size = acl.mdl.get_output_size_by_index(self.ae_model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            self._check_ret(ret, f"Failed to malloc AE output buffer {i}")

            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.ae_output_dataset, data_buffer)
            self._check_ret(ret, f"Failed to add AE output buffer {i}")

            self.ae_output_data.append(
                {
                    "buffer": buffer,
                    "data": data_buffer,
                    "size": buffer_size,
                    "owned": True,
                }
            )
        logger(f"AE output dataset created with {ae_output_num} buffers")

    def _detect_float_dtype(self) -> type:
        """从 AE 模型描述符自动推断浮点精度 (fp16 / fp32)."""
        shared_count = len(self.vlm_output_data)
        time_input_idx = shared_count  # 'time' 是第一个非共享输入
        ae_input_num = acl.mdl.get_num_inputs(self.ae_model_desc)
        if time_input_idx >= ae_input_num:
            logger("Warning: cannot locate 'time' input in AE model, defaulting to float16")
            return np.float16
        try:
            acl_dtype = acl.mdl.get_input_data_type(self.ae_model_desc, time_input_idx)
            np_dtype = _ACL_DTYPE_TO_NP.get(acl_dtype)
            if np_dtype is not None:
                logger(
                    f"Auto-detected AE float dtype from input[{time_input_idx}]: "
                    f"{np_dtype.__name__} (ACL code={acl_dtype})"
                )
                return np_dtype
            logger(
                f"Warning: unexpected ACL dtype code {acl_dtype} for AE input[{time_input_idx}], defaulting to float16"
            )
            return np.float16
        except Exception as exc:
            logger(f"Warning: failed to query AE input dtype ({exc}), defaulting to float16")
            return np.float16

    def _validate_buffer_sizes(self):
        """验证 VLM output 和 AE input 的共享 buffer 大小匹配."""
        for i in range(min(len(self.vlm_output_data), len(self.ae_input_data))):
            vlm_size = self.vlm_output_data[i]["size"]
            ae_size = acl.mdl.get_input_size_by_index(self.ae_model_desc, i)
            if vlm_size != ae_size:
                raise ValueError(f"Buffer size mismatch at index {i}: VLM output={vlm_size}, AE input={ae_size}")
        logger("Buffer size validation passed")

    def _ensure_context(self) -> None:
        """Bind this model's ACL context to the current thread on first use.

        ACL contexts live in thread-local storage on the host side. Any worker
        thread spawned *after* ``__init__`` (rclpy executor threads, asyncio
        thread pools, ...) starts out with a NULL context and will fail any
        ``acl.rt.memcpy`` / ``acl.mdl.execute`` call with error 107002
        (``ACL_ERROR_RT_CONTEXT_NULL_ERROR``).

        We pay the one-shot ``set_context`` per thread instead of doing it on
        every ``forward`` call so the semantics stay explicit and cheap.
        """
        if not getattr(self._tls, "bound", False):
            ret = acl.rt.set_context(self._context)
            self._check_ret(ret, "Failed to bind ACL context to current thread")
            self._tls.bound = True

    def forward(
        self,
        images: list[np.ndarray],
        tokens: np.ndarray,
        masks: np.ndarray,
        prefix_att_2d_masks_4d: np.ndarray,
        noise: np.ndarray | None = None,
    ) -> Tensor:
        """
        执行完整的 PI05 推理流程.

        Args:
            images: 所有摄像头图像列表, 每个 [B, C, H, W] float32.
                    顺序必须与 VLM ONNX/OM 的输入顺序一致.
            tokens: 语言 token IDs [B, seq_len], int64
            masks: 注意力 mask [B, seq_len], bool
            prefix_att_2d_masks_4d: (B, 1, S, S) fp32 additive mask.
            noise: 可选的初始噪声 [B, chunk_size, max_action_dim].
                   不传则随机生成.

        Returns:
            Action tensor [B, chunk_size, action_dim]
        """
        batch_size = tokens.shape[0]
        self._ensure_context()
        if _DEBUG:
            logger(f"--- forward() begin (batch_size={batch_size}) ---")
            logger(
                f"VLM has {len(self.vlm_input_data)} input slot(s); "
                f"received {len(images)} image(s) + tokens + masks "
                f"= {len(images) + 2} tensor(s)"
            )
            for i in range(len(self.vlm_input_data)):
                om_size = self.vlm_input_data[i]["size"]
                try:
                    om_dt = acl.mdl.get_input_data_type(self.vlm_model_desc, i)
                except Exception:
                    om_dt = -1
                logger(f"  VLM slot[{i}]: om_size={om_size} om_dtype_code={om_dt}")
            for i, img in enumerate(images):
                logger(f"  image[{i}]: {_tensor_stats(img)}  nbytes={img.nbytes}")
            logger(f"  tokens: {_tensor_stats(tokens)}  nbytes={tokens.nbytes}")
            logger(f"  masks:  {_tensor_stats(masks)}  nbytes={masks.nbytes}")

        # === Step 1: VLM H2D 传输 ===
        # VLM inputs (Plan A order): [image_0, ..., tokens, masks, prefix_att_2d_masks_4d]
        prefix_mask_fp32 = np.ascontiguousarray(prefix_att_2d_masks_4d, dtype=np.float32)
        vlm_inputs = [*images, tokens, masks, prefix_mask_fp32]

        if _DUMP_VLM_DIR is not None and not getattr(self, "_vlm_in_dumped", False):
            try:
                os.makedirs(_DUMP_VLM_DIR, exist_ok=True)
                for i, img in enumerate(images):
                    np.save(os.path.join(_DUMP_VLM_DIR, f"vlm_in_image_{i}.npy"), img)
                np.save(os.path.join(_DUMP_VLM_DIR, "vlm_in_lang_tokens.npy"), tokens)
                np.save(os.path.join(_DUMP_VLM_DIR, "vlm_in_lang_masks.npy"), masks)
                np.save(
                    os.path.join(_DUMP_VLM_DIR, "vlm_in_prefix_mask_4d.npy"),
                    prefix_mask_fp32,
                )
                self._vlm_in_dumped = True
                logger(
                    f"  dumped VLM inputs ({len(images)} image(s) + tokens + "
                    f"masks + prefix_mask_4d) under {_DUMP_VLM_DIR}"
                )
            except Exception as exc:
                logger(f"  VLM input dump failed: {exc}")

        if len(vlm_inputs) != len(self.vlm_input_data):
            raise ValueError(
                f"VLM input count mismatch: OM expects {len(self.vlm_input_data)} "
                f"tensor(s) but got {len(vlm_inputs)} "
                f"({len(images)} image(s) + tokens + masks + prefix_att_2d_masks_4d). "
                f"Check that config.image_features matches the exported ONNX."
            )
        for i, inp in enumerate(vlm_inputs):
            bytes_data = inp.tobytes()
            if _DEBUG and len(bytes_data) != self.vlm_input_data[i]["size"]:
                logger(
                    f"  WARNING: VLM input[{i}] byte-size mismatch "
                    f"host={len(bytes_data)} om_slot={self.vlm_input_data[i]['size']}"
                )
            bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
            ret = acl.rt.memcpy(
                self.vlm_input_data[i]["buffer"],
                self.vlm_input_data[i]["size"],
                bytes_ptr,
                len(bytes_data),
                ACL_MEMCPY_HOST_TO_DEVICE,
            )
            self._check_ret(ret, f"Failed to H2D VLM input {i}")

        # === Step 2: VLM Execute ===
        ret = acl.mdl.execute(self.vlm_model_id, self.vlm_input_dataset, self.vlm_output_dataset)
        self._check_ret(ret, "Failed to execute VLM")

        # kv_cache 和 pad_masks 现在在 VLM output buffer 中
        # 由于 AE input 共享这些 buffer，无需搬运

        if _DEBUG:
            self._debug_peek_vlm_outputs()

        if _DUMP_VLM_DIR is not None and not getattr(self, "_vlm_dumped", False):
            self._dump_vlm_outputs(_DUMP_VLM_DIR)
            self._vlm_dumped = True

        # === Step 3: Denoising Loop ===
        fdtype = self._float_np_dtype  # fp16 or fp32, auto-detected from OM
        noise_shape = (batch_size, self.chunk_size, self.max_action_dim)
        if noise is not None:
            x_t = noise.astype(fdtype)
            if _DEBUG:
                logger(f"  noise (external): {_tensor_stats(x_t)}")
        else:
            x_t = np.random.randn(*noise_shape).astype(fdtype)
            if _DEBUG:
                logger(f"  noise (random):   {_tensor_stats(x_t)}")

        dt = fdtype(-1.0 / self.num_inference_steps)
        time_val = fdtype(1.0)

        if _DUMP_AE_DIR is not None and not getattr(self, "_ae_in_dumped", False):
            try:
                os.makedirs(_DUMP_AE_DIR, exist_ok=True)
                self._dump_ae_shared_inputs(_DUMP_AE_DIR)
                np.save(os.path.join(_DUMP_AE_DIR, "ae_in_noise.npy"), x_t)
                self._ae_in_dumped = True
                logger(f"  dumped AE inputs (past_kv, pad_masks, noise) under {_DUMP_AE_DIR}")
            except Exception as exc:
                logger(f"  AE input dump failed: {exc}")

        step_idx = 0
        while time_val >= -dt / 2:
            # 准备 time 和 x_t
            time_arr = np.full((batch_size,), time_val, dtype=fdtype)

            if _DEBUG:
                logger(f"  [denoise step={step_idx:02d}] time={float(time_val):+.4f} x_t_in: {_tensor_stats(x_t)}")

            # H2D: time (index 2) 和 x_t (index 3)
            ae_dynamic_inputs = [time_arr, x_t]

            if _DUMP_AE_DIR is not None and not getattr(self, "_ae_dumped", False):
                try:
                    np.save(
                        os.path.join(_DUMP_AE_DIR, f"ae_in_time_step{step_idx:02d}.npy"),
                        time_arr,
                    )
                except Exception as exc:
                    logger(f"  AE time dump (step {step_idx}) failed: {exc}")

            for j, inp in enumerate(ae_dynamic_inputs):
                idx = len(self.vlm_output_data) + j  # 跳过共享的 buffer
                bytes_data = inp.tobytes()
                bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
                ret = acl.rt.memcpy(
                    self.ae_input_data[idx]["buffer"],
                    self.ae_input_data[idx]["size"],
                    bytes_ptr,
                    len(bytes_data),
                    ACL_MEMCPY_HOST_TO_DEVICE,
                )
                self._check_ret(ret, f"Failed to H2D AE input {idx}")

            # Execute Action Expert
            ret = acl.mdl.execute(self.ae_model_id, self.ae_input_dataset, self.ae_output_dataset)
            self._check_ret(ret, "Failed to execute Action Expert")

            # D2H: x_t (AE 的 sample_actions 内部已执行 Euler step,
            #       输出直接是更新后的 x_t, 不是 v_t)
            buffer_host, ret = acl.rt.malloc_host(self.ae_output_data[0]["size"])
            self._check_ret(ret, "Failed to malloc host buffer for AE D2H")
            ret = acl.rt.memcpy(
                buffer_host,
                self.ae_output_data[0]["size"],
                self.ae_output_data[0]["buffer"],
                self.ae_output_data[0]["size"],
                ACL_MEMCPY_DEVICE_TO_HOST,
            )
            self._check_ret(ret, "Failed to D2H AE output")

            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.ae_output_data[0]["size"])
            x_t = np.frombuffer(bytes_out, dtype=fdtype).reshape(*noise_shape)

            ret = acl.rt.free_host(buffer_host)
            self._check_ret(ret, "Failed to free host buffer")

            if _DEBUG:
                logger(f"  [denoise step={step_idx:02d}] x_t_out: {_tensor_stats(x_t)}")

            if _DUMP_AE_DIR is not None and not getattr(self, "_ae_dumped", False):
                try:
                    os.makedirs(_DUMP_AE_DIR, exist_ok=True)
                    save_path = os.path.join(_DUMP_AE_DIR, f"x_t_step{step_idx:02d}.npy")
                    np.save(save_path, x_t)
                    if _DEBUG:
                        logger(f"  dumped AE step{step_idx:02d} -> {save_path}")
                except Exception as exc:
                    logger(f"  AE dump (step {step_idx}) failed: {exc}")

            # AE 的 sample_actions 内部已执行 Euler step (x_t = x_t + dt * v_t),
            # 此处只推进时间, 不再重复做 Euler step
            time_val += dt
            step_idx += 1

        if _DUMP_AE_DIR is not None and not getattr(self, "_ae_dumped", False):
            self._ae_dumped = True
            logger(f"  AE trajectory dumped: {step_idx} step(s) under {_DUMP_AE_DIR}")

        # === Step 4: 返回结果 ===
        actions = torch.from_numpy(x_t.astype(np.float32))
        if _DEBUG:
            logger(f"--- forward() done: ran {step_idx} denoising step(s) ---")
        return actions

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _debug_peek_vlm_outputs(self) -> None:
        """D2H copy of every VLM output and print stats (debug only)."""
        for i, item in enumerate(self.vlm_output_data):
            size = item["size"]
            buffer_host, ret = acl.rt.malloc_host(size)
            if ret != 0:
                logger(f"  VLM output[{i}]: malloc_host failed ({ret})")
                continue
            ret = acl.rt.memcpy(buffer_host, size, item["buffer"], size, ACL_MEMCPY_DEVICE_TO_HOST)
            if ret != 0:
                logger(f"  VLM output[{i}]: D2H failed ({ret})")
                acl.rt.free_host(buffer_host)
                continue
            raw = acl.util.ptr_to_bytes(buffer_host, size)
            try:
                dt_code = acl.mdl.get_output_data_type(self.vlm_model_desc, i)
            except Exception:
                dt_code = -1
            np_dt_map = {0: np.float32, 1: np.float16, 9: np.int64, 12: np.bool_}
            np_dt = np_dt_map.get(dt_code, np.float16)
            try:
                arr = np.frombuffer(raw, dtype=np_dt)
                logger(f"  VLM output[{i}] (acl_code={dt_code} dtype={np_dt.__name__}): {_tensor_stats(arr)}")
            except Exception as exc:
                logger(f"  VLM output[{i}]: peek failed ({exc})")
            acl.rt.free_host(buffer_host)

    def _dump_ae_shared_inputs(self, out_dir: str) -> None:
        """D2H the shared kv_cache + pad_masks buffers and save as .npy."""
        np_dt_map = {0: np.float32, 1: np.float16, 9: np.int64, 12: np.bool_}
        names = ["ae_in_past_kv.npy", "ae_in_prefix_pad_masks.npy"]
        for i in range(min(2, len(self.ae_input_data))):
            item = self.ae_input_data[i]
            size = item["size"]
            buffer_host, ret = acl.rt.malloc_host(size)
            if ret != 0:
                logger(f"  AE shared dump[{i}]: malloc_host failed ({ret})")
                continue
            try:
                ret = acl.rt.memcpy(
                    buffer_host,
                    size,
                    item["buffer"],
                    size,
                    ACL_MEMCPY_DEVICE_TO_HOST,
                )
                if ret != 0:
                    logger(f"  AE shared dump[{i}]: D2H failed ({ret})")
                    continue
                raw = acl.util.ptr_to_bytes(buffer_host, size)
                try:
                    dt_code = acl.mdl.get_input_data_type(self.ae_model_desc, i)
                except Exception:
                    dt_code = -1
                np_dt = np_dt_map.get(dt_code, np.float16)
                try:
                    dims = acl.mdl.get_input_dims(self.ae_model_desc, i)[0]["dims"]
                except Exception:
                    dims = None
                arr = np.frombuffer(raw, dtype=np_dt).copy()
                if dims:
                    try:
                        arr = arr.reshape(tuple(int(d) for d in dims))
                    except Exception as exc:
                        logger(f"  AE shared dump[{i}]: reshape({dims}) failed ({exc})")
                path = os.path.join(out_dir, names[i] if i < len(names) else f"ae_in_{i}.npy")
                np.save(path, arr)
                logger(f"  dumped AE input[{i}] -> {path}  {_tensor_stats(arr)}")
            finally:
                acl.rt.free_host(buffer_host)

    def _dump_vlm_outputs(self, out_dir: str) -> None:
        """D2H copy every VLM output and save as ``<name>.npy`` under *out_dir*."""
        os.makedirs(out_dir, exist_ok=True)
        output_names = ["past_kv_tensor", "prefix_pad_masks"]
        np_dt_map = {0: np.float32, 1: np.float16, 9: np.int64, 12: np.bool_}

        for i, item in enumerate(self.vlm_output_data):
            name = output_names[i] if i < len(output_names) else f"output_{i}"
            size = item["size"]
            buffer_host, ret = acl.rt.malloc_host(size)
            if ret != 0:
                logger(f"  dump VLM output[{i}]({name}): malloc_host failed ({ret})")
                continue
            try:
                ret = acl.rt.memcpy(buffer_host, size, item["buffer"], size, ACL_MEMCPY_DEVICE_TO_HOST)
                if ret != 0:
                    logger(f"  dump VLM output[{i}]({name}): D2H failed ({ret})")
                    continue
                raw = acl.util.ptr_to_bytes(buffer_host, size)
                try:
                    dt_code = acl.mdl.get_output_data_type(self.vlm_model_desc, i)
                except Exception:
                    dt_code = -1
                np_dt = np_dt_map.get(dt_code, np.float16)
                try:
                    dims = acl.mdl.get_output_dims(self.vlm_model_desc, i)[0]["dims"]
                except Exception:
                    dims = None
                arr = np.frombuffer(raw, dtype=np_dt).copy()
                if dims:
                    try:
                        arr = arr.reshape(tuple(int(d) for d in dims))
                    except Exception as exc:
                        logger(f"  dump VLM output[{i}]({name}): reshape({dims}) failed ({exc})")
                path = os.path.join(out_dir, f"{name}.npy")
                np.save(path, arr)
                logger(f"  dumped VLM output[{i}] -> {path}  {_tensor_stats(arr)}")
            finally:
                acl.rt.free_host(buffer_host)

    def __del__(self):
        """释放所有 ACL 资源."""
        if acl is None or not hasattr(self, "vlm_input_data"):
            return
        try:
            # VLM input buffers (owned)
            for item in self.vlm_input_data:
                if item.get("owned", False):
                    acl.destroy_data_buffer(item["data"])
                    acl.rt.free(item["buffer"])
            acl.mdl.destroy_dataset(self.vlm_input_dataset)

            # VLM output buffers (owned, also shared source for AE inputs)
            for item in self.vlm_output_data:
                if item.get("owned", False):
                    acl.destroy_data_buffer(item["data"])
                    acl.rt.free(item["buffer"])
            acl.mdl.destroy_dataset(self.vlm_output_dataset)

            # AE input buffers
            for item in self.ae_input_data:
                acl.destroy_data_buffer(item["data"])  # 总是销毁 data_buffer 引用
                if item.get("owned", False):
                    acl.rt.free(item["buffer"])  # 只释放独立分配的 buffer
            acl.mdl.destroy_dataset(self.ae_input_dataset)

            # AE output buffers (owned)
            for item in self.ae_output_data:
                if item.get("owned", False):
                    acl.destroy_data_buffer(item["data"])
                    acl.rt.free(item["buffer"])
            acl.mdl.destroy_dataset(self.ae_output_dataset)

            # 卸载模型
            acl.mdl.destroy_desc(self.vlm_model_desc)
            acl.mdl.unload(self.vlm_model_id)
            acl.mdl.destroy_desc(self.ae_model_desc)
            acl.mdl.unload(self.ae_model_id)

            # 销毁显式 context (与 __init__ 中的 create_context 配对)
            ctx = getattr(self, "_context", None)
            if ctx is not None:
                acl.rt.destroy_context(ctx)
                self._context = None

            # ACL 清理
            acl.rt.reset_device(self.device_id)
            acl.finalize()
            logger("PI05OMModel resources released")
        except Exception:
            # Interpreter is shutting down — nothing meaningful we can do.
            pass

    def _check_ret(self, ret, msg):
        """检查 ACL 返回值."""
        if ret != 0:
            raise RuntimeError(f"{msg}, Error code: {ret}")

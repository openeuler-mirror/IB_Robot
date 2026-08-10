# tensormsg/converter.py
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
import torch
from rosidl_runtime_py.utilities import get_message
from torch import Tensor

from tensormsg.registry import (
    DECODER_REGISTRY,
    ENCODER_REGISTRY,
    register_decoder,
    register_encoder,
)
from tensormsg.utils import dot_get, dot_set, nearest_resize_any, nearest_resize_rgb

_COLOR_ENCODING_CHANNELS = {
    "mono8": 1,
    "8uc1": 1,
    "rgb8": 3,
    "bgr8": 3,
    "8uc3": 3,
    "rgba8": 4,
    "bgra8": 4,
    "8uc4": 4,
}
_BT709_COLOR_RANGES = {"limited", "full"}


class TensorMsgConverter:
    """Central converter for ROS messages and Tensors."""

    @staticmethod
    def encode(
        ros_type: str,
        data: np.ndarray | Tensor | Sequence,
        names: list[str] | None = None,
        clamp: tuple[float, float] | None = None,
    ) -> Any:
        encoder = ENCODER_REGISTRY.get(ros_type)
        if not encoder:
            # Fallback for simple types or explicit dotted paths if names are provided
            if names:
                return _encode_via_dotted_paths(ros_type, names, data, clamp)
            raise ValueError(f"No encoder registered for {ros_type}")
        return encoder(names, data, clamp)

    @staticmethod
    def decode(msg, spec: Any = None) -> np.ndarray:
        """
        Decode a ROS message to a numpy array.

        spec can be an object with .names, .image_encoding, .image_resize attributes.
        """
        pkg_name = msg.__class__.__module__.split(".")[0]
        ros_type = f"{pkg_name}/msg/{msg.__class__.__name__}"

        decoder = DECODER_REGISTRY.get(ros_type)
        if not decoder:
            # Try to decode via names if spec has them
            if spec and hasattr(spec, "names") and spec.names:
                return _decode_via_names(msg, spec.names)
            raise ValueError(f"No decoder registered for {ros_type}")
        return decoder(msg, spec)

    @staticmethod
    def to_variant(batch: dict[str, Any]) -> Any:
        """Encode a dictionary of Tensors into a ibrobot_msgs/msg/VariantsList."""
        msg_cls = get_message("ibrobot_msgs/msg/VariantsList")
        msg = msg_cls()
        msg.variants = []

        for key, value in batch.items():
            if not any(key.startswith(p) for p in ["task", "observation", "action"]):
                continue

            variant_msg = get_message("ibrobot_msgs/msg/Variant")()
            variant_msg.key = key

            if isinstance(value, Tensor | np.ndarray):
                tensor = value if isinstance(value, Tensor) else torch.from_numpy(value)
                _fill_variant_from_tensor(variant_msg, tensor)
            elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                variant_msg.type = "string_array"
                variant_msg.string_array = value
            else:
                continue
            msg.variants.append(variant_msg)
        return msg

    @staticmethod
    def from_variant(msg, device: torch.device | None = None) -> dict[str, Any]:
        """Decode a ibrobot_msgs/msg/VariantsList into a dictionary of Tensors."""
        result = {}
        for variant_msg in msg.variants:
            result[str(variant_msg.key)] = _decode_variant(variant_msg, device)
        return result


# ---------- Internal Helpers ----------


def _validated_resize(resize: Sequence[int] | None) -> tuple[int, int] | None:
    if resize is None:
        return None
    if isinstance(resize, str | bytes):
        raise ValueError("image resize must contain exactly (height, width)")
    try:
        height, width = resize
    except (TypeError, ValueError) as exc:
        raise ValueError("image resize must contain exactly (height, width)") from exc
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, Integral)
        or not isinstance(width, Integral)
    ):
        raise ValueError("image resize dimensions must be positive integers")
    resize_hw = (int(height), int(width))
    if resize_hw[0] <= 0 or resize_hw[1] <= 0:
        raise ValueError("image resize dimensions must be positive integers")
    return resize_hw


def _image_rows(msg: Any, packed_row_bytes: int) -> np.ndarray:
    height = int(msg.height)
    width = int(msg.width)
    if height <= 0 or width <= 0:
        raise ValueError(f"image dimensions must be positive, got {height}x{width}")

    step = int(getattr(msg, "step", 0)) or packed_row_bytes
    if step < packed_row_bytes:
        raise ValueError(f"image step {step} is smaller than packed row size {packed_row_bytes}")

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    required = height * step
    if raw.size < required:
        raise ValueError(
            f"image data has {raw.size} bytes, expected at least {required} for height={height}, step={step}"
        )
    return raw[:required].reshape(height, step)[:, :packed_row_bytes]


def decoded_frame_to_hwc_uint8(
    frame: np.ndarray,
    *,
    encoding: str,
    output_encoding: str = "rgb8",
    resize: Sequence[int] | None = None,
) -> np.ndarray:
    """Convert a backend-decoded color frame to contiguous RGB or BGR HWC uint8."""
    source_encoding = str(encoding).lower()
    target_encoding = str(output_encoding).lower()
    if source_encoding not in _COLOR_ENCODING_CHANNELS:
        raise ValueError(f"Unsupported color image encoding '{source_encoding}'")
    if target_encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"output_encoding must be 'rgb8' or 'bgr8', got '{target_encoding}'")

    image = np.asarray(frame)
    if image.dtype != np.uint8:
        raise ValueError(f"color image dtype must be uint8, got {image.dtype}")
    channels = _COLOR_ENCODING_CHANNELS[source_encoding]
    expected_shape = image.shape[:2] if channels == 1 else (*image.shape[:2], channels)
    if image.ndim != (2 if channels == 1 else 3) or image.shape != expected_shape:
        shape = "HxW" if channels == 1 else f"HxWx{channels}"
        raise ValueError(f"encoding '{source_encoding}' requires a {shape} frame, got shape {image.shape}")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"image dimensions must be positive, got {image.shape[0]}x{image.shape[1]}")

    if channels == 1:
        color = np.repeat(image[..., None], 3, axis=-1)
    else:
        color = image[..., :3]
        source_is_rgb = source_encoding in ("rgb8", "rgba8")
        if source_is_rgb != (target_encoding == "rgb8"):
            color = color[..., ::-1]

    resize_hw = _validated_resize(resize)
    if resize_hw is not None:
        color = nearest_resize_rgb(color, *resize_hw)
    return np.ascontiguousarray(color, dtype=np.uint8)


def ros_image_to_hwc_uint8(
    msg: Any, *, output_encoding: str = "rgb8", resize: Sequence[int] | None = None
) -> np.ndarray:
    """Extract a padded ``sensor_msgs/Image`` as contiguous encoder-ready HWC uint8."""
    encoding = str(getattr(msg, "encoding", "bgr8")).lower()
    channels = _COLOR_ENCODING_CHANNELS.get(encoding)
    if channels is None:
        raise ValueError(f"Unsupported color image encoding '{encoding}'")
    packed = _image_rows(msg, int(msg.width) * channels)
    if channels == 1:
        frame = packed.reshape(int(msg.height), int(msg.width))
    else:
        frame = packed.reshape(int(msg.height), int(msg.width), channels)
    return decoded_frame_to_hwc_uint8(
        frame,
        encoding=encoding,
        output_encoding=output_encoding,
        resize=resize,
    )


def decoded_frame_to_chw_float(
    frame: np.ndarray,
    *,
    encoding: str,
    output_encoding: str = "rgb8",
    resize: Sequence[int] | None = None,
) -> np.ndarray:
    """Convert a backend-decoded color frame to canonical contiguous CHW float32."""
    hwc = decoded_frame_to_hwc_uint8(
        frame,
        encoding=encoding,
        output_encoding=output_encoding,
        resize=resize,
    )
    return np.ascontiguousarray(np.transpose(hwc, (2, 0, 1)), dtype=np.float32) / 255.0


def hwc_uint8_to_nv12(
    frame: np.ndarray,
    *,
    encoding: str = "rgb8",
    color_space: str = "bt709",
    color_range: str = "limited",
    stride: int | None = None,
) -> np.ndarray:
    """Convert RGB/BGR HWC uint8 into an optionally padded NV12 surface."""
    if color_space.lower() != "bt709":
        raise ValueError(f"NV12 color_space must be 'bt709', got {color_space!r}")
    normalized_range = color_range.lower()
    if normalized_range not in _BT709_COLOR_RANGES:
        raise ValueError(f"NV12 color_range must be one of {sorted(_BT709_COLOR_RANGES)}, got {color_range!r}")
    source_encoding = str(encoding).lower()
    source = np.asarray(frame)
    direct_channels = source_encoding in {"rgb8", "bgr8"}
    if direct_channels:
        if source.dtype != np.uint8:
            raise ValueError(f"color image dtype must be uint8, got {source.dtype}")
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(f"encoding '{source_encoding}' requires a HxWx3 frame, got shape {source.shape}")
        height, width, _ = source.shape
        if height <= 0 or width <= 0:
            raise ValueError(f"image dimensions must be positive, got {height}x{width}")
        rgb = None
    else:
        rgb = decoded_frame_to_hwc_uint8(frame, encoding=encoding, output_encoding="rgb8")
        height, width, _ = rgb.shape
    if height % 2 or width % 2:
        raise ValueError(f"NV12 requires even dimensions, got {height}x{width}")
    if stride is None:
        surface_stride = width
    elif isinstance(stride, bool) or not isinstance(stride, Integral):
        raise ValueError("NV12 stride must be a positive integer")
    else:
        surface_stride = int(stride)
    if surface_stride < width:
        raise ValueError(f"NV12 stride {surface_stride} is smaller than width {width}")

    if direct_channels:
        red = source[..., 0 if source_encoding == "rgb8" else 2].astype(np.int32)
        green = source[..., 1].astype(np.int32)
        blue = source[..., 2 if source_encoding == "rgb8" else 0].astype(np.int32)
    else:
        assert rgb is not None
        red = rgb[..., 0].astype(np.int32)
        green = rgb[..., 1].astype(np.int32)
        blue = rgb[..., 2].astype(np.int32)

    # Keep the conversion integer-only on the hot RGB/BGR path.  The fixed
    # point coefficients preserve the BT.709 equations while avoiding the
    # float32 channel and subsampling temporaries on small ARM CPUs.
    scale = 1_000_000
    if normalized_range == "limited":
        y_coefficients = (182586, 614231, 62007)
        u_coefficients = (-100644, -338572, 439216)
        v_coefficients = (439216, -398942, -40274)
        y_offset = 16
    else:
        y_coefficients = (212600, 715200, 72200)
        u_coefficients = (-114572, -385428, 500000)
        v_coefficients = (500000, -454153, -45847)
        y_offset = 0

    y_numerator = y_coefficients[0] * red + y_coefficients[1] * green + y_coefficients[2] * blue
    y_plane = np.clip((y_numerator + y_offset * scale + scale // 2) // scale, 0, 255).astype(np.uint8)
    red_sum = red[0::2, 0::2] + red[0::2, 1::2] + red[1::2, 0::2] + red[1::2, 1::2]
    green_sum = green[0::2, 0::2] + green[0::2, 1::2] + green[1::2, 0::2] + green[1::2, 1::2]
    blue_sum = blue[0::2, 0::2] + blue[0::2, 1::2] + blue[1::2, 0::2] + blue[1::2, 1::2]
    u_numerator = u_coefficients[0] * red_sum + u_coefficients[1] * green_sum + u_coefficients[2] * blue_sum
    v_numerator = v_coefficients[0] * red_sum + v_coefficients[1] * green_sum + v_coefficients[2] * blue_sum
    chroma_denominator = 4 * scale
    chroma_offset = 128 * chroma_denominator + chroma_denominator // 2
    u_subsampled = np.clip((u_numerator + chroma_offset) // chroma_denominator, 0, 255).astype(np.uint8)
    v_subsampled = np.clip((v_numerator + chroma_offset) // chroma_denominator, 0, 255).astype(np.uint8)
    surface = np.zeros((height + height // 2, surface_stride), dtype=np.uint8)
    surface[:height, :width] = y_plane
    surface[height:, :width:2] = u_subsampled
    surface[height:, 1:width:2] = v_subsampled
    return surface


def nv12_to_hwc_uint8(
    frame: np.ndarray | bytes | bytearray | memoryview,
    *,
    width: int,
    height: int,
    stride: int | None = None,
    output_encoding: str = "rgb8",
    color_space: str = "bt709",
    color_range: str = "limited",
    resize: Sequence[int] | None = None,
) -> np.ndarray:
    """Convert a packed or padded NV12 surface into RGB/BGR HWC uint8."""
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError(f"NV12 requires positive even dimensions, got {height}x{width}")
    if color_space.lower() != "bt709":
        raise ValueError(f"NV12 color_space must be 'bt709', got {color_space!r}")
    normalized_range = color_range.lower()
    if normalized_range not in _BT709_COLOR_RANGES:
        raise ValueError(f"NV12 color_range must be one of {sorted(_BT709_COLOR_RANGES)}, got {color_range!r}")
    target_encoding = output_encoding.lower()
    if target_encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"output_encoding must be 'rgb8' or 'bgr8', got {output_encoding!r}")
    surface_stride = width if stride is None else int(stride)
    if surface_stride < width:
        raise ValueError(f"NV12 stride {surface_stride} is smaller than width {width}")
    raw = np.frombuffer(frame, dtype=np.uint8) if not isinstance(frame, np.ndarray) else np.asarray(frame)
    required = (height + height // 2) * surface_stride
    if raw.dtype != np.uint8:
        raise ValueError(f"NV12 surface dtype must be uint8, got {raw.dtype}")
    if raw.size < required:
        raise ValueError(f"NV12 surface has {raw.size} bytes, expected at least {required}")
    surface = raw.reshape(-1)[:required].reshape(height + height // 2, surface_stride)
    y_plane = surface[:height, :width].astype(np.float32)
    uv_plane = surface[height:, :width]
    u_plane = np.repeat(np.repeat(uv_plane[:, 0::2], 2, axis=0), 2, axis=1).astype(np.float32) - 128.0
    v_plane = np.repeat(np.repeat(uv_plane[:, 1::2], 2, axis=0), 2, axis=1).astype(np.float32) - 128.0

    if normalized_range == "limited":
        luminance = 1.164384 * (y_plane - 16.0)
        red = luminance + 1.792741 * v_plane
        green = luminance - 0.213249 * u_plane - 0.532909 * v_plane
        blue = luminance + 2.112402 * u_plane
    else:
        red = y_plane + 1.5748 * v_plane
        green = y_plane - 0.187324 * u_plane - 0.468124 * v_plane
        blue = y_plane + 1.8556 * u_plane
    rgb = _rounded_uint8(np.stack((red, green, blue), axis=-1))
    if target_encoding == "bgr8":
        rgb = rgb[..., ::-1]
    resize_hw = _validated_resize(resize)
    if resize_hw is not None:
        rgb = nearest_resize_rgb(rgb, *resize_hw)
    return np.ascontiguousarray(rgb)


def _rounded_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _encode_via_dotted_paths(
    ros_type: str, names: list[str], data: Any, clamp: tuple[float, float] | None = None
) -> Any:
    msg_cls = get_message(ros_type)
    msg = msg_cls()
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])
    for i, path in enumerate(names):
        if i < arr.size:
            dot_set(msg, path, float(arr[i]))
    return msg


def _decode_via_names(msg, names: list[str]) -> np.ndarray:
    out: list[float] = []
    for name in names:
        try:
            out.append(float(dot_get(msg, name)))
        except Exception:
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float32)


def _decode_name_value_fields(msg: Any, names: list[str], value_field: str, prefix: str) -> np.ndarray:
    values: list[float] = []
    msg_names = list(getattr(msg, "name", []))
    msg_values = list(getattr(msg, value_field, []))

    for selector in names:
        key = selector.split(".", 1)[1] if selector.startswith(f"{prefix}.") else selector
        if key not in msg_names:
            values.append(float("nan"))
            continue

        idx = msg_names.index(key)
        if idx < len(msg_values):
            values.append(float(msg_values[idx]))
        else:
            values.append(float("nan"))

    return np.asarray(values, dtype=np.float32)


def _fill_variant_from_tensor(variant_msg, vec: Tensor):
    if vec.dtype == torch.bool:
        variant_msg.type = "bool_array"
        variant_msg.bool_array = vec.reshape(-1).tolist()
    elif vec.dtype == torch.int32:
        variant_msg.type = "int_32_array"
        variant_msg.int_32_array = _create_multiarray_msg(vec, "Int32")
    elif vec.dtype == torch.int64:
        variant_msg.type = "int_64_array"
        variant_msg.int_64_array = _create_multiarray_msg(vec, "Int64")
    elif vec.dtype == torch.float32:
        variant_msg.type = "float_32_array"
        variant_msg.float_32_array = _create_multiarray_msg(vec, "Float32")
    elif vec.dtype == torch.float64:
        variant_msg.type = "float_64_array"
        variant_msg.float_64_array = _create_multiarray_msg(vec, "Float64")
    else:
        raise ValueError(f"Unsupported dtype {vec.dtype}")


def _create_multiarray_msg(vec: Tensor | np.ndarray, msg_type: str):
    if isinstance(vec, Tensor):
        v_np = vec.detach().cpu().numpy()
    else:
        v_np = vec

    msg_cls_name = f"std_msgs/msg/{msg_type}MultiArray"
    msg = get_message(msg_cls_name)()
    msg.data = v_np.reshape(-1).tolist()

    for size in v_np.shape:
        dim = get_message("std_msgs/msg/MultiArrayDimension")()
        dim.size = int(size)
        msg.layout.dim.append(dim)
    return msg


def _decode_variant(msg, device: torch.device | None = None) -> Any:
    v_type = str(msg.type).strip()
    if v_type == "string_array":
        return list(msg.string_array)

    type_map = {
        "bool_array": (msg.bool_array, torch.bool),
        "int_32_array": (msg.int_32_array, torch.int32),
        "int_64_array": (msg.int_64_array, torch.int64),
        "float_32_array": (msg.float_32_array, torch.float32),
        "float_64_array": (msg.float_64_array, torch.float64),
    }

    if v_type not in type_map:
        raise ValueError(f"Unsupported variant type: {v_type}")

    data_source, torch_dtype = type_map[v_type]

    if v_type == "bool_array":
        res = torch.tensor(data_source, dtype=torch_dtype).unsqueeze(0)
    else:
        res = torch.tensor(data_source.data, dtype=torch_dtype)
        if data_source.layout.dim:
            shape = tuple(dim.size for dim in data_source.layout.dim)
            res = res.reshape(shape)

    if device:
        res = res.to(device)
    return res


# ---------- Registration of Standard Types ----------


@register_encoder("geometry_msgs/msg/Twist")
def _enc_twist(names, data, clamp):
    if names:
        return _encode_via_dotted_paths("geometry_msgs/msg/Twist", names, data, clamp)
    msg = get_message("geometry_msgs/msg/Twist")()
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])
    if len(arr) >= 1:
        msg.linear.x = float(arr[0])
    if len(arr) >= 2:
        msg.angular.z = float(arr[1])
    return msg


@register_decoder("sensor_msgs/msg/Image")
def _dec_image(msg, spec):
    """
    Robust image decoder for common ROS encodings.

    Ported from rosetta/common/decoders.py
    """
    if spec and hasattr(spec, "names") and spec.names:
        return _decode_via_names(msg, spec.names)

    h, w = int(msg.height), int(msg.width)
    enc = getattr(msg, "encoding", "bgr8").lower()
    resize_hw = _validated_resize(spec.image_resize if spec and hasattr(spec, "image_resize") else None)

    if enc in ("32fc1", "32fc"):
        byte_order = ">" if bool(getattr(msg, "is_bigendian", False)) else "<"
        packed = np.ascontiguousarray(_image_rows(msg, w * 4))
        arr = packed.view(np.dtype(f"{byte_order}f4")).reshape(h, w).astype(np.float32)
        hwc = arr[..., None]
        if resize_hw is not None:
            hwc = nearest_resize_any(hwc, *resize_hw)
        hwc_normalized = np.where(np.isfinite(hwc), np.clip(hwc, 0, 50) / 50, hwc)
        return np.ascontiguousarray(
            np.transpose(np.repeat(hwc_normalized, 3, axis=-1), (2, 0, 1)),
            dtype=np.float32,
        )

    elif enc in ("16uc1", "mono16"):
        byte_order = ">" if bool(getattr(msg, "is_bigendian", False)) else "<"
        packed = np.ascontiguousarray(_image_rows(msg, w * 2))
        arr16 = packed.view(np.dtype(f"{byte_order}u2")).reshape(h, w).astype(np.uint16)
        arr_m = arr16.astype(np.float32)
        arr_m[arr16 == 0] = np.nan
        arr_m[arr16 != 0] *= 1.0 / 1000.0
        hwc = arr_m[..., None]
        if resize_hw is not None:
            hwc = nearest_resize_any(hwc, *resize_hw)
        hwc_normalized = np.where(np.isfinite(hwc), np.clip(hwc, 0, 10) / 10, hwc)
        return np.ascontiguousarray(
            np.transpose(np.repeat(hwc_normalized, 3, axis=-1), (2, 0, 1)),
            dtype=np.float32,
        )

    hwc_rgb = ros_image_to_hwc_uint8(msg, resize=resize_hw)
    return np.ascontiguousarray(np.transpose(hwc_rgb, (2, 0, 1)), dtype=np.float32) / 255.0


@register_decoder("sensor_msgs/msg/JointState")
def _dec_joint_state(msg, spec):
    if spec and hasattr(spec, "names") and spec.names:
        return _decode_via_names(msg, spec.names)
    return np.asarray(msg.position, dtype=np.float32)


@register_decoder("ibrobot_msgs/msg/JointCurrent")
def _dec_joint_current(msg, spec):
    if spec and hasattr(spec, "names") and spec.names:
        return _decode_name_value_fields(msg, spec.names, "current", "current")
    return np.asarray(msg.current, dtype=np.float32)


@register_encoder("sensor_msgs/msg/JointState")
def _enc_joint_state(names, data, clamp):
    msg = get_message("sensor_msgs/msg/JointState")()
    if not names:
        return msg
    msg.name = [n.split(".")[1] for n in names]
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])
    for i, path in enumerate(names):
        dot_set(msg, path, float(arr[i]))
    return msg


@register_decoder("std_msgs/msg/Float32MultiArray")
def _dec_f32(msg, spec):
    return np.asarray(msg.data, dtype=np.float32)


@register_decoder("ibrobot_msgs/msg/StampedFloat32MultiArray")
def _dec_stamped_f32(msg, spec):
    """
    Decode a StampedFloat32MultiArray to a float32 ndarray.

    Only the value payload is decoded; header.stamp is left untouched.
    The timestamp is still read by the existing stamp_from_header_ns(message)
    path in robot_config.contract_utils, never embedded into the tensor.
    """
    return np.asarray(msg.value.data, dtype=np.float32)


@register_decoder("std_msgs/msg/Float64MultiArray")
def _dec_f64(msg, spec):
    return np.asarray(msg.data, dtype=np.float64)


@register_decoder("std_msgs/msg/Int32MultiArray")
def _dec_i32(msg, spec):
    return np.asarray(msg.data, dtype=np.int32)


@register_decoder("sensor_msgs/msg/PointCloud2")
def _dec_pointcloud2(msg, spec):
    """
    解码无序 PointCloud2（height=1, width=N_valid）.

    返回 {"xyz": (N,3) float32, "rgb": (N,3) uint8}.
    """
    import sensor_msgs_py.point_cloud2 as pc2

    N = msg.width  # height=1 for unorganized cloud

    xyz = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=False)
    xyz = xyz.reshape(N, 3).astype(np.float32)

    rgb = np.zeros((N, 3), dtype=np.uint8)
    field_names = [f.name for f in msg.fields]
    if "rgb" in field_names:
        rgb_raw = pc2.read_points_numpy(msg, field_names=("rgb",), skip_nans=False)
        rgb_packed = rgb_raw.reshape(N).view(np.uint32)
        rgb[:, 0] = (rgb_packed >> 16) & 0xFF  # R
        rgb[:, 1] = (rgb_packed >> 8) & 0xFF  # G
        rgb[:, 2] = rgb_packed & 0xFF  # B

    return {"xyz": xyz, "rgb": rgb}

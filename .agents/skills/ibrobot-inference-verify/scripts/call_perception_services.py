#!/usr/bin/env python3
"""Call perception typed services with a synthetic image and print results.

Contract notes baked in (see references/troubleshooting.md):
- SigLIP2 EncodeEmbeddings only encodes provided masks; an image-only request
  returns "encoded 0 masks" without running the encoder.
- The mask message must carry the EXACT same header stamp as the image.
- GenerateMasks response masks live under `detections.detections` (Detection2D[]
  with a `mask` image field), not a `masks` field.

Usage: python3 call_perception_services.py [--include-grounding] [--siglip2-timeout 120] [--sam2-timeout 240]
"""

import argparse

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from ibrobot_msgs.srv import EncodeEmbeddings, GenerateMasks, GroundingDetect, RecognizeTags

SIZE = (480, 640)  # height, width


def make_image(node: Node, stamp) -> Image:
    h, w = SIZE
    img = Image()
    img.header.stamp = stamp
    img.header.frame_id = "camera"
    img.height, img.width = h, w
    img.encoding = "rgb8"
    img.is_bigendian = False
    img.step = w * 3
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[120:360, 180:460] = [220, 40, 30]  # red rectangle
    img.data = data.tobytes()
    return img


def make_mask(node: Node, stamp) -> Image:
    h, w = SIZE
    m = Image()
    m.header.stamp = stamp  # must match the source image exactly
    m.header.frame_id = "camera"
    m.height, m.width = h, w
    m.encoding = "mono8"
    m.is_bigendian = False
    m.step = w
    data = np.zeros((h, w), dtype=np.uint8)
    data[120:360, 180:460] = 255
    m.data = data.tobytes()
    return m


def call(node: Node, cli, req, timeout: float):
    if not cli.wait_for_service(timeout_sec=10):
        return "SERVICE_NOT_AVAILABLE"
    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if future.result() is None:
        return f"CALL_TIMEOUT({timeout}s)"
    return future.result()


def info(res, extra: str = "") -> str:
    return (
        f"success={res.success} state={res.model.runtime_state} "
        f"infer_ms={res.inference_time_ms:.1f} msg='{res.message[:120]}'{extra}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--include-grounding", action="store_true", help="also call GroundingDetect (needs a host-runnable bundle)"
    )
    p.add_argument("--siglip2-timeout", type=float, default=120)
    p.add_argument("--ram-timeout", type=float, default=180)
    p.add_argument("--sam2-timeout", type=float, default=240)  # CPU automatic masks ~100 s
    p.add_argument("--grounding-timeout", type=float, default=180)
    args = p.parse_args()

    rclpy.init()
    node = Node("perception_verify")
    results = {}

    cli = node.create_client(EncodeEmbeddings, "/perception/siglip2/encode_embeddings")
    stamp = node.get_clock().now().to_msg()
    req = EncodeEmbeddings.Request()
    req.image = make_image(node, stamp)
    req.masks = [make_mask(node, stamp)]
    req.candidate_labels = ["red object", "background"]
    r = call(node, cli, req, args.siglip2_timeout)
    results["siglip2"] = (
        info(r, f" embeddings={len(r.results)} dims={[e.embedding_dim for e in r.results]}")
        if not isinstance(r, str)
        else r
    )

    cli = node.create_client(RecognizeTags, "/perception/ram_plus/recognize_tags")
    req = RecognizeTags.Request()
    req.image = make_image(node, node.get_clock().now().to_msg())
    req.score_threshold = 0.5
    r = call(node, cli, req, args.ram_timeout)
    results["ram_plus"] = info(r, f" tags={len(r.tags)}") if not isinstance(r, str) else r

    cli = node.create_client(GenerateMasks, "/perception/sam2/generate_masks")
    req = GenerateMasks.Request()
    req.image = make_image(node, node.get_clock().now().to_msg())
    req.max_masks = 5
    r = call(node, cli, req, args.sam2_timeout)
    results["sam2"] = info(r, f" masks={len(r.detections.detections)}") if not isinstance(r, str) else r

    if args.include_grounding:
        cli = node.create_client(GroundingDetect, "/perception/grounding_dino/detect")
        req = GroundingDetect.Request()
        req.image = make_image(node, node.get_clock().now().to_msg())
        req.text_prompt = "banana . red object ."
        req.box_threshold = 0.3
        r = call(node, cli, req, args.grounding_timeout)
        if isinstance(r, str):
            results["grounding_dino"] = r
        else:
            dets = r.detections.detections
            boxes = [f"{d.label}:{d.confidence:.2f}@{[round(v, 1) for v in d.bbox]}" for d in dets]
            results["grounding_dino"] = info(r, f" boxes={len(dets)} {boxes[:4]}")

    for k, v in results.items():
        print(f"[{k}] {v}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

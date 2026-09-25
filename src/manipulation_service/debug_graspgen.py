#!/usr/bin/env python3
"""Standalone test for GraspGen wrapper using existing perception_service output data.

Saves results as JSON and produces Open3D visualizations:
  - grasp_result.json          — per-grasp pose, confidence, collision status
  - object_cloud.ply           — segmented object point cloud used for grasping
  - scene_cloud.ply            — background scene point cloud used for collision
  - grasp_cloud.ply            — object + scene point cloud (for reference)
  - grasp_grippers.ply         — transformed gripper meshes for top grasps
  - grasp_lines.ply            — transformed gripper control-point wireframes
  - grasp_preview.png          — top-down screenshot of all grasps on point cloud
  - (with --show) interactive Open3D viewer window
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manipulation_service.graspgen_wrapper import DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP, GraspGenWrapper

_LEGACY_INTRINSICS = (614.0, 614.0, 320.0, 240.0)


def _parse_float_csv(value: str) -> list[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def _print_cuda_preflight():
    try:
        import torch
    except ModuleNotFoundError:
        print("[cuda] torch: MISSING")
        return

    print(
        "[cuda] "
        f"torch={torch.__version__}, "
        f"torch.version.cuda={torch.version.cuda}, "
        f"available={torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        print(f"[cuda] device={torch.cuda.get_device_name(0)}")


def _points_from_pixels(depth_m, ys, xs, fx, fy, cx, cy):
    zs = depth_m[ys, xs]
    x3d = (xs - cx) * zs / fx
    y3d = (ys - cy) * zs / fy
    return np.stack([x3d, y3d, zs], axis=-1).astype(np.float32)


def _build_point_clouds(depth_m, mask, fx, fy, cx, cy):
    h, w = depth_m.shape
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    valid = depth_m > 0
    obj_mask = valid & (mask > 0)
    scene_mask = valid & ~obj_mask

    obj_ys, obj_xs = np.where(obj_mask)
    scene_ys, scene_xs = np.where(scene_mask)

    object_pts = _points_from_pixels(depth_m, obj_ys, obj_xs, fx, fy, cx, cy)
    scene_pts = _points_from_pixels(depth_m, scene_ys, scene_xs, fx, fy, cx, cy)

    object_colors = np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (len(object_pts), 1))
    scene_colors = np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (len(scene_pts), 1))

    all_pts = np.vstack([object_pts, scene_pts]).astype(np.float32)
    all_colors = np.vstack([object_colors, scene_colors]).astype(np.uint8)
    return scene_pts, scene_colors, object_pts, object_colors, all_pts, all_colors


def _load_result_metadata(data_dir):
    result_path = data_dir / "result.json"
    if not result_path.exists():
        return {}
    return json.loads(result_path.read_text(encoding="utf-8"))


def _intrinsics_from_mapping(data):
    if not isinstance(data, dict):
        return None

    k = data.get("k") or data.get("K")
    if isinstance(k, list) and len(k) == 9:
        return float(k[0]), float(k[4]), float(k[2]), float(k[5])

    keys = ("fx", "fy", "cx", "cy")
    if all(key in data for key in keys):
        return tuple(float(data[key]) for key in keys)

    return None


def _intrinsics_from_metadata(metadata):
    candidates = [
        metadata.get("camera_info"),
        metadata.get("camera_intrinsics"),
        metadata.get("pointcloud", {}).get("camera_info", {}),
        metadata.get("pointcloud", {}).get("camera_intrinsics", {}),
    ]
    for candidate in candidates:
        intrinsics = _intrinsics_from_mapping(candidate)
        if intrinsics is not None:
            return intrinsics
    return None


def _resolve_camera_intrinsics(data_dir, fx=None, fy=None, cx=None, cy=None):
    cli_values = (fx, fy, cx, cy)
    if any(value is not None for value in cli_values):
        if not all(value is not None for value in cli_values):
            raise ValueError("--fx, --fy, --cx, and --cy must be provided together")
        return tuple(float(value) for value in cli_values), "cli"

    metadata = _load_result_metadata(data_dir)
    intrinsics = _intrinsics_from_metadata(metadata)
    if intrinsics is not None:
        return intrinsics, "result.json"

    print(
        "[warn] Camera intrinsics not found in result.json; "
        f"using legacy defaults fx/fy/cx/cy={_LEGACY_INTRINSICS}. "
        "Provide a fixture with camera intrinsics or pass --fx --fy --cx --cy."
    )
    return _LEGACY_INTRINSICS, "legacy-default"


def _build_gripper_lines(transform, gripper_name):
    from grasp_gen.robot import load_control_points_for_visualization

    ctrl_pts_list = load_control_points_for_visualization(gripper_name)
    lines = []
    for ctrl_pts in ctrl_pts_list:
        pts = np.array(ctrl_pts, dtype=np.float32)
        pts_h = np.hstack([pts, np.ones([len(pts), 1])])
        transformed = (transform[:3, :] @ pts_h.T).T
        lines.append(transformed)
    return lines


def _save_json(candidates, elapsed, out_path, gripper_name, diagnostic):
    diagnostic_data = asdict(diagnostic)
    for key in (
        "object_pc_raw",
        "object_pc_after_completion",
        "object_pc_inference_input",
        "scene_pc_after_completion",
    ):
        value = diagnostic_data.pop(key, None)
        if value is not None:
            diagnostic_data[f"{key}_shape"] = list(value.shape)
            diagnostic_data[f"{key}_dtype"] = str(value.dtype)

    results = {
        "gripper": gripper_name,
        "inference_time_s": round(elapsed, 3),
        "num_grasps": len(candidates),
        "diagnostic": diagnostic_data,
        "grasps": [],
    }
    for i, g in enumerate(candidates):
        pos = g.pose_4x4[:3, 3].tolist()
        rot = g.pose_4x4[:3, :3].tolist()
        results["grasps"].append(
            {
                "index": i,
                "confidence": round(float(g.confidence), 6),
                "target_width_m": round(float(g.target_width_m), 6),
                "target_width_quality": round(float(g.target_width_quality), 3),
                "width_axis_camera": [round(float(v), 6) for v in g.width_axis_camera],
                "target_width_min_offset_m": round(float(g.target_width_min_offset_m), 6),
                "target_width_max_offset_m": round(float(g.target_width_max_offset_m), 6),
                "position_xyz": [round(v, 6) for v in pos],
                "rotation_3x3": [[round(v, 6) for v in row] for row in rot],
                "pose_4x4_rowmajor": g.pose_4x4.flatten().tolist(),
            }
        )
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[saved] {out_path}")


def _save_ply(pts, colors, out_path):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"[saved] {out_path}")


def _confidence_color(confidence):
    t = float(np.clip(confidence, 0.0, 1.0))
    return [1.0 - t, t, 0.0]


def _make_gripper_mesh_visual(gripper_mesh_template, transform, color):
    import open3d as o3d

    mesh = gripper_mesh_template.copy()
    mesh.apply_transform(transform)
    mesh_visual = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces),
    )
    mesh_visual.compute_vertex_normals()
    mesh_visual.paint_uniform_color(color)
    return mesh_visual


def _save_gripper_meshes(candidates, gripper_name, out_path, max_grasps=10):
    import open3d as o3d
    from grasp_gen.robot import get_gripper_info

    gripper_info = get_gripper_info(gripper_name)
    gripper_mesh_template = gripper_info.collision_mesh
    combined = o3d.geometry.TriangleMesh()
    count = min(len(candidates), max(0, int(max_grasps)))

    for g in candidates[:count]:
        mesh_visual = _make_gripper_mesh_visual(
            gripper_mesh_template,
            g.pose_4x4,
            _confidence_color(g.confidence),
        )
        combined += mesh_visual

    combined.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_path), combined)
    print(f"[saved] {out_path} ({count} gripper meshes)")


def _save_gripper_lines(candidates, gripper_name, out_path, max_grasps=10):
    import open3d as o3d

    points = []
    lines = []
    colors = []
    count = min(len(candidates), max(0, int(max_grasps)))

    for g in candidates[:count]:
        color = _confidence_color(g.confidence)
        for line_pts in _build_gripper_lines(g.pose_4x4, gripper_name):
            start = len(points)
            points.extend(np.asarray(line_pts, dtype=np.float32).tolist())
            lines.extend([[start + i, start + i + 1] for i in range(len(line_pts) - 1)])
            colors.extend([color] * (len(line_pts) - 1))

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_line_set(str(out_path), line_set)
    print(f"[saved] {out_path} ({count} gripper wireframes)")


def _visualize_open3d(
    pts,
    colors,
    candidates,
    gripper_name,
    out_dir,
    show=False,
    show_scores=False,
):
    import open3d as o3d
    from grasp_gen.robot import get_gripper_info

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    gripper_info = get_gripper_info(gripper_name)
    gripper_mesh_template = gripper_info.collision_mesh

    geometries = [pcd]
    max_show = min(len(candidates), 10)

    label_positions = []
    for i in range(max_show):
        g = candidates[i]
        mesh_visual = _make_gripper_mesh_visual(
            gripper_mesh_template,
            g.pose_4x4,
            _confidence_color(g.confidence),
        )
        geometries.append(mesh_visual)

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        coord.compute_vertex_normals()
        geometries.append(coord)

        if show_scores:
            pos = g.pose_4x4[:3, 3]
            label_positions.append((pos, i, g.confidence))

    if show:
        print("[vis] Opening Open3D viewer (close window to continue)...")
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("GraspGen", 1280, 720)
        vis.add_geometry("pointcloud", pcd)
        for i in range(max_show):
            g = candidates[i]
            gname = f"gripper_{i}"
            vis.add_geometry(
                gname,
                _make_gripper_mesh_visual(
                    gripper_mesh_template,
                    g.pose_4x4,
                    _confidence_color(g.confidence),
                ),
            )
            cname = f"coord_{i}"
            c = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            c.compute_vertex_normals()
            vis.add_geometry(cname, c)
        if show_scores:
            for pos, idx, conf in label_positions:
                vis.add_3d_label(pos, f"#{idx} {conf:.3f}")
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(visible=False, width=1280, height=720)
    for geom in geometries:
        vis2.add_geometry(geom)
    ctr = vis2.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat(pcd.get_center())
    ctr.set_zoom(0.5)
    vis2.poll_events()
    vis2.update_renderer()

    screenshot_path = out_dir / "grasp_preview.png"
    vis2.capture_screen_image(str(screenshot_path))
    print(f"[saved] {screenshot_path}")
    vis2.destroy_window()


def test_with_existing_data(
    data_dir: str,
    show: bool = False,
    show_scene: bool = False,
    show_scores: bool = False,
    collision_gripper: str | None = None,
    max_save_grasps: int = 10,
    num_grasps: int = 1500,
    topk_num_grasps: int = 200,
    min_grasps: int = 80,
    max_tries: int = 4,
    grasp_threshold: float = 0.5,
    collision_threshold: float = 0.005,
    depth_scale: float = 1000.0,
    enable_tabletop_filter: bool = True,
    enable_source_gripper_tabletop_sweep: bool = DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP,
    require_tabletop_filter: bool = True,
    tabletop_clearance: float = 0.003,
    tabletop_pregrasp_distance: float = 0.08,
    tabletop_pregrasp_steps: int = 5,
    tabletop_ransac_threshold: float = 0.006,
    tabletop_min_inlier_ratio: float = 0.15,
    tabletop_filter_mode: str = "strict",
    adaptive_tabletop_low_profile_height: float = 0.035,
    adaptive_tabletop_clearance_min: float = 0.001,
    adaptive_tabletop_clearance_max: float = 0.002,
    adaptive_tabletop_clearance_height_ratio: float = 0.12,
    adaptive_tabletop_pregrasp_min: float = 0.02,
    adaptive_tabletop_pregrasp_height_ratio: float = 2.0,
    adaptive_tabletop_hard_floor: float = -0.002,
    adaptive_tabletop_auto_tune: bool = True,
    adaptive_tabletop_retry_clearances: str = "0.002,0.001",
    adaptive_tabletop_retry_pregrasp_height_ratio: float = 1.5,
    adaptive_tabletop_retry_hard_floor: float = -0.003,
    enable_scene_cloud_table_holes: bool = False,
    scene_cloud_table_holes_max_points: int = 8000,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
):
    data_dir = Path(data_dir)
    depth_raw_path = data_dir / "depth_raw.npy"
    mask_path = None
    for p in data_dir.iterdir():
        if p.name.startswith("mask_") and p.suffix == ".png":
            mask_path = p
            break

    if not depth_raw_path.exists():
        print(f"ERROR: {depth_raw_path} not found")
        return False
    if mask_path is None:
        print(f"ERROR: no mask_*.png found in {data_dir}")
        return False

    depth_raw = np.load(str(depth_raw_path))
    print(f"Depth raw shape: {depth_raw.shape}, dtype: {depth_raw.dtype}")
    print(f"  min={depth_raw.min()}, max={depth_raw.max()}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    binary_mask = (mask > 0).astype(np.int32)
    print(f"Mask shape: {mask.shape}, unique: {np.unique(mask)}, object pixels: {binary_mask.sum()}")

    if binary_mask.sum() == 0:
        print("ERROR: mask is empty")
        return False

    (fx, fy, cx, cy), intrinsics_source = _resolve_camera_intrinsics(
        data_dir,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    print(f"Camera intrinsics ({intrinsics_source}): fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    print("\n--- Initializing GraspGenWrapper ---")
    _print_cuda_preflight()
    wrapper = GraspGenWrapper(
        device="cuda",
        collision_gripper=collision_gripper,
    )
    print(f"Model gripper: {wrapper.gripper_name}")
    if collision_gripper:
        print(f"Collision/viz gripper: {wrapper.collision_gripper_name}")

    print("\n--- Running GraspGen inference ---")
    t0 = time.time()

    candidates, diag = wrapper.plan_grasps(
        depth_image=depth_raw,
        segmentation_mask=binary_mask,
        camera_intrinsics=K,
        depth_scale=depth_scale,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
        min_grasps=min_grasps,
        max_tries=max_tries,
        enable_collision_filter=True,
        collision_threshold=collision_threshold,
        enable_tabletop_filter=enable_tabletop_filter,
        enable_source_gripper_tabletop_sweep=enable_source_gripper_tabletop_sweep,
        require_tabletop_filter=require_tabletop_filter,
        tabletop_clearance=tabletop_clearance,
        tabletop_pregrasp_distance=tabletop_pregrasp_distance,
        tabletop_pregrasp_steps=tabletop_pregrasp_steps,
        tabletop_ransac_threshold=tabletop_ransac_threshold,
        tabletop_min_inlier_ratio=tabletop_min_inlier_ratio,
        tabletop_filter_mode=tabletop_filter_mode,
        adaptive_tabletop_low_profile_height=adaptive_tabletop_low_profile_height,
        adaptive_tabletop_clearance_min=adaptive_tabletop_clearance_min,
        adaptive_tabletop_clearance_max=adaptive_tabletop_clearance_max,
        adaptive_tabletop_clearance_height_ratio=adaptive_tabletop_clearance_height_ratio,
        adaptive_tabletop_pregrasp_min=adaptive_tabletop_pregrasp_min,
        adaptive_tabletop_pregrasp_height_ratio=adaptive_tabletop_pregrasp_height_ratio,
        adaptive_tabletop_hard_floor=adaptive_tabletop_hard_floor,
        adaptive_tabletop_auto_tune=adaptive_tabletop_auto_tune,
        adaptive_tabletop_retry_clearances=_parse_float_csv(adaptive_tabletop_retry_clearances),
        adaptive_tabletop_retry_pregrasp_height_ratio=adaptive_tabletop_retry_pregrasp_height_ratio,
        adaptive_tabletop_retry_hard_floor=adaptive_tabletop_retry_hard_floor,
        enable_scene_cloud_table_holes=enable_scene_cloud_table_holes,
        scene_cloud_table_holes_max_points=scene_cloud_table_holes_max_points,
    )

    elapsed = time.time() - t0
    print(f"\n--- Results ({elapsed:.2f}s) ---")
    print(f"Total grasp candidates: {len(candidates)}")
    print(f"Diagnostic: stage={diag.failure_stage or 'ok'}, reason={diag.failure_reason or 'none'}")
    print(f"  object_points={diag.object_point_count}, scene_points={diag.scene_point_count}")
    print(
        "  mask="
        f"{diag.mask_pixel_count} px, valid_depth_in_mask="
        f"{diag.valid_depth_in_mask_count} ({diag.valid_depth_ratio_in_mask:.3f})"
    )
    print(
        f"  raw_grasps={diag.raw_grasp_count}, collision={diag.collision_filter_after}/{diag.collision_filter_before}"
    )
    print(
        f"  tabletop={diag.tabletop_filter_after}/{diag.tabletop_filter_before}, "
        f"plane_found={diag.tabletop_plane_found}, best_inlier={diag.tabletop_best_inlier_ratio:.3f}"
    )
    print(
        f"  tabletop_mode={diag.tabletop_filter_mode}, low_profile={diag.tabletop_low_profile}, "
        f"relaxed={diag.tabletop_relaxed}, object_height={diag.tabletop_object_height_m:.4f}m"
    )
    print(
        f"  tabletop_clearance_used={diag.tabletop_clearance_used_m:.4f}m, "
        f"pregrasp_used={diag.tabletop_pregrasp_distance_used_m:.4f}m, "
        f"best_candidate_clearance={diag.tabletop_best_candidate_clearance_m:.4f}m"
    )
    print(
        f"  tabletop_auto_tuned={diag.tabletop_auto_tuned}, attempts={diag.tabletop_auto_tune_attempts}, "
        f"reason={diag.tabletop_auto_tune_reason or 'none'}"
    )

    out_dir = data_dir / "graspgen_output"
    out_dir.mkdir(exist_ok=True)
    _save_json(candidates, elapsed, out_dir / "grasp_result.json", wrapper.gripper_name, diag)

    if len(candidates) == 0:
        print(f"WARNING: No grasps generated! Reason: {diag.failure_reason}")
        return False

    vis_gripper = wrapper.collision_gripper_name

    depth_m = depth_raw.astype(np.float64) / depth_scale
    scene_pts, scene_colors, object_pts, object_colors, all_pts, all_colors = _build_point_clouds(
        depth_m, binary_mask, fx, fy, cx, cy
    )
    print(f"Visualization point clouds: object={len(object_pts)}, scene={len(scene_pts)}, all={len(all_pts)}")

    for i, g in enumerate(candidates[:5]):
        pos = g.pose_4x4[:3, 3]
        print(
            f"  [{i}] conf={g.confidence:.4f} width={g.target_width_m:.4f} "
            f"width_q={g.target_width_quality:.3f} pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
        )

    _save_ply(object_pts, object_colors, out_dir / "object_cloud.ply")
    _save_ply(scene_pts, scene_colors, out_dir / "scene_cloud.ply")
    _save_ply(all_pts, all_colors, out_dir / "grasp_cloud.ply")
    _save_gripper_meshes(
        candidates,
        vis_gripper,
        out_dir / "grasp_grippers.ply",
        max_grasps=max_save_grasps,
    )
    _save_gripper_lines(
        candidates,
        vis_gripper,
        out_dir / "grasp_lines.ply",
        max_grasps=max_save_grasps,
    )

    vis_pts = all_pts if show_scene else object_pts
    vis_colors = all_colors if show_scene else object_colors

    try:
        _visualize_open3d(
            vis_pts,
            vis_colors,
            candidates,
            vis_gripper,
            out_dir,
            show=show,
            show_scores=show_scores,
        )
    except Exception as exc:
        print(f"[warn] Open3D visualization skipped: {exc}")

    print(f"\n=== TEST PASSED — output in {out_dir} ===")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GraspGen with save & visualization")
    parser.add_argument("--data-dir", default="outputs/grounded_sam2/20260601_195932_banana")
    parser.add_argument("--show", action="store_true", help="Open interactive Open3D viewer")
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Show confidence score labels on each grasp in the viewer",
    )
    parser.add_argument(
        "--show-scene",
        action="store_true",
        help="Show full scene point cloud instead of segmented object-only cloud",
    )
    parser.add_argument(
        "--collision-gripper",
        default=None,
        help="Gripper name for collision filter and visualization (default: same as model gripper).",
    )
    parser.add_argument(
        "--max-save-grasps",
        type=int,
        default=10,
        help="Maximum number of top-ranked grasps to save as 3D gripper geometry.",
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=1500,
        help="Number of grasp candidates to generate per iteration (default: 1500).",
    )
    parser.add_argument(
        "--topk-num-grasps",
        type=int,
        default=200,
        help="Top-K grasps to keep per iteration (default: 200).",
    )
    parser.add_argument(
        "--min-grasps",
        type=int,
        default=80,
        help="Minimum total grasps before stopping (default: 80).",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=4,
        help="Maximum inference iterations (default: 4).",
    )
    parser.add_argument(
        "--grasp-threshold",
        type=float,
        default=0.5,
        help="Discriminator confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=0.005,
        help="Collision distance threshold in meters (default: 0.005).",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Depth scale factor: raw_depth / scale = meters (default: 1000).",
    )
    parser.add_argument(
        "--enable-tabletop-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable tabletop plane signed-distance filter (default: True).",
    )
    parser.add_argument(
        "--require-tabletop-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return no grasps when tabletop filtering is enabled but no table plane is found (default: True).",
    )
    parser.add_argument(
        "--enable-source-gripper-tabletop-sweep",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP,
        help="Check every candidate with the source-gripper mesh after fitting the table (default: False).",
    )
    parser.add_argument(
        "--tabletop-clearance",
        type=float,
        default=0.003,
        help="Min signed distance from gripper mesh to table plane in meters (default: 0.003).",
    )
    parser.add_argument(
        "--tabletop-pregrasp-distance",
        type=float,
        default=0.08,
        help="Retreat distance along -grasp Z to sweep-check before final grasp (default: 0.08m).",
    )
    parser.add_argument(
        "--tabletop-pregrasp-steps",
        type=int,
        default=5,
        help="Number of intermediate pre-grasp sweep poses checked against the table (default: 5).",
    )
    parser.add_argument(
        "--tabletop-ransac-threshold",
        type=float,
        default=0.006,
        help="RANSAC inlier distance threshold for table plane fitting (default: 0.006).",
    )
    parser.add_argument(
        "--tabletop-min-inlier-ratio",
        type=float,
        default=0.15,
        help="Min inlier ratio to accept a fitted table plane (default: 0.15).",
    )
    parser.add_argument(
        "--tabletop-filter-mode",
        choices=("strict", "adaptive", "soft", "diagnostic"),
        default="strict",
        help=(
            "Tabletop filtering mode: strict keeps fixed thresholds; adaptive relaxes low-profile objects; "
            "soft also keeps near-safe fallback grasps; diagnostic records Robotiq tabletop clearance "
            "without filtering candidates."
        ),
    )
    parser.add_argument("--adaptive-tabletop-low-profile-height", type=float, default=0.035)
    parser.add_argument("--adaptive-tabletop-clearance-min", type=float, default=0.001)
    parser.add_argument("--adaptive-tabletop-clearance-max", type=float, default=0.002)
    parser.add_argument("--adaptive-tabletop-clearance-height-ratio", type=float, default=0.12)
    parser.add_argument("--adaptive-tabletop-pregrasp-min", type=float, default=0.02)
    parser.add_argument("--adaptive-tabletop-pregrasp-height-ratio", type=float, default=2.0)
    parser.add_argument("--adaptive-tabletop-hard-floor", type=float, default=-0.002)
    parser.add_argument(
        "--adaptive-tabletop-auto-tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry lower clearance/pre-grasp settings for low-profile adaptive tabletop failures (default: True).",
    )
    parser.add_argument(
        "--adaptive-tabletop-retry-clearances",
        default="0.002,0.001",
        help="Comma-separated adaptive retry clearances in meters (default: 0.002,0.001).",
    )
    parser.add_argument("--adaptive-tabletop-retry-pregrasp-height-ratio", type=float, default=1.5)
    parser.add_argument("--adaptive-tabletop-retry-hard-floor", type=float, default=-0.003)
    parser.add_argument(
        "--enable-scene-cloud-table-holes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable scene table dense local patch generation for offline collision/tabletop testing (default: False).",
    )
    parser.add_argument(
        "--scene-cloud-table-holes-max-points",
        type=int,
        default=8000,
        help="Maximum generated scene table patch points when enabled (default: 8000).",
    )
    parser.add_argument("--fx", type=float, default=None, help="Camera focal length fx")
    parser.add_argument("--fy", type=float, default=None, help="Camera focal length fy")
    parser.add_argument("--cx", type=float, default=None, help="Camera principal point cx")
    parser.add_argument("--cy", type=float, default=None, help="Camera principal point cy")
    args = parser.parse_args()

    success = test_with_existing_data(
        args.data_dir,
        show=args.show,
        show_scene=args.show_scene,
        show_scores=args.show_scores,
        collision_gripper=args.collision_gripper,
        max_save_grasps=args.max_save_grasps,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk_num_grasps,
        min_grasps=args.min_grasps,
        max_tries=args.max_tries,
        grasp_threshold=args.grasp_threshold,
        collision_threshold=args.collision_threshold,
        depth_scale=args.depth_scale,
        enable_tabletop_filter=args.enable_tabletop_filter,
        enable_source_gripper_tabletop_sweep=args.enable_source_gripper_tabletop_sweep,
        require_tabletop_filter=args.require_tabletop_filter,
        tabletop_clearance=args.tabletop_clearance,
        tabletop_pregrasp_distance=args.tabletop_pregrasp_distance,
        tabletop_pregrasp_steps=args.tabletop_pregrasp_steps,
        tabletop_ransac_threshold=args.tabletop_ransac_threshold,
        tabletop_min_inlier_ratio=args.tabletop_min_inlier_ratio,
        tabletop_filter_mode=args.tabletop_filter_mode,
        adaptive_tabletop_low_profile_height=args.adaptive_tabletop_low_profile_height,
        adaptive_tabletop_clearance_min=args.adaptive_tabletop_clearance_min,
        adaptive_tabletop_clearance_max=args.adaptive_tabletop_clearance_max,
        adaptive_tabletop_clearance_height_ratio=args.adaptive_tabletop_clearance_height_ratio,
        adaptive_tabletop_pregrasp_min=args.adaptive_tabletop_pregrasp_min,
        adaptive_tabletop_pregrasp_height_ratio=args.adaptive_tabletop_pregrasp_height_ratio,
        adaptive_tabletop_hard_floor=args.adaptive_tabletop_hard_floor,
        adaptive_tabletop_auto_tune=args.adaptive_tabletop_auto_tune,
        adaptive_tabletop_retry_clearances=args.adaptive_tabletop_retry_clearances,
        adaptive_tabletop_retry_pregrasp_height_ratio=args.adaptive_tabletop_retry_pregrasp_height_ratio,
        adaptive_tabletop_retry_hard_floor=args.adaptive_tabletop_retry_hard_floor,
        enable_scene_cloud_table_holes=args.enable_scene_cloud_table_holes,
        scene_cloud_table_holes_max_points=args.scene_cloud_table_holes_max_points,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
    )
    sys.exit(0 if success else 1)

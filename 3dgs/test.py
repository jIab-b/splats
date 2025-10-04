import argparse
import math
import os
from pprint import pformat
from typing import Dict, List

import torch

from .data import GardenDataset
from .train_local import (
    _ensure_dgr,
    build_K,
    init_gaussians_constant_density,
    render_dgr,
    save_png,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and differentiate constant-density Gaussian splats",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="./scenes/garden",
        help="Path to the scene directory.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional override for the images directory inside the scene.",
    )
    parser.add_argument(
        "--camera-idx",
        type=int,
        default=0,
        help="Camera index to render.",
    )
    parser.add_argument(
        "--approx-points",
        type=int,
        default=4096,
        help="Approximate Gaussian count to target when building the constant-density grid.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20000,
        help="Safety cap on the number of Gaussians to instantiate.",
    )
    parser.add_argument(
        "--pad-fraction",
        type=float,
        default=0.1,
        help="Relative padding applied to the camera-derived scene bounds.",
    )
    parser.add_argument(
        "--min-extent",
        type=float,
        default=1.0,
        help="Minimum per-axis extent enforced on the bounds to avoid degeneracy.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./out_local",
        help="Directory to store debug renders.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=2,
        help="Number of consecutive dataset camera views to render and differentiate.",
    )
    parser.add_argument(
        "--orbit-views",
        type=int,
        default=10,
        help="Number of synthetic spherical views to render around the Gaussian volume.",
    )
    return parser.parse_args()


def _assert_not_flat(pred: torch.Tensor) -> None:
    with torch.no_grad():
        p = pred.detach()
        vmin = float(p.min().item())
        vmax = float(p.max().item())
        vrange = vmax - vmin
        if vrange < 1e-3:
            raise RuntimeError(f"render produced flat image (range={vrange:.5f})")
        if vmax <= 0.05:
            raise RuntimeError(f"render too dark (max={vmax:.4f})")
        if vmin >= 0.95:
            raise RuntimeError(f"render too bright (min={vmin:.4f})")


def _grad_max(t: torch.Tensor) -> float:
    g = t.grad
    if g is None:
        return 0.0
    return float(g.detach().abs().max().item())


def _format_bounds(bounds: Dict[str, object]) -> Dict[str, object]:
    formatted: Dict[str, object] = {}
    for k, v in bounds.items():
        if hasattr(v, "tolist"):
            formatted[k] = v.tolist()
        else:
            formatted[k] = v
    return formatted


def _build_orbit_cameras(
    meta: Dict[str, object],
    base_cam: Dict[str, torch.Tensor],
    num_views: int,
    device: torch.device,
    elevation_deg: float = 25.0,
    radius_scale: float = 1.2,
) -> List[Dict[str, torch.Tensor]]:
    if num_views <= 0:
        return []
    bounds = meta["bounds"]
    center = torch.tensor(bounds["center"], device=device, dtype=torch.float32)
    radius_base = float(meta.get("extent_scalar", bounds.get("radius", 1.0)))
    radius = max(radius_base * radius_scale, 0.25)
    up_world = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    elev = math.radians(elevation_deg)
    fx, fy = base_cam["fx"], base_cam["fy"]
    cx, cy = base_cam["cx"], base_cam["cy"]
    H, W = base_cam["H"], base_cam["W"]
    cameras: List[Dict[str, torch.Tensor]] = []
    for i in range(num_views):
        az = 2.0 * math.pi * (i / num_views)
        dir_vec = torch.tensor(
            [
                math.cos(az) * math.cos(elev),
                math.sin(elev),
                math.sin(az) * math.cos(elev),
            ],
            device=device,
            dtype=torch.float32,
        )
        eye = center + radius * dir_vec
        forward = center - eye
        forward = forward / torch.clamp(forward.norm(), min=1e-6)
        right = torch.cross(forward, up_world, dim=0)
        if right.norm() < 1e-6:
            alt_up = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
            right = torch.cross(forward, alt_up, dim=0)
        right = right / torch.clamp(right.norm(), min=1e-6)
        true_up = torch.cross(right, forward, dim=0)
        true_up = true_up / torch.clamp(true_up.norm(), min=1e-6)
        c2w = torch.eye(4, device=device, dtype=torch.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = true_up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = eye
        cameras.append(
            {
                "H": H,
                "W": W,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "c2w": c2w,
            }
        )
    return cameras


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for diff_gaussian_rasterization tests")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    dgr = _ensure_dgr()
    print("diff_gaussian_rasterization module loaded")

    dataset = GardenDataset(args.scene, images_dir=args.images_dir)
    print(f"Loaded dataset '{dataset.scene_dir}' with {len(dataset)} frames")

    means, scales, quats, opacity, colors, meta = init_gaussians_constant_density(
        scene=dataset.scene_dir,
        device=device,
        approx_points=args.approx_points,
        pad_fraction=args.pad_fraction,
        min_extent=args.min_extent,
        max_points=args.max_points,
        images_dir=args.images_dir,
        dataset=dataset,
    )

    bounds_for_log = _format_bounds(meta["bounds"])
    meta_log = {**meta, "bounds": bounds_for_log}
    print("Initialisation meta:")
    print(pformat(meta_log))

    num_views = max(1, int(args.num_views))
    view_indices: List[int] = [ (args.camera_idx + i) % len(dataset) for i in range(num_views) ]
    print(f"Rendering views: {view_indices}")

    base_sample = dataset[view_indices[0]]
    base_cam = base_sample["camera"]

    losses = []
    render_paths: List[str] = []
    target_paths: List[str] = []
    os.makedirs(args.out_dir, exist_ok=True)

    for offset, view_idx in enumerate(view_indices):
        sample = dataset[view_idx]
        cam = sample["camera"]
        K = build_K(cam, device)
        c2w = cam["c2w"].to(device)
        target = sample["image"].to(device)

        pred = render_dgr(
            dgr,
            means,
            scales,
            quats,
            opacity,
            colors,
            K,
            c2w,
            cam["W"],
            cam["H"],
        )

        _assert_not_flat(pred)

        stats = pred.detach()
        print(
            f"View {view_idx}: mean={stats.mean().item():.3f}, min={stats.min().item():.3f}, max={stats.max().item():.3f}"
        )

        loss = torch.mean((pred - target) ** 2)
        losses.append(loss)

        render_path = os.path.join(args.out_dir, f"gaussians_constant_density_view{view_idx:03d}.png")
        target_path = os.path.join(args.out_dir, f"target_frame_{view_idx:03d}.png")
        save_png(pred, render_path)
        save_png(target, target_path)
        render_paths.append(render_path)
        target_paths.append(target_path)

    total_loss = torch.stack(losses).mean()
    for tensor in (means, scales, quats, opacity, colors):
        if tensor.grad is not None:
            tensor.grad.zero_()
    total_loss.backward()
    print(f"Mean loss over {len(losses)} views: {total_loss.item():.6f}")

    grad_stats = {
        "means_max": _grad_max(means),
        "scales_max": _grad_max(scales),
        "quats_max": _grad_max(quats),
        "opacity_max": _grad_max(opacity),
        "colors_max": _grad_max(colors),
    }
    print("Gradient max abs per parameter:")
    print(pformat(grad_stats))
    if not any(v > 0.0 for v in grad_stats.values()):
        raise RuntimeError("Backpropagation produced zero gradients")

    print("Saved renders:")
    for rp, tp in zip(render_paths, target_paths):
        print(f"  pred -> {rp}")
        print(f"  target -> {tp}")

    if args.orbit_views > 0:
        orbit_cams = _build_orbit_cameras(meta, base_cam, args.orbit_views, device)
        orbit_paths: List[str] = []
        for idx, orbit_cam in enumerate(orbit_cams):
            K_orbit = build_K(orbit_cam, device)
            pred_orbit = render_dgr(
                dgr,
                means,
                scales,
                quats,
                opacity,
                colors,
                K_orbit,
                orbit_cam["c2w"],
                orbit_cam["W"],
                orbit_cam["H"],
            )
            _assert_not_flat(pred_orbit)
            orbit_path = os.path.join(args.out_dir, f"gaussians_orbit_view{idx:03d}.png")
            save_png(pred_orbit, orbit_path)
            orbit_paths.append(orbit_path)
            stats_orbit = pred_orbit.detach()
            print(
                f"Orbit {idx}: mean={stats_orbit.mean().item():.3f}, min={stats_orbit.min().item():.3f}, max={stats_orbit.max().item():.3f}"
            )
        print("Saved orbit renders:")
        for path in orbit_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()

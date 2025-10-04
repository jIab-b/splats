import os, sys, glob, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from PIL import Image
from .data import GardenDataset

def _ensure_dgr():
    try:
        import diff_gaussian_rasterization as dgr
        return dgr
    except Exception:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cand = glob.glob(os.path.join(root, "diff-gaussian-rasterization", "build", "lib.*"))
        for p in cand:
            if p not in sys.path:
                sys.path.insert(0, p)
        import diff_gaussian_rasterization as dgr2
        return dgr2

def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-8)

def build_K(cam, device):
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    return torch.tensor([[fx,0.0,cx],[0.0,fy,cy],[0.0,0.0,1.0]], dtype=torch.float32, device=device)

def build_view_proj(K: torch.Tensor, c2w: torch.Tensor, W: int, H: int):
    w2c = torch.linalg.inv(c2w)
    view = w2c.clone()
    view[:3, 1:3] *= -1
    view[2, :] *= -1
    near = 0.01
    far = 100.0
    proj = torch.zeros(4, 4, device=c2w.device, dtype=c2w.dtype)
    proj[0, 0] = 2.0 * K[0, 0] / float(W)
    proj[1, 1] = 2.0 * K[1, 1] / float(H)
    proj[0, 2] = 1.0 - 2.0 * K[0, 2] / float(W)
    proj[1, 2] = 2.0 * K[1, 2] / float(H) - 1.0
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2.0 * far * near) / (far - near)
    proj[3, 2] = -1.0
    proj_view = proj @ view
    return view.transpose(0, 1).contiguous(), proj_view.transpose(0, 1).contiguous()

def init_gaussians(count: int, radius: float, device: torch.device):
    means = torch.empty(count, 3, device=device).uniform_(-1.0, 1.0)
    means = torch.nn.functional.normalize(means, dim=-1) * radius
    scales = torch.full((count, 3), 0.02, device=device, requires_grad=True)
    quats = torch.zeros(count, 4, device=device, requires_grad=True)
    with torch.no_grad():
        quats[:, 0] = 1.0
    opacity = torch.full((count, 1), 0.8, device=device, requires_grad=True)
    colors = torch.full((count, 3), 0.5, device=device, requires_grad=True)
    means.requires_grad_(True)
    return means, scales, quats, opacity, colors


def init_gaussians_constant_density(
    scene: str,
    device: torch.device,
    approx_points: int = 4096,
    pad_fraction: float = 0.1,
    min_extent: float = 1.0,
    max_points: int = 20000,
    images_dir: str | None = None,
    dataset: GardenDataset | None = None,
    extent_scale: float = 0.4,
):
    ds = dataset if dataset is not None else GardenDataset(scene, images_dir=images_dir)
    bounds = ds.scene_bounds(pad_fraction=pad_fraction, min_extent=min_extent)
    center_np = bounds["center"]
    extent_scalar = float(bounds["radius"]) * float(extent_scale)
    extent_scalar = float(np.clip(extent_scalar, 0.25, 1.5))
    extent_np = np.full(3, extent_scalar, dtype=np.float32)
    min_corner_np = center_np - extent_np
    max_corner_np = center_np + extent_np
    min_corner = torch.from_numpy(min_corner_np.astype(np.float32)).to(device=device)
    max_corner = torch.from_numpy(max_corner_np.astype(np.float32)).to(device=device)
    extent = max_corner - min_corner
    volume = float((extent[0] * extent[1] * extent[2]).item())
    approx_points = max(1, int(approx_points))
    spacing = float(volume / approx_points) ** (1.0 / 3.0) if approx_points > 0 else 1.0
    spacing = max(spacing, 1e-2)
    counts = torch.ceil(extent / spacing).to(torch.int64) + 1
    counts = torch.clamp(counts, min=1)
    total = int(torch.prod(counts).item())
    attempts = 0
    while total > max_points and attempts < 64:
        spacing *= 1.25
        counts = torch.ceil(extent / spacing).to(torch.int64) + 1
        counts = torch.clamp(counts, min=1)
        total = int(torch.prod(counts).item())
        attempts += 1
    if total > max_points:
        scale = (total / max_points) ** (1.0 / 3.0)
        counts = torch.clamp((counts.to(torch.float32) / scale).floor().to(torch.int64), min=1)
        total = int(torch.prod(counts).item())
    xs = torch.linspace(float(min_corner[0]), float(max_corner[0]), int(counts[0].item()), device=device)
    ys = torch.linspace(float(min_corner[1]), float(max_corner[1]), int(counts[1].item()), device=device)
    zs = torch.linspace(float(min_corner[2]), float(max_corner[2]), int(counts[2].item()), device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
    grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)
    means = grid.reshape(-1, 3)
    jitter = (torch.rand_like(means) - 0.5) * torch.mean(extent) * 0.02
    means = (means + jitter).clone().detach().requires_grad_(True)
    counts_f = counts.to(dtype=extent.dtype)
    spacing_axis = torch.where(
        counts > 1,
        extent / torch.clamp(counts_f - 1.0, min=1e-3),
        extent.max().expand_as(extent),
    )
    spacing_axis = torch.clamp(spacing_axis, min=1e-2)
    base_scale = float(spacing_axis.mean().item()) * 0.15
    base_scale = float(np.clip(base_scale, 0.02, 0.1))
    scales = torch.full((total, 3), base_scale, device=device, requires_grad=True)
    quats = torch.zeros(total, 4, device=device, requires_grad=True)
    with torch.no_grad():
        quats[:, 0] = 1.0
    opacity = torch.full((total, 1), 0.8, device=device, requires_grad=True)
    positions_norm = (means.detach() - min_corner) / torch.clamp(extent, min=1e-3)
    positions_norm = torch.where(torch.isfinite(positions_norm), positions_norm, torch.zeros_like(positions_norm))
    phase = torch.tensor([0.0, 2.0943951, 4.1887902], device=device)
    freq = torch.tensor([6.2831853, 7.8539816, 9.4247779], device=device)
    mixing = torch.tensor(
        [
            [1.0, 0.5, 0.25],
            [0.3, 1.2, 0.6],
            [0.4, 0.7, 1.4],
        ],
        device=device,
        dtype=positions_norm.dtype,
    )
    waves = positions_norm @ mixing.T
    colors = 0.5 + 0.5 * torch.sin(waves * freq[None, :] + phase[None, :])
    colors = torch.clamp(colors, 0.0, 1.0).contiguous().detach().requires_grad_(True)
    meta = {
        "bounds": bounds,
        "grid_counts": tuple(int(c.item()) for c in counts),
        "approx_points": approx_points,
        "total_points": total,
        "spacing": float(spacing),
        "scene_dir": ds.scene_dir,
        "dataset_len": len(ds),
        "extent_scalar": extent_scalar,
    }
    return means, scales, quats, opacity, colors, meta

def init_gaussians_axis(scene: str, count: int, device: torch.device, depth: float = 4.0, jitter_xy: float = 0.2, images_dir: str | None = None):
    ds = GardenDataset(scene, images_dir=images_dir)
    s0 = ds[0]
    c2w = s0["camera"]["c2w"].to(device)
    cam_pos = c2w[:3, 3]
    forward = -c2w[:3, 2]
    forward = forward / torch.clamp(forward.norm() + 1e-8, min=1e-8)
    means = cam_pos[None, :].expand(count, 3).clone()
    t = depth + torch.randn(count, device=device) * 0.5
    means = means + forward[None, :] * t[:, None]
    up_guess = torch.tensor([0.0, 1.0, 0.0], device=device)
    side = torch.cross(forward, up_guess)
    if side.norm() < 1e-6:
        up_guess = torch.tensor([1.0, 0.0, 0.0], device=device)
        side = torch.cross(forward, up_guess)
    side = side / torch.clamp(side.norm() + 1e-8, min=1e-8)
    up = torch.cross(side, forward)
    up = up / torch.clamp(up.norm() + 1e-8, min=1e-8)
    jitter = (torch.randn(count, 2, device=device) * jitter_xy)
    means = means + side[None, :] * jitter[:, :1] + up[None, :] * jitter[:, 1:2]
    scales = torch.full((count, 3), 0.02, device=device, requires_grad=True)
    quats = torch.zeros(count, 4, device=device, requires_grad=True)
    with torch.no_grad():
        quats[:, 0] = 1.0
    opacity = torch.full((count, 1), 0.8, device=device, requires_grad=True)
    colors = torch.full((count, 3), 0.5, device=device, requires_grad=True)
    means.requires_grad_(True)
    return means, scales, quats, opacity, colors

def frustum_coverage_fraction(means: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor, W: int, H: int) -> float:
    view_t, proj_view_t = build_view_proj(K, c2w, W, H)
    view = view_t.transpose(0, 1).contiguous()
    proj_view = proj_view_t.transpose(0, 1).contiguous()
    N = means.shape[0]
    homo = torch.cat([means, torch.ones(N, 1, device=means.device, dtype=means.dtype)], dim=1)
    clip = (proj_view @ (view @ homo.T)).T
    w = clip[:, 3:4]
    ndc = clip[:, :3] / torch.clamp(w, min=1e-8)
    in_ndc = (ndc[:, 0].abs() <= 1.0) & (ndc[:, 1].abs() <= 1.0) & (ndc[:, 2] >= 0.0) & (ndc[:, 2] <= 1.0) & (w.squeeze(-1) > 0)
    return float(in_ndc.float().mean().item())

def clamp_params(scales: torch.Tensor, opacity: torch.Tensor):
    with torch.no_grad():
        scales.clamp_(min=1e-4, max=0.2)
        opacity.clamp_(min=0.01, max=1.0)

def render_dgr(dgr, means, scales, quats, opacity, colors, K, c2w, W, H):
    qn = quat_normalize(quats)
    view_t, proj_view_t = build_view_proj(K, c2w, W, H)
    campos = c2w[:3, 3].detach().clone()
    campos[1:3] *= -1
    tanfovx = float(W) / (2.0 * float(K[0, 0]))
    tanfovy = float(H) / (2.0 * float(K[1, 1]))
    settings = dgr.GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, device=means.device, dtype=means.dtype),  # Temp black bg for debug - revert to ones for training
        scale_modifier=1.0,
        viewmatrix=view_t,
        projmatrix=proj_view_t,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
    )
    rast = dgr.GaussianRasterizer(settings)
    means2D = torch.zeros(means.shape[0], 3, device=means.device, dtype=means.dtype)
    out, _ = rast(means, means2D, opacity, colors_precomp=colors, scales=scales, rotations=qn, cov3D_precomp=None)
    if out.dim() == 3 and out.shape[0] == 3:
        return out
    return out.permute(2, 0, 1)

def l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)

def save_png(t: torch.Tensor, path: str):
    td = t.detach()
    if td.dim() == 3 and td.shape[0] == 3:
        img = torch.clamp(td.permute(1, 2, 0), 0, 1).cpu().numpy()
    else:
        img = torch.clamp(td, 0, 1).cpu().numpy()
    Image.fromarray((img * 255.0).astype(np.uint8)).save(path)

def sanity_render(scene: str, idx: int, means, scales, quats, opacity, colors, device: torch.device, out_dir: str, images_dir: str | None = None):
    ds = GardenDataset(scene, images_dir=images_dir)
    s = ds[idx % len(ds)]
    img = s["image"].to(device)
    cam = s["camera"]
    K = build_K(cam, device)
    dgr = _ensure_dgr()
    pred = render_dgr(dgr, means, scales, quats, opacity, colors, K, cam["c2w"].to(device), cam["W"], cam["H"])
    os.makedirs(out_dir, exist_ok=True)
    save_png(img, os.path.join(out_dir, f"gt_{idx:04d}.png"))
    save_png(pred, os.path.join(out_dir, f"pred_{idx:04d}.png"))

def sanity_intrinsics(scene: str, images_dir: str | None = None):
    ds = GardenDataset(scene, images_dir=images_dir)
    s = ds[0]
    H, W = s["camera"]["H"], s["camera"]["W"]
    fx, fy = s["camera"]["fx"], s["camera"]["fy"]
    tanx = W / (2.0 * fx)
    tany = H / (2.0 * fy)
    print({"H": H, "W": W, "fx": fx, "fy": fy, "tanfovx": tanx, "tanfovy": tany})

def sanity_check_all(scene: str, out_dir: str, device: torch.device, count: int = 1024, idx0: int = 0, idx1: int = 1, images_dir: str | None = None):
    os.makedirs(os.path.join(out_dir, "sanity_full"), exist_ok=True)
    dgr = _ensure_dgr()
    print({"dgr_module": getattr(dgr, "__file__", str(dgr))})
    ds = GardenDataset(scene, images_dir=images_dir)
    s0 = ds[idx0 % len(ds)]
    s1 = ds[idx1 % len(ds)]
    means, scales, quats, opacity, colors = init_gaussians(count, 1.0, device)
    cam0 = s0["camera"]
    cam1 = s1["camera"]
    K0 = build_K(cam0, device)
    K1 = build_K(cam1, device)
    pred0 = render_dgr(dgr, means, scales, quats, opacity, colors, K0, cam0["c2w"].to(device), cam0["W"], cam0["H"]).detach()
    save_png(s0["image"].to(device), os.path.join(out_dir, "sanity_full", "gt0.png"))
    save_png(pred0, os.path.join(out_dir, "sanity_full", "pred0.png"))
    with torch.no_grad():
        means_shift = means.clone()
        means_shift[: min(8, count), 0] += 0.05
    pred_shift = render_dgr(dgr, means_shift, scales, quats, opacity, colors, K0, cam0["c2w"].to(device), cam0["W"], cam0["H"]).detach()
    save_png(pred_shift, os.path.join(out_dir, "sanity_full", "pred0_shiftx.png"))
    diff = torch.mean(torch.abs(pred_shift - pred0)).item()
    print({"shift_sensitivity_mean_abs": diff})
    pred1 = render_dgr(dgr, means, scales, quats, opacity, colors, K1, cam1["c2w"].to(device), cam1["W"], cam1["H"]).detach()
    save_png(s1["image"].to(device), os.path.join(out_dir, "sanity_full", "gt1.png"))
    save_png(pred1, os.path.join(out_dir, "sanity_full", "pred1.png"))
    means_g = means.clone().detach().requires_grad_(True)
    scales_g = scales.clone().detach().requires_grad_(True)
    quats_g = quats.clone().detach().requires_grad_(True)
    opacity_g = opacity.clone().detach().requires_grad_(True)
    colors_g = colors.clone().detach().requires_grad_(True)
    pred_g = render_dgr(dgr, means_g, scales_g, quats_g, opacity_g, colors_g, K0, cam0["c2w"].to(device), cam0["W"], cam0["H"]) 
    loss_g = torch.mean((pred_g - s0["image"].to(device)) ** 2)
    loss_g.backward()
    gstats = {
        "grad_means_max": float(means_g.grad.abs().max().item()) if means_g.grad is not None else 0.0,
        "grad_scales_max": float(scales_g.grad.abs().max().item()) if scales_g.grad is not None else 0.0,
        "grad_quats_max": float(quats_g.grad.abs().max().item()) if quats_g.grad is not None else 0.0,
        "grad_opacity_max": float(opacity_g.grad.abs().max().item()) if opacity_g.grad is not None else 0.0,
        "grad_colors_max": float(colors_g.grad.abs().max().item()) if colors_g.grad is not None else 0.0,
    }
    print(gstats)

def dump_images_test(scene: str, out_dir: str, num: int = 8, device: torch.device | None = None, images_dir: str | None = None):
    dd = os.path.join(out_dir, "images_test")
    os.makedirs(dd, exist_ok=True)
    ds = GardenDataset(scene, images_dir=images_dir)
    k = min(num, len(ds))
    for i in range(k):
        s = ds[i]
        t = s["image"] if device is None else s["image"].to(device)
        save_png(t, os.path.join(dd, f"img_{i:04d}.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--count", type=int, default=50000)
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--lr_pos", type=float, default=1e-2)
    ap.add_argument("--lr_other", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="./out_local")
    ap.add_argument("--sanity_every", type=int, default=200)
    ap.add_argument("--sanity_full", action="store_true")
    ap.add_argument("--dump_test_images", type=int, default=0)
    ap.add_argument("--dump_images_test", action="store_true")
    ap.add_argument("--dump_images_n", type=int, default=8)
    ap.add_argument("--use_axis_init", action="store_true")
    ap.add_argument("--axis_init", action="store_true")
    ap.add_argument("--images_dir", type=str, default="images_8", help="subdirectory with training images or 'auto'")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = None if args.images_dir == "auto" else args.images_dir
    if args.sanity_full:
        sanity_check_all(args.scene, args.out, device, count=min(4096, args.count), images_dir=images_dir)
    ds = GardenDataset(args.scene, images_dir=images_dir)
    if args.dump_images_test:
        dump_images_test(args.scene, args.out, num=args.dump_images_n, device=device, images_dir=images_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    if args.axis_init:
        means, scales, quats, opacity, colors = init_gaussians_axis(args.scene, args.count, device, images_dir=images_dir)
    else:
        means, scales, quats, opacity, colors = init_gaussians(args.count, args.radius, device)
    opt = Adam([
        {"params": [means], "lr": args.lr_pos},
        {"params": [scales, quats, opacity, colors], "lr": args.lr_other},
    ])
    dgr = _ensure_dgr()
    step = 0
    sanity_intrinsics(args.scene, images_dir=images_dir)
    os.makedirs(args.out, exist_ok=True)
    s0 = ds[0]
    K0 = build_K(s0["camera"], device)
    frac0 = frustum_coverage_fraction(means.detach(), K0, s0["camera"]["c2w"].to(device), s0["camera"]["W"], s0["camera"]["H"])
    print({"frustum_coverage_start": frac0})
    while step < args.iters:
        for batch in dl:
            img = batch["image"].to(device).squeeze(0)
            cam = {
                "H": int(batch["camera"]["H"]),
                "W": int(batch["camera"]["W"]),
                "fx": float(batch["camera"]["fx"]),
                "fy": float(batch["camera"]["fy"]),
                "cx": float(batch["camera"]["cx"]),
                "cy": float(batch["camera"]["cy"]),
                "c2w": batch["camera"]["c2w"].to(device)[0] if batch["camera"]["c2w"].dim()==3 else batch["camera"]["c2w"].to(device),
            }
            K = build_K(cam, device)
            pred = render_dgr(dgr, means, scales, quats, opacity, colors, K, cam["c2w"], cam["W"], cam["H"])
            loss = l2(pred, img) + 1e-3 * (scales ** 2).mean() + 1e-3 * (opacity.clamp(0,1) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            clamp_params(scales, opacity)
            opt.step()
            step += 1
            if step % 10 == 0:
                print(step, float(loss.detach().cpu()))
            if step % args.sanity_every == 0:
                sanity_render(args.scene, step % len(ds), means, scales, quats, opacity, colors, device, args.out, images_dir=images_dir)
                frac = frustum_coverage_fraction(means.detach(), K, cam["c2w"], cam["W"], cam["H"])
                print({"frustum_coverage": frac})
            if step >= args.iters:
                break
    sanity_render(args.scene, 0, means, scales, quats, opacity, colors, device, args.out, images_dir=images_dir)

if __name__ == "__main__":
    main()

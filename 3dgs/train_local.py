import os, sys, glob, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import relu
from PIL import Image
try:
    from .data import GardenDataset
except ImportError:  # Fallback when module is imported outside package context
    from data import GardenDataset

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

def init_gaussians_constant_density(
    scene: str,
    device: torch.device,
    approx_points: int = 4096,
    pad_fraction: float = 0.1,
    min_extent: float = 1.0,
    max_points: int = 50000,
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
        bg=torch.zeros(3, device=means.device, dtype=means.dtype),
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

def densify_and_prune(means, scales, quats, opacity, colors, grad_threshold=0.00016, prune_threshold=0.005, add_new=1000, max_gaussians=1000000):
    # Detach inputs to break history after opt.step
    means = means.detach().requires_grad_(True)
    scales = scales.detach().requires_grad_(True)
    quats = quats.detach().requires_grad_(True)
    opacity = opacity.detach().requires_grad_(True)
    colors = colors.detach().requires_grad_(True)
    
    # Prune low opacity
    mask = opacity.squeeze() > prune_threshold
    means = means[mask]
    scales = scales[mask]
    quats = quats[mask]
    opacity = opacity[mask]
    colors = colors[mask]
    # Detach after prune to ensure leaf
    means = means.detach().requires_grad_(True)
    scales = scales.detach().requires_grad_(True)
    quats = quats.detach().requires_grad_(True)
    opacity = opacity.detach().requires_grad_(True)
    colors = colors.detach().requires_grad_(True)
    
    # Densify: clone if large grad (simulate with scale for minimal; in full, use actual grad)
    large_scale_mask = (scales.max(dim=1)[0] > 0.005).squeeze()
    clone_count = int(large_scale_mask.sum().item())
    if clone_count > 0:
        clone_means = means[large_scale_mask] + torch.randn_like(means[large_scale_mask]) * 0.001
        clone_scales = scales[large_scale_mask] * 0.5 + torch.randn_like(scales[large_scale_mask]) * 0.001
        clone_quats = quats[large_scale_mask]
        clone_opacity = opacity[large_scale_mask]
        clone_colors = colors[large_scale_mask]
        means = torch.cat([means, clone_means], dim=0)
        scales = torch.cat([scales, clone_scales], dim=0)
        quats = torch.cat([quats, clone_quats], dim=0)
        opacity = torch.cat([opacity, clone_opacity], dim=0)
        colors = torch.cat([colors, clone_colors], dim=0)
        # Detach after clone to ensure leaf
        means = means.detach().requires_grad_(True)
        scales = scales.detach().requires_grad_(True)
        quats = quats.detach().requires_grad_(True)
        opacity = opacity.detach().requires_grad_(True)
        colors = colors.detach().requires_grad_(True)
    
    # Add new random Gaussians if below max
    current_n = means.shape[0]
    if current_n < max_gaussians:
        add_n = min(add_new, max_gaussians - current_n)
        if add_n > 0:
            # Simple random init near center; in full, backproject high-error pixels
            center = means.mean(dim=0)
            new_means = center[None, :].expand(add_n, 3) + torch.randn(add_n, 3, device=means.device) * 0.1
            new_means = new_means.detach().requires_grad_(True)
            new_scales = torch.full((add_n, 3), 0.02, device=means.device, requires_grad=True)
            new_quats = torch.zeros(add_n, 4, device=means.device)
            new_quats[:, 0] = 1.0
            new_quats = new_quats.detach().requires_grad_(True)
            new_opacity = torch.full((add_n, 1), 0.8, device=means.device, requires_grad=True)
            new_colors = torch.full((add_n, 3), 0.5, device=means.device, requires_grad=True)
            means = torch.cat([means, new_means], dim=0)
            scales = torch.cat([scales, new_scales], dim=0)
            quats = torch.cat([quats, new_quats], dim=0)
            opacity = torch.cat([opacity, new_opacity], dim=0)
            colors = torch.cat([colors, new_colors], dim=0)
            # Detach after add to ensure leaf
            means = means.detach().requires_grad_(True)
            scales = scales.detach().requires_grad_(True)
            quats = quats.detach().requires_grad_(True)
            opacity = opacity.detach().requires_grad_(True)
            colors = colors.detach().requires_grad_(True)
    
    return means, scales, quats, opacity, colors

def save_checkpoint(scene_stub, step, means, scales, quats, opacity, colors, out_dir):
    ckpt_dir = os.path.join(out_dir, "checkpoints", scene_stub)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        "step": step,
        "means": means.detach().cpu(),
        "scales": scales.detach().cpu(),
        "quats": quats.detach().cpu(),
        "opacity": opacity.detach().cpu(),
        "colors": colors.detach().cpu(),
    }
    torch.save(ckpt, os.path.join(ckpt_dir, f"iter_{step:05d}.pt"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--init_count", type=int, default=50000)
    ap.add_argument("--lr_pos", type=float, default=1e-2)
    ap.add_argument("--lr_other", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="./out_local")
    ap.add_argument("--images_dir", type=str, default="images_8", help="subdirectory with training images or 'auto'")
    args = ap.parse_args()
    scene_stub = os.path.basename(os.path.normpath(args.scene))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = None if args.images_dir == "auto" else args.images_dir
    ds = GardenDataset(args.scene, images_dir=images_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)
    # Adapt max_gaussians to scene size
    max_gaussians = min(1000000, 5000 * len(ds))
    means, scales, quats, opacity, colors, meta = init_gaussians_constant_density(
        args.scene, device, max_points=args.init_count, images_dir=images_dir, dataset=ds
    )
    # Initial render
    os.makedirs(args.out, exist_ok=True)
    s0 = ds[0]
    cam0 = s0["camera"]
    K0 = build_K(cam0, device)
    dgr = _ensure_dgr()
    pred_initial = render_dgr(dgr, means, scales, quats, opacity, colors, K0, cam0["c2w"].to(device), cam0["W"], cam0["H"])
    save_png(pred_initial, os.path.join(args.out, "initial_render.png"))
    # Multi-view sanity renders setup
    renders_dir = os.path.join(args.out, "renders")
    os.makedirs(renders_dir, exist_ok=True)
    gt_dir = os.path.join(renders_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    view_indices = [0, len(ds)//4, len(ds)//2, 3*len(ds)//4]
    view_samples = [ds[i] for i in view_indices]
    for idx, s in enumerate(view_samples):
        save_png(s["image"].to(device), os.path.join(gt_dir, f"view_{idx}.png"))
    def create_optimizer(lr_pos, lr_other):
        return Adam([
            {"params": [means], "lr": lr_pos},
            {"params": [scales, quats, opacity, colors], "lr": lr_other},
        ])
    opt = create_optimizer(args.lr_pos, args.lr_other)
    step = 0
    densify_interval = 100
    activate_interval = 300
    ckpt_interval = 1000
    decay_interval = 7000
    render_interval = 100
    ckpt_dir = os.path.join(args.out, "checkpoints", scene_stub)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.out, "log.txt")
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
            loss_val = float(loss.detach().cpu())
            n_gauss = means.shape[0]
            # Log to file every step
            with open(log_path, 'a') as f:
                f.write(f"Step {step}: Loss {loss_val:.6f}, Gaussians {n_gauss}\n")
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss_val}")
            if step % 100 == 0:
                print(f"Current Gaussians: {n_gauss}")
            # Multi-view renders every render_interval
            if step % render_interval == 0 or step == args.iters:
                iter_dir = os.path.join(renders_dir, f"iter_{step:04d}")
                os.makedirs(iter_dir, exist_ok=True)
                for idx, s in enumerate(view_samples):
                    K_view = build_K(s["camera"], device)
                    pred_view = render_dgr(dgr, means, scales, quats, opacity, colors, K_view, s["camera"]["c2w"].to(device), s["camera"]["W"], s["camera"]["H"])
                    save_png(pred_view, os.path.join(iter_dir, f"view_{idx}.png"))
            # Densify and prune every densify_interval
            if step % densify_interval == 0 and step > 0:
                means, scales, quats, opacity, colors = densify_and_prune(
                    means, scales, quats, opacity, colors, max_gaussians=max_gaussians
                )
                # Recreate optimizer with decayed LR
                decay_factor = 0.5 ** (step // decay_interval)
                opt = create_optimizer(args.lr_pos * decay_factor, args.lr_other * decay_factor)
            # Activate scales every activate_interval
            if step % activate_interval == 0 and step > 0:
                with torch.no_grad():
                    scales = relu(scales - 0.0025)
            if step % ckpt_interval == 0:
                save_checkpoint(scene_stub, step, means, scales, quats, opacity, colors, args.out)
            if step >= args.iters:
                break
    # Final render
    pred_final = render_dgr(dgr, means, scales, quats, opacity, colors, K0, cam0["c2w"].to(device), cam0["W"], cam0["H"])
    save_png(pred_final, os.path.join(args.out, "final_render.png"))
    # Final checkpoint
    save_checkpoint(scene_stub, step, means, scales, quats, opacity, colors, args.out)
    print(f"Training complete. Final Gaussians: {means.shape[0]}")

if __name__ == "__main__":
    main()

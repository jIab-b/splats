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

def render_dgr(dgr, means, scales, quats, opacity, colors, sh_coeffs, sh_degree, prefiltered, K, c2w, W, H):
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
        sh_degree=int(sh_degree),
        campos=campos,
        prefiltered=bool(prefiltered),
        debug=False,
    )
    rast = dgr.GaussianRasterizer(settings)
    means2D = torch.zeros(means.shape[0], 3, device=means.device, dtype=means.dtype)
    if sh_degree and sh_degree > 0 and sh_coeffs is not None:
        out, _ = rast(means, means2D, opacity, shs=sh_coeffs, colors_precomp=None, scales=scales, rotations=qn, cov3D_precomp=None)
    else:
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

def densify_and_prune(means, scales, quats, opacity, colors, sh_coeffs=None, sh_degree=0, grad_threshold=0.00016, prune_threshold=0.02, add_new=250, max_gaussians=1000000, clone_max=50000):
    # Detach inputs to break history after opt.step
    means = means.detach().requires_grad_(True)
    scales = scales.detach().requires_grad_(True)
    quats = quats.detach().requires_grad_(True)
    opacity = opacity.detach().requires_grad_(True)
    colors = colors.detach().requires_grad_(True)
    if sh_coeffs is not None and sh_degree and sh_degree > 0:
        sh_coeffs = sh_coeffs.detach().requires_grad_(True)
    
    # Prune low opacity
    mask = opacity.squeeze() > prune_threshold
    means = means[mask]
    scales = scales[mask]
    quats = quats[mask]
    opacity = opacity[mask]
    colors = colors[mask]
    if sh_coeffs is not None and sh_degree and sh_degree > 0:
        sh_coeffs = sh_coeffs[mask]
    # Detach after prune to ensure leaf
    means = means.detach().requires_grad_(True)
    scales = scales.detach().requires_grad_(True)
    quats = quats.detach().requires_grad_(True)
    opacity = opacity.detach().requires_grad_(True)
    colors = colors.detach().requires_grad_(True)
    if sh_coeffs is not None and sh_degree and sh_degree > 0:
        sh_coeffs = sh_coeffs.detach().requires_grad_(True)
    
    large_scale_mask = (scales.max(dim=1)[0] > 0.01).squeeze()
    clone_idx = torch.nonzero(large_scale_mask).flatten()
    if clone_idx.numel() > 0:
        room = max(0, int(max_gaussians - means.shape[0]))
        allow = min(room, int(clone_max)) if clone_max is not None else room
        if allow > 0:
            if clone_idx.numel() > allow:
                perm = torch.randperm(clone_idx.numel(), device=clone_idx.device)
                clone_idx = clone_idx[perm[:allow]]
            clone_means = means[clone_idx] + torch.randn_like(means[clone_idx]) * 0.001
            clone_scales = scales[clone_idx] * 0.5 + torch.randn_like(scales[clone_idx]) * 0.001
            clone_quats = quats[clone_idx]
            clone_opacity = opacity[clone_idx]
            clone_colors = colors[clone_idx]
            if sh_coeffs is not None and sh_degree and sh_degree > 0:
                clone_sh = sh_coeffs[clone_idx]
            means = torch.cat([means, clone_means], dim=0)
            scales = torch.cat([scales, clone_scales], dim=0)
            quats = torch.cat([quats, clone_quats], dim=0)
            opacity = torch.cat([opacity, clone_opacity], dim=0)
            colors = torch.cat([colors, clone_colors], dim=0)
            if sh_coeffs is not None and sh_degree and sh_degree > 0:
                sh_coeffs = torch.cat([sh_coeffs, clone_sh], dim=0)
            means = means.detach().requires_grad_(True)
            scales = scales.detach().requires_grad_(True)
            quats = quats.detach().requires_grad_(True)
            opacity = opacity.detach().requires_grad_(True)
            colors = colors.detach().requires_grad_(True)
            if sh_coeffs is not None and sh_degree and sh_degree > 0:
                sh_coeffs = sh_coeffs.detach().requires_grad_(True)
    
    # Add new random Gaussians if below max
    current_n = means.shape[0]
    if current_n < max_gaussians:
        add_n = min(add_new, max(0, max_gaussians - current_n))
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
            if sh_coeffs is not None and sh_degree and sh_degree > 0:
                num_sh = sh_coeffs.shape[1]
                new_sh = torch.zeros(add_n, num_sh, 3, device=means.device, requires_grad=True)
                new_sh[:, 0, :] = new_colors
            means = torch.cat([means, new_means], dim=0)
            scales = torch.cat([scales, new_scales], dim=0)
            quats = torch.cat([quats, new_quats], dim=0)
            opacity = torch.cat([opacity, new_opacity], dim=0)
            colors = torch.cat([colors, new_colors], dim=0)
            if sh_coeffs is not None and sh_degree and sh_degree > 0:
                sh_coeffs = torch.cat([sh_coeffs, new_sh], dim=0)
            # Detach after add to ensure leaf
            means = means.detach().requires_grad_(True)
            scales = scales.detach().requires_grad_(True)
            quats = quats.detach().requires_grad_(True)
            opacity = opacity.detach().requires_grad_(True)
            colors = colors.detach().requires_grad_(True)
            if sh_coeffs is not None and sh_degree and sh_degree > 0:
                sh_coeffs = sh_coeffs.detach().requires_grad_(True)
    
    return means, scales, quats, opacity, colors, sh_coeffs

def save_checkpoint(scene_stub, step, means, scales, quats, opacity, colors, out_dir, sh_coeffs=None):
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
    if sh_coeffs is not None:
        ckpt["sh_coeffs"] = sh_coeffs.detach().cpu()
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
    ap.add_argument("--downscale", type=float, default=1.0)
    ap.add_argument("--progressive_until", type=int, default=0)
    ap.add_argument("--max_gaussians", type=int, default=3000000)
    ap.add_argument("--densify_add", type=int, default=250)
    ap.add_argument("--prune_threshold", type=float, default=0.02)
    ap.add_argument("--clone_max", type=int, default=50000)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--sh_degree", type=int, default=0)
    ap.add_argument("--no_prefilter", action="store_true")
    args = ap.parse_args()
    scene_stub = os.path.basename(os.path.normpath(args.scene))
    if torch.cuda.is_available() and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = None if args.images_dir == "auto" else args.images_dir
    ds = GardenDataset(args.scene, images_dir=images_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)
    max_gaussians = int(args.max_gaussians)
    means, scales, quats, opacity, colors, meta = init_gaussians_constant_density(
        args.scene, device, max_points=args.init_count, images_dir=images_dir, dataset=ds
    )
    sh_coeffs = None
    if args.sh_degree and args.sh_degree > 0:
        num_sh = (args.sh_degree + 1) * (args.sh_degree + 1)
        sh_coeffs = torch.zeros(means.shape[0], num_sh, 3, device=device, requires_grad=True)
        with torch.no_grad():
            sh_coeffs[:, 0, :] = colors
    # Initial render
    os.makedirs(args.out, exist_ok=True)
    s0 = ds[0]
    cam0 = s0["camera"]
    K0 = build_K(cam0, device)
    dgr = _ensure_dgr()
    init_ds = max(1.0, float(args.downscale))
    if init_ds > 1.0:
        H0 = int(cam0["H"] // init_ds)
        W0 = int(cam0["W"] // init_ds)
        cam0_ds = dict(cam0)
        cam0_ds["H"], cam0_ds["W"] = H0, W0
        cam0_ds["fx"] /= init_ds
        cam0_ds["fy"] /= init_ds
        cam0_ds["cx"] /= init_ds
        cam0_ds["cy"] /= init_ds
        K0 = build_K(cam0_ds, device)
        pred_initial = render_dgr(dgr, means, scales, quats, opacity, colors, sh_coeffs, args.sh_degree, not args.no_prefilter, K0, cam0["c2w"].to(device), W0, H0)
    else:
        pred_initial = render_dgr(dgr, means, scales, quats, opacity, colors, sh_coeffs, args.sh_degree, not args.no_prefilter, K0, cam0["c2w"].to(device), cam0["W"], cam0["H"])
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
        params = [
            {"params": [means], "lr": lr_pos},
            {"params": [scales, quats, opacity, colors], "lr": lr_other},
        ]
        if sh_coeffs is not None:
            params.append({"params": [sh_coeffs], "lr": lr_other})
        return Adam(params)
    opt = create_optimizer(args.lr_pos, args.lr_other)
    use_amp = (not args.no_amp) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
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
            ds_factor = 1.0
            if args.progressive_until and step < args.progressive_until:
                ds_factor = max(1.0, float(args.downscale))
            if ds_factor > 1.0:
                Hds = int(cam["H"] // ds_factor)
                Wds = int(cam["W"] // ds_factor)
                cam_ds = dict(cam)
                cam_ds["H"], cam_ds["W"] = Hds, Wds
                cam_ds["fx"] /= ds_factor
                cam_ds["fy"] /= ds_factor
                cam_ds["cx"] /= ds_factor
                cam_ds["cy"] /= ds_factor
                K = build_K(cam_ds, device)
                img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(Hds, Wds), mode="bilinear", align_corners=False).squeeze(0)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = render_dgr(dgr, means, scales, quats, opacity, colors, sh_coeffs, args.sh_degree, not args.no_prefilter, K, cam["c2w"], Wds, Hds)
                    loss = l2(pred, img) + 1e-3 * (scales ** 2).mean() + 1e-3 * (opacity.clamp(0,1) ** 2).mean()
            else:
                K = build_K(cam, device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = render_dgr(dgr, means, scales, quats, opacity, colors, sh_coeffs, args.sh_degree, not args.no_prefilter, K, cam["c2w"], cam["W"], cam["H"])
                    loss = l2(pred, img) + 1e-3 * (scales ** 2).mean() + 1e-3 * (opacity.clamp(0,1) ** 2).mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            clamp_params(scales, opacity)
            scaler.step(opt)
            scaler.update()
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
                with torch.no_grad():
                    for idx, s in enumerate(view_samples):
                        camv = s["camera"]
                        v_ds = ds_factor
                        if v_ds > 1.0:
                            Hvw = int(camv["H"] // v_ds)
                            Wvw = int(camv["W"] // v_ds)
                            camv_ds = dict(camv)
                            camv_ds["H"], camv_ds["W"] = Hvw, Wvw
                            camv_ds["fx"] /= v_ds
                            camv_ds["fy"] /= v_ds
                            camv_ds["cx"] /= v_ds
                            camv_ds["cy"] /= v_ds
                            Kv = build_K(camv_ds, device)
                            pred_view = render_dgr(dgr, means, scales, quats, opacity, colors, sh_coeffs, args.sh_degree, not args.no_prefilter, Kv, camv["c2w"].to(device), Wvw, Hvw)
                        else:
                            Kv = build_K(camv, device)
                            pred_view = render_dgr(dgr, means, scales, quats, opacity, colors, sh_coeffs, args.sh_degree, not args.no_prefilter, Kv, camv["c2w"].to(device), camv["W"], camv["H"])
                        save_png(pred_view, os.path.join(iter_dir, f"view_{idx}.png"))
            # Densify and prune every densify_interval
            if step % densify_interval == 0 and step > 0:
                means, scales, quats, opacity, colors, sh_coeffs = densify_and_prune(
                    means, scales, quats, opacity, colors, sh_coeffs,
                    sh_degree=args.sh_degree,
                    prune_threshold=args.prune_threshold,
                    add_new=args.densify_add,
                    max_gaussians=max_gaussians,
                    clone_max=args.clone_max,
                )
                # Recreate optimizer with decayed LR
                decay_factor = 0.5 ** (step // decay_interval)
                opt = create_optimizer(args.lr_pos * decay_factor, args.lr_other * decay_factor)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # Activate scales every activate_interval
            if step % activate_interval == 0 and step > 0:
                with torch.no_grad():
                    scales = relu(scales - 0.0025)
            if step % ckpt_interval == 0:
                save_checkpoint(scene_stub, step, means, scales, quats, opacity, colors, args.out, sh_coeffs)
            if step >= args.iters:
                break
    # Final render
    if init_ds > 1.0:
        pred_final = render_dgr(dgr, means, scales, quats, opacity, colors, sh_coeffs, args.sh_degree, not args.no_prefilter, K0, cam0["c2w"].to(device), int(cam0["W"] // init_ds), int(cam0["H"] // init_ds))
    else:
        pred_final = render_dgr(dgr, means, scales, quats, opacity, colors, sh_coeffs, args.sh_degree, not args.no_prefilter, K0, cam0["c2w"].to(device), cam0["W"], cam0["H"])
    save_png(pred_final, os.path.join(args.out, "final_render.png"))
    # Final checkpoint
    save_checkpoint(scene_stub, step, means, scales, quats, opacity, colors, args.out, sh_coeffs)
    print(f"Training complete. Final Gaussians: {means.shape[0]}")

if __name__ == "__main__":
    main()

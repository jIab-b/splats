import os
import json
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from safetensors.torch import save_file, load_file
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LinearLR, SequentialLR
from collections import deque

class TransformsDataset(torch.utils.data.Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        with open(self.root / "transforms.json", "r") as f:
            meta = json.load(f)
        self.W = int(meta["w"]) 
        self.H = int(meta["h"]) 
        self.fx = float(meta["fl_x"]) 
        self.fy = float(meta["fl_y"]) 
        self.cx = float(meta["cx"]) 
        self.cy = float(meta["cy"]) 
        self.frames = meta["frames"]
        self.paths = [self.root / f["file_path"] for f in self.frames]
        self.c2w = torch.tensor(np.stack([np.array(f["transform_matrix"], dtype=np.float32) for f in self.frames]), dtype=torch.float32)
        self.K = torch.tensor([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
        self.WH = (self.W, self.H)
    def __len__(self) -> int:
        return len(self.paths)
    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB").resize(self.WH, Image.BILINEAR)
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.permute(2, 0, 1)
        return {"image": img, "c2w": self.c2w[idx], "K": self.K, "W": self.W, "H": self.H}

def init_gaussians(count: int, radius: float, device: torch.device):
    means = torch.empty(count, 3, device=device).uniform_(-1.0, 1.0)
    means = torch.nn.functional.normalize(means, dim=-1) * radius
    scales = torch.full((count, 3), 0.02, device=device)
    quats = torch.zeros(count, 4, device=device)
    quats[:, 0] = 1.0
    opac = torch.full((count, 1), 0.8, device=device)
    colors = torch.full((count, 3), 0.8, device=device)
    means.requires_grad_(True)
    scales.requires_grad_(True)
    quats.requires_grad_(True)
    opac.requires_grad_(True)
    colors.requires_grad_(True)
    return means, scales, quats, opac, colors

def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-8)

def clamp_params(scales: torch.Tensor, opac: torch.Tensor) -> None:
    with torch.no_grad():
        scales.clamp_(min=1e-4, max=0.2)
        opac.clamp_(min=0.01, max=1.0)

class Renderer:
    def __init__(self, device: torch.device):
        import diff_gaussian_rasterization as dgr
        self.dgr = dgr
        self.device = device

    def _build_matrices(self, K: torch.Tensor, c2w: torch.Tensor, W: int, H: int):
        w2c = torch.linalg.inv(c2w)
        view = w2c.clone()
        view[:3, 1:3] *= -1
        near = 0.01
        far = 100.0
        proj = torch.zeros(4, 4, device=c2w.device, dtype=c2w.dtype)
        proj[0, 0] = 2.0 * K[0, 0] / float(W)
        proj[1, 1] = 2.0 * K[1, 1] / float(H)
        proj[2, 0] = 1.0 - 2.0 * K[0, 2] / float(W)
        proj[2, 1] = 2.0 * K[1, 2] / float(H) - 1.0
        proj[2, 2] = (far + near) / (far - near)
        proj[2, 3] = 1.0
        proj[3, 2] = -(2.0 * far * near) / (far - near)
        proj_view = proj @ view
        return view, proj_view

    def forward(self, means, scales, quats, opac, colors, K, c2w, W, H):
        qn = quat_normalize(quats)
        a = opac
        bg = torch.ones(3, device=means.device, dtype=means.dtype)
        view, proj_view = self._build_matrices(K, c2w, W, H)
        campos = c2w[:3, 3].clone()
        campos[1:3] *= -1
        tanfovx = float(W) / (2.0 * float(K[0, 0]))
        tanfovy = float(H) / (2.0 * float(K[1, 1]))
        settings = self.dgr.GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=view.transpose(0, 1).contiguous(),
            projmatrix=proj_view.transpose(0, 1).contiguous(),
            sh_degree=0,
            campos=campos,
            prefiltered=False,
            debug=False,
        )
        rasterizer = self.dgr.GaussianRasterizer(settings)
        means2D = torch.zeros(means.shape[0], 3, device=means.device, dtype=means.dtype)
        out_color, _ = rasterizer(
            means,
            means2D,
            a,
            colors_precomp=colors,
            scales=scales,
            rotations=qn,
            cov3D_precomp=None,
        )
        return out_color

def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x * mu_y + C1) * (sigma_x + sigma_y + C2)
    s = (num + 1e-8) / (den + 1e-8)
    return s.clamp(0, 1).mean()

def loss_photometric(pred: torch.Tensor, gt: torch.Tensor, lam: float = 0.2) -> torch.Tensor:
    import torch.nn.functional as F
    l1 = (pred - gt).abs().mean()
    s = 1.0 - ssim(pred, gt)
    return l1 + lam * s

def prune(means, scales, quats, opac, colors, keep_mask):
    return means[keep_mask], scales[keep_mask], quats[keep_mask], opac[keep_mask], colors[keep_mask]

def clone(means, scales, quats, opac, colors, sel, noise: float = 0.01):
    m = means[sel] + torch.randn_like(means[sel]) * noise
    s = scales[sel] * 0.8
    q = quats[sel]
    a = opac[sel] * 0.9
    c = colors[sel]
    return torch.cat([means, m], 0), torch.cat([scales, s], 0), torch.cat([quats, q], 0), torch.cat([opac, a], 0), torch.cat([colors, c], 0)

def write_ply(path: str, xyz, colors, scales, quats, opac):
    n = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float scale_x\n")
        f.write("property float scale_y\n")
        f.write("property float scale_z\n")
        f.write("property float qw\n")
        f.write("property float qx\n")
        f.write("property float qy\n")
        f.write("property float qz\n")
        f.write("property float opacity\n")
        f.write("end_header\n")
        for i in range(n):
            r, g, b = (np.clip(colors[i] * 255.0, 0, 255)).astype(np.uint8)
            sx, sy, sz = scales[i]
            qw, qx, qy, qz = quats[i]
            a = opac[i]
            x, y, z = xyz[i]
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)} {sx} {sy} {sz} {qw} {qx} {qy} {qz} {a}\n")

def save_checkpoint(dir_path: Path, step: int, means: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor, opac: torch.Tensor, colors: torch.Tensor, opt: torch.optim.Optimizer | None):
    dir_path.mkdir(parents=True, exist_ok=True)
    save_file({
        "means": means.detach().cpu(),
        "scales": scales.detach().cpu(),
        "quats": quat_normalize(quats).detach().cpu(),
        "opac": opac.detach().cpu(),
        "colors": colors.detach().cpu(),
    }, str(dir_path / f"state_{step:06d}.safetensors"))
    if opt is not None:
        torch.save({"step": step, "opt": opt.state_dict()}, str(dir_path / f"opt_{step:06d}.pt"))

def load_checkpoint(state_path: Path, device: torch.device):
    st = load_file(str(state_path))
    means = st["means"].to(device).requires_grad_(True)
    scales = st["scales"].to(device).requires_grad_(True)
    quats = st["quats"].to(device).requires_grad_(True)
    opac = st["opac"].to(device).requires_grad_(True)
    colors = st["colors"].to(device).requires_grad_(True)
    return means, scales, quats, opac, colors

def find_latest_ckpt(dir_path: Path):
    if not dir_path.exists():
        return None, None
    states = sorted(dir_path.glob("state_*.safetensors"))
    if not states:
        return None, None
    latest_state = states[-1]
    step_str = latest_state.stem.split("_")[-1]
    try:
        step_val = int(step_str)
    except Exception:
        step_val = None
    opt_path = dir_path / f"opt_{step_str}.pt"
    return latest_state, (opt_path if opt_path.exists() else None), step_val

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_count", type=int, default=50000)
    p.add_argument("--radius", type=float, default=1.5)
    p.add_argument("--lr_pos", type=float, default=1e-2)
    p.add_argument("--lr_other", type=float, default=1e-3)
    p.add_argument("--densify_every", type=int, default=200)
    p.add_argument("--clone_frac", type=float, default=0.05)
    p.add_argument("--prune_alpha", type=float, default=0.05)
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--resume_opt", action="store_true")
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "plateau"])
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--plateau_patience", type=int, default=5)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--early_stop_delta", type=float, default=1e-3)
    p.add_argument("--stop_clone_after", type=int, default=5000)
    p.add_argument("--max_count", type=int, default=100000)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TransformsDataset(args.data)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    ckpt_root = Path(args.ckpt_dir) if args.ckpt_dir is not None else Path(args.out) / "ckpts"
    start_step = 0
    if args.resume:
        state_path = None
        opt_path = None
        step_val = None
        if args.resume == "latest":
            state_path, opt_path, step_val = find_latest_ckpt(ckpt_root)
        else:
            state_path = Path(args.resume)
            if state_path.exists():
                stem = state_path.stem.split("_")[-1]
                op_try = state_path.parent / f"opt_{stem}.pt"
                opt_path = op_try if op_try.exists() else None
                try:
                    step_val = int(stem)
                except Exception:
                    step_val = 0
        if state_path is not None and state_path.exists():
            means, scales, quats, opac, colors = load_checkpoint(state_path, device)
            start_step = step_val or 0
        else:
            means, scales, quats, opac, colors = init_gaussians(args.init_count, args.radius, device)
    else:
        means, scales, quats, opac, colors = init_gaussians(args.init_count, args.radius, device)
    opt = torch.optim.Adam([
        {"params": [means], "lr": args.lr_pos},
        {"params": [scales, quats, opac, colors], "lr": args.lr_other},
    ])
    scheduler = None
    if args.scheduler == "cosine":
        if args.warmup_steps > 0:
            warm = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps)
            main = CosineAnnealingLR(opt, T_max=max(1, args.iters - args.warmup_steps), eta_min=args.min_lr)
            scheduler = SequentialLR(opt, schedulers=[warm, main], milestones=[args.warmup_steps], last_epoch=0)
        else:
            scheduler = CosineAnnealingLR(opt, T_max=args.iters, eta_min=args.min_lr)
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(opt, factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr)
    if args.resume and args.resume_opt and 'opt_path' in locals() and opt_path is not None and Path(opt_path).exists():
        try:
            meta = torch.load(str(opt_path), map_location='cpu')
            if isinstance(meta, dict) and "opt" in meta:
                opt.load_state_dict(meta["opt"])
        except Exception:
            pass
    rend = Renderer(device)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    step = start_step
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and torch.cuda.is_available()))
    loss_sum_interval = 0.0
    loss_count_interval = 0
    best_interval_loss = None
    no_improve_intervals = 0
    while step < args.iters:
        for batch in dl:
            img = batch["image"].to(device).squeeze(0)
            c2w = batch["c2w"].to(device)[0]
            K = batch["K"].to(device)
            if K.dim() == 3 and K.size(0) == 1:
                K = K[0]
            W = batch["W"]
            H = batch["H"]
            if torch.is_tensor(W):
                W = int(W.item())
            else:
                W = int(W)
            if torch.is_tensor(H):
                H = int(H.item())
            else:
                H = int(H)
            with torch.cuda.amp.autocast(enabled=(args.amp and torch.cuda.is_available())):
                pred = rend.forward(means, scales, quats, opac, colors, K, c2w, W, H)
                if pred.dim() == 3 and pred.shape[0] == H and pred.shape[1] == W:
                    pred = pred.permute(2, 0, 1)
                loss = loss_photometric(pred, img)
            opt.zero_grad()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                clamp_params(scales, opac)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                clamp_params(scales, opac)
                opt.step()
            if args.scheduler == "cosine" and scheduler is not None:
                scheduler.step()
            loss_sum_interval += float(loss.detach().item())
            loss_count_interval += 1
            step += 1
            if step % args.densify_every == 0:
                with torch.no_grad():
                    keep = (opac.squeeze(-1) > args.prune_alpha)
                    means, scales, quats, opac, colors = prune(means, scales, quats, opac, colors, keep)
                    do_clone = (step < args.stop_clone_after) and (means.shape[0] < args.max_count)
                    if do_clone:
                        nadd = int(max(1, int(means.shape[0] * args.clone_frac)))
                        if means.shape[0] + nadd > args.max_count:
                            nadd = max(0, args.max_count - means.shape[0])
                        if nadd > 0:
                            idx = torch.randperm(means.shape[0], device=device)[:nadd]
                            means, scales, quats, opac, colors = clone(means, scales, quats, opac, colors, idx)
                means.requires_grad_(True)
                scales.requires_grad_(True)
                quats.requires_grad_(True)
                opac.requires_grad_(True)
                colors.requires_grad_(True)
                opt = torch.optim.Adam([
                    {"params": [means], "lr": args.lr_pos},
                    {"params": [scales, quats, opac, colors], "lr": args.lr_other},
                ])
                if args.scheduler == "cosine":
                    if args.warmup_steps > 0:
                        warm = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps)
                        main = CosineAnnealingLR(opt, T_max=max(1, args.iters - args.warmup_steps), eta_min=args.min_lr)
                        scheduler = SequentialLR(opt, schedulers=[warm, main], milestones=[args.warmup_steps], last_epoch=step)
                    else:
                        scheduler = CosineAnnealingLR(opt, T_max=args.iters, eta_min=args.min_lr, last_epoch=step)
                elif args.scheduler == "plateau":
                    scheduler = ReduceLROnPlateau(opt, factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr)
            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                with torch.no_grad():
                    save_checkpoint(ckpt_root, step, means, scales, quats, opac, colors, opt)
                if args.scheduler == "plateau" and scheduler is not None and loss_count_interval > 0:
                    avg_loss = loss_sum_interval / max(1, loss_count_interval)
                    scheduler.step(avg_loss)
                if args.early_stop_patience > 0 and loss_count_interval > 0:
                    interval_loss = loss_sum_interval / max(1, loss_count_interval)
                    if best_interval_loss is None or interval_loss < best_interval_loss * (1.0 - args.early_stop_delta):
                        best_interval_loss = interval_loss
                        no_improve_intervals = 0
                    else:
                        no_improve_intervals += 1
                        if no_improve_intervals >= args.early_stop_patience:
                            step = args.iters
                    loss_sum_interval = 0.0
                    loss_count_interval = 0
            if step % 1000 == 0:
                with torch.no_grad():
                    xyz = means.detach().cpu().numpy()
                    rgb = colors.detach().cpu().numpy()
                    s = scales.detach().cpu().numpy()
                    q = quat_normalize(quats).detach().cpu().numpy()
                    a = opac.detach().cpu().numpy().squeeze(-1)
                    write_ply(os.path.join(args.out, f"step_{step:06d}.ply"), xyz, rgb, s, q, a)
            if step >= args.iters:
                break
    with torch.no_grad():
        xyz = means.detach().cpu().numpy()
        rgb = colors.detach().cpu().numpy()
        s = scales.detach().cpu().numpy()
        q = quat_normalize(quats).detach().cpu().numpy()
        a = opac.detach().cpu().numpy().squeeze(-1)
        write_ply(os.path.join(args.out, "final.ply"), xyz, rgb, s, q, a)
        save_checkpoint(ckpt_root, step, means, scales, quats, opac, colors, opt)

if __name__ == "__main__":
    main()

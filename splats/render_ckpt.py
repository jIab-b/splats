import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from PIL import Image

def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-8)

def load_transforms(root: Path):
    with open(root / "transforms.json", "r") as f:
        meta = json.load(f)
    W = int(meta["w"])
    H = int(meta["h"])
    fx = float(meta["fl_x"]) 
    fy = float(meta["fl_y"]) 
    cx = float(meta["cx"]) 
    cy = float(meta["cy"]) 
    frames = meta["frames"]
    c2w = torch.tensor(np.stack([np.array(f["transform_matrix"], dtype=np.float32) for f in frames]), dtype=torch.float32)
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return W, H, K, c2w

def build_view_proj(K: torch.Tensor, c2w: torch.Tensor, W: int, H: int):
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
    return view.transpose(0, 1).contiguous(), (proj @ view).transpose(0, 1).contiguous()

def render_frames(ckpt_path: Path, data_root: Path, out_dir: Path, every: int, max_frames: int | None, device: torch.device):
    import diff_gaussian_rasterization as dgr
    st = load_file(str(ckpt_path))
    means = st["means"].to(device)
    scales = st["scales"].to(device)
    quats = quat_normalize(st["quats"].to(device))
    opac = st["opac"].to(device)
    colors = st["colors"].to(device)
    W, H, K, c2w_all = load_transforms(data_root)
    K = K.to(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    bg = torch.zeros(3, device=device)
    tanfovx = float(W) / (2.0 * float(K[0, 0]))
    tanfovy = float(H) / (2.0 * float(K[1, 1]))
    r = None
    done = 0
    with torch.no_grad():
        for i in range(0, c2w_all.shape[0], max(1, every)):
            if max_frames is not None and done >= max_frames:
                break
            c2w = c2w_all[i].to(device)
            view, proj_view = build_view_proj(K, c2w, W, H)
            campos = c2w[:3, 3].clone().to(device)
            campos[1:3] *= -1
            settings = dgr.GaussianRasterizationSettings(
                image_height=H,
                image_width=W,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg,
                scale_modifier=1.0,
                viewmatrix=view,
                projmatrix=proj_view,
                sh_degree=0,
                campos=campos,
                prefiltered=False,
                debug=False,
            )
            if r is None:
                r = dgr.GaussianRasterizer(settings)
            else:
                r.raster_settings = settings
            means2D = torch.zeros(means.shape[0], 3, device=device, dtype=means.dtype)
            img, _ = r(means, means2D, opac, colors_precomp=colors, scales=scales, rotations=quats, cov3D_precomp=None)
            if img.dim() == 3 and img.shape[0] == 3:
                img_np = torch.clamp(img.permute(1, 2, 0), 0, 1).cpu().numpy()
            elif img.dim() == 3 and img.shape[-1] == 3:
                img_np = torch.clamp(img, 0, 1).cpu().numpy()
            else:
                img_np = torch.clamp(img.squeeze(), 0, 1).unsqueeze(-1).repeat(1,1,3).cpu().numpy()
            Image.fromarray((img_np * 255.0).astype(np.uint8)).save(out_dir / f"frame_{i:05d}.png")
            done += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_frames(Path(args.ckpt), Path(args.data), Path(args.out), args.every, args.max_frames, device)

if __name__ == "__main__":
    main()



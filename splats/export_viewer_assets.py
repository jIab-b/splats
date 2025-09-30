import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-8)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_points", type=int, default=500_000)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    st = load_file(str(ckpt))
    means = st["means"].cpu()
    scales = st["scales"].cpu()
    quats = quat_normalize(st["quats"]).cpu()
    opac = st["opac"].cpu().squeeze(-1)
    colors = st["colors"].cpu()

    n = means.shape[0]
    if args.max_points and n > args.max_points:
        idx = torch.randperm(n)[: args.max_points]
        means = means[idx]
        scales = scales[idx]
        quats = quats[idx]
        opac = opac[idx]
        colors = colors[idx]
        n = means.shape[0]

    # Pack as float32 interleaved: x,y,z, r,g,b, sx,sy,sz, opacity
    xyz = means.numpy().astype(np.float32)
    rgb = np.clip(colors.numpy(), 0.0, 1.0).astype(np.float32)
    s3 = np.clip(scales.numpy(), 1e-6, 1.0).astype(np.float32)
    a = np.clip(opac.numpy(), 0.0, 1.0).astype(np.float32).reshape(-1, 1)

    data = np.concatenate([xyz, rgb, s3, a], axis=1)
    assert data.shape[1] == 10
    bin_path = out_dir / "viewer_data.bin"
    data.tofile(bin_path)

    center = xyz.mean(axis=0).tolist()
    mins = xyz.min(axis=0).tolist()
    maxs = xyz.max(axis=0).tolist()
    radius = float(np.linalg.norm((np.array(maxs) - np.array(mins)) * 0.5))
    meta = {
        "count": int(n),
        "stride": 10,
        "attrs": ["x","y","z","r","g","b","sx","sy","sz","opacity"],
        "center": center,
        "bbox_min": mins,
        "bbox_max": maxs,
        "radius": radius,
        "source": str(ckpt),
    }
    with open(out_dir / "viewer_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Write a simple index file for convenience
    with open(out_dir / "index.html", "w") as f:
        f.write("""<!doctype html><html><head><meta http-equiv=\"refresh\" content=\"0; url=viewer.html\"></head><body></body></html>""")

if __name__ == "__main__":
    main()



import os, glob, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def _pick_img_dir(scene_dir):
    for d in ["images_8","images_4","images_2","images"]:
        p = os.path.join(scene_dir, d)
        if os.path.isdir(p):
            return p
    return os.path.join(scene_dir, "images")

def _load_llff_poses(scene_dir):
    p = os.path.join(scene_dir, "poses_bounds.npy")
    if not os.path.isfile(p):
        return None
    a = np.load(p)
    poses = a[:, :15].reshape(-1, 3, 5)
    hwf = poses[:, :, 4]
    h = int(hwf[0, 0]); w = int(hwf[0, 1]); f = float(hwf[0, 2])
    m34 = poses[:, :, :4]
    c2w = np.tile(np.eye(4, dtype=np.float32), (m34.shape[0], 1, 1))
    c2w[:, :3, :4] = m34
    return h, w, f, c2w

class GardenDataset(Dataset):
    def __init__(self, scene_dir, images_dir=None):
        self.scene_dir = scene_dir
        if images_dir is not None:
            self.img_dir = os.path.join(scene_dir, images_dir)
            # Parse scale from dir name, e.g., 'images_8' -> 1/8.0
            if '_' in images_dir:
                try:
                    scale_str = images_dir.split('_')[1]
                    scale = 1.0 / float(scale_str)
                except:
                    scale = 1.0
            else:
                scale = 1.0
            self.scale = scale
        else:
            self.img_dir = _pick_img_dir(scene_dir)
            scale = 1.0
            self.scale = scale
        self.paths = sorted(glob.glob(os.path.join(self.img_dir, "*.JPG"))) + \
                     sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        if len(self.paths) == 0:
            raise RuntimeError("no images")
        poses = _load_llff_poses(scene_dir)
        if poses is None:
            img = Image.open(self.paths[0]).convert("RGB")
            self.w_full, self.h_full = img.size
            self.f_full = max(self.h_full, self.w_full) * 0.5
            self.c2w = [np.eye(4, dtype=np.float32) for _ in self.paths]
        else:
            self.h_full, self.w_full, self.f_full, c2w = poses
            self.c2w = [c2w[i] for i in range(len(self.paths))]
        self._c2w_np = np.stack(self.c2w, axis=0)
        self._camera_positions = self._c2w_np[:, :3, 3].copy()
        self._base_bounds = compute_camera_bounds(
            self._camera_positions,
            pad_fraction=0.0,
            min_extent=1e-3,
        )
        # Apply scale
        self.h = int(self.h_full * scale)
        self.w = int(self.w_full * scale)
        self.f = self.f_full * scale

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize((self.w, self.h))
        x = torch.from_numpy(np.array(img)).float() / 255.0
        x = x.permute(2, 0, 1).contiguous()
        cam = {
            "H": self.h, "W": self.w, 
            "fx": self.f, "fy": self.f,  # Scaled focal
            "cx": self.w * 0.5, "cy": self.h * 0.5,  # Scaled principal point
            "c2w": torch.from_numpy(self.c2w[i]).float()
        }
        return {"image": x, "camera": cam}

    @property
    def camera_positions(self):
        return self._camera_positions.copy()

    def scene_bounds(self, pad_fraction: float = 0.1, min_extent: float = 1.0):
        return compute_camera_bounds(
            self._camera_positions,
            pad_fraction=pad_fraction,
            min_extent=min_extent,
        )

def dataset_stats(scene_dir):
    ds = GardenDataset(scene_dir)
    return {
        "num_images": len(ds),
        "H": ds.h,
        "W": ds.w,
        "focal": ds.f,
        "img_dir": ds.img_dir,
    }

def sample_info(scene_dir, idx=0):
    ds = GardenDataset(scene_dir)
    i = idx % len(ds)
    s = ds[i]
    return {
        "idx": i,
        "image_shape": tuple(s["image"].shape),
        "c2w_shape": tuple(s["camera"]["c2w"].shape),
    }

def make_loader(scene_dir, batch_size=1, shuffle=True, num_workers=0):
    ds = GardenDataset(scene_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def _list_image_dirs(scene_dir):
    dirs = []
    for d in ["images", "images_2", "images_4", "images_8"]:
        p = os.path.join(scene_dir, d)
        if os.path.isdir(p):
            dirs.append(p)
    return dirs

def _list_images(img_dir):
    paths = sorted(glob.glob(os.path.join(img_dir, "*.JPG"))) + sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    return paths

def _read_image_size(path):
    im = Image.open(path)
    return im.size

def _validate_c2w(c2w):
    import numpy as np
    if c2w.shape != (4, 4):
        return False, {"reason": "shape"}
    if not np.allclose(c2w[3], np.array([0, 0, 0, 1], dtype=c2w.dtype), atol=1e-4):
        return False, {"reason": "bottom_row"}
    R = c2w[:3, :3]
    RtR = R.T @ R
    if not np.allclose(RtR, np.eye(3), atol=5e-3):
        return False, {"reason": "orthonormal"}
    det = np.linalg.det(R)
    if not (0.9 < det < 1.1):
        return False, {"reason": "det"}
    return True, {}

def validate_dataset(scene_dir, check_all_images=True):
    res = {"scene_dir": scene_dir, "dirs": {}, "poses": {}, "warnings": []}
    poses_path = os.path.join(scene_dir, "poses_bounds.npy")
    if not os.path.isfile(poses_path):
        res["poses"]["present"] = False
        return res
    a = np.load(poses_path)
    if a.ndim != 2 or a.shape[1] < 17:
        res["poses"]["present"] = True
        res["poses"]["shape"] = tuple(a.shape)
        res["warnings"].append("poses_bounds_bad_shape")
        return res
    n = a.shape[0]
    poses = a[:, :15].reshape(-1, 3, 5)
    hwf = poses[:, :, 4]
    H0 = int(hwf[0, 0]); W0 = int(hwf[0, 1]); F0 = float(hwf[0, 2])
    m34 = poses[:, :, :4]
    c2w = np.tile(np.eye(4, dtype=np.float32), (m34.shape[0], 1, 1))
    c2w[:, :3, :4] = m34
    near = a[:, 15]; far = a[:, 16]
    res["poses"] = {"present": True, "frames": int(n), "H": H0, "W": W0, "focal": float(F0), "near_min": float(near.min()), "far_max": float(far.max())}
    bad = 0
    for i in range(min(n, 50)):
        ok, info = _validate_c2w(c2w[i])
        if not ok:
            bad += 1
    if bad > 0:
        res["warnings"].append(f"bad_c2w_{bad}")
    for d in _list_image_dirs(scene_dir):
        paths = _list_images(d)
        if len(paths) == 0:
            res["dirs"][d] = {"images": 0}
            continue
        if check_all_images:
            sizes = [ _read_image_size(p) for p in paths ]
            same = all(s == sizes[0] for s in sizes)
            Wd, Hd = sizes[0]
        else:
            Wd, Hd = _read_image_size(paths[0])
            same = True
        scale_w = Wd / float(W0)
        scale_h = Hd / float(H0)
        focal_scaled = F0 * scale_w
        res["dirs"][d] = {
            "images": len(paths),
            "W": int(Wd),
            "H": int(Hd),
            "sizes_consistent": bool(same),
            "scale_w": float(scale_w),
            "scale_h": float(scale_h),
            "focal_expected": float(focal_scaled),
        }
        if abs(scale_w - scale_h) > 1e-3:
            res["warnings"].append(f"non_uniform_scale_{os.path.basename(d)}")
        if len(paths) != n:
            res["warnings"].append(f"count_mismatch_{os.path.basename(d)}_{len(paths)}vs{n}")
    return res


def compute_camera_bounds(positions: np.ndarray, pad_fraction: float = 0.1, min_extent: float = 1.0):
    if positions is None or len(positions) == 0:
        positions = np.zeros((1, 3), dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    extent = maxs - mins
    extent = np.maximum(extent, np.full(3, min_extent, dtype=np.float32))
    mins = center - 0.5 * extent
    maxs = center + 0.5 * extent
    pad = extent * pad_fraction
    mins_padded = mins - pad
    maxs_padded = maxs + pad
    extent_padded = maxs_padded - mins_padded
    diag = float(np.linalg.norm(extent_padded, ord=2))
    radius = 0.5 * diag
    return {
        "min": mins_padded.astype(np.float32),
        "max": maxs_padded.astype(np.float32),
        "center": (0.5 * (mins_padded + maxs_padded)).astype(np.float32),
        "extent": extent_padded.astype(np.float32),
        "radius": float(radius),
        "pad_fraction": float(pad_fraction),
        "min_extent": float(min_extent),
    }


def scene_bounds(scene_dir, images_dir=None, pad_fraction: float = 0.1, min_extent: float = 1.0):
    ds = GardenDataset(scene_dir, images_dir=images_dir)
    return ds.scene_bounds(pad_fraction=pad_fraction, min_extent=min_extent)


def main():
    scene = "/home/beed/splats/scenes/garden"
    print(dataset_stats(scene))
    print(sample_info(scene, idx=3))
    loader = make_loader(scene, batch_size=2)
    batch = next(iter(loader))
    print(batch["image"].shape, batch["camera"]["c2w"].shape)
    v = validate_dataset(scene, check_all_images=False)
    print({k: v[k] for k in ["scene_dir","poses","warnings"]})
    for d, info in v["dirs"].items():
        print(d, info)

    
if __name__ == "__main__":
    main()
    

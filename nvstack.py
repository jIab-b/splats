import os
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np


def set_build_env(base_mount: str) -> None:
    os.environ["CCACHE_DIR"] = str(Path(base_mount, "build", "ccache"))
    os.environ["CCACHE_NOHARDLINK"] = "true"
    os.environ["CCACHE_FILE_CLONE"] = "false"
    os.environ["CCACHE_TEMPDIR"] = "/tmp/ccache-tmp"
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    os.environ["CARGO_HOME"] = str(Path(base_mount, "build", "cargo"))
    os.environ["CARGO_TARGET_DIR"] = str(Path(base_mount, "build", "cargo", "target"))
    os.environ["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"
    os.environ["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
    os.environ["CMAKE_CUDA_COMPILER_LAUNCHER"] = "ccache"
    for p in [
        os.environ["CCACHE_DIR"],
        os.environ["CARGO_HOME"],
        os.environ["CARGO_TARGET_DIR"],
        os.environ["CCACHE_TEMPDIR"],
        str(Path(base_mount, "workspace")),
        str(Path(base_mount, "logs")),
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def clone_or_update_repo(target_dir: Path, repo_url: str, ref: str) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if not target_dir.exists():
        _run(["git", "clone", "--recursive", repo_url, str(target_dir)])
    _run(["bash", "-lc", f"cd {target_dir} && git fetch && git checkout {ref} && git submodule update --init --recursive"]) 


def build_3dgs(workspace_root: Path, repo_url: str = "https://github.com/graphdeco-inria/gaussian-splatting", ref: str = "main", torch_arch_list: str | None = None) -> Path:
    repo_dir = workspace_root / "3dgs"
    clone_or_update_repo(repo_dir, repo_url, ref)
    if torch_arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arch_list
    _run(["pip", "install", "-r", str(repo_dir / "requirements.txt")])
    _run(["pip", "install", str(repo_dir / "submodules" / "diff-gaussian-rasterization")])
    _run(["pip", "install", str(repo_dir / "submodules" / "simple-knn")])
    return repo_dir


def _load_gltf_meshes(gltf_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import trimesh
    scene = trimesh.load(str(gltf_path), force="scene")
    vertices = []
    faces = []
    for geom in scene.geometry.values():
        if not hasattr(geom, "vertices") or not hasattr(geom, "faces"):
            continue
        offset = len(vertices)
        vertices.extend(geom.vertices.tolist())
        faces.extend((geom.faces + offset).tolist())
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("No mesh data in glTF")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def initialize_splats_from_gltf(gltf_path: Path, target_count: int, out_npz: Path) -> Path:
    import trimesh
    v, f = _load_gltf_meshes(gltf_path)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    samples, face_idx = trimesh.sample.sample_surface_even(mesh, target_count)
    normals = mesh.face_normals[face_idx]
    albedo = np.full((target_count, 3), 0.8, dtype=np.float32)
    scale = np.full((target_count, 3), 0.01, dtype=np.float32)
    opacity = np.full((target_count, 1), 1.0, dtype=np.float32)
    rotation = np.zeros((target_count, 4), dtype=np.float32)
    rotation[:, 0] = 1.0
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        position=samples.astype(np.float32),
        normal=normals.astype(np.float32),
        albedo=albedo,
        scale=scale,
        opacity=opacity,
        rotation=rotation,
    )
    return out_npz


def export_splats_ply(npz_path: Path, ply_path: Path) -> Path:
    data = np.load(npz_path)
    xyz = data["position"].astype(np.float32)
    rgb = np.clip(data["albedo"], 0.0, 1.0)
    rgb8 = (rgb * 255.0).astype(np.uint8)
    n = xyz.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ply_path, "w") as f:
        for line in header:
            f.write(line + "\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb8[i]
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
    return ply_path



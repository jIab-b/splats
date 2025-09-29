import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh
import pyrender


def load_gltf_scene(gltf_path: Path) -> pyrender.Scene:
    tm_scene = trimesh.load(str(gltf_path), force="scene")
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
    for name, geom in tm_scene.geometry.items():
        if not hasattr(geom, "visual"):
            mesh = pyrender.Mesh.from_trimesh(geom, smooth=True)
        else:
            mesh = pyrender.Mesh.from_trimesh(geom, smooth=True)
        node_tf = np.eye(4)
        scene.add(mesh, pose=node_tf)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4))
    return scene


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    f = (target - eye)
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0:3, 3] = eye
    return m


def sample_orbit_views(num_views: int, radius: float, elevation_deg: float) -> list[np.ndarray]:
    views = []
    elev = math.radians(elevation_deg)
    for i in range(num_views):
        theta = 2.0 * math.pi * i / num_views
        eye = np.array([
            radius * math.cos(theta) * math.cos(elev),
            radius * math.sin(elev),
            radius * math.sin(theta) * math.cos(elev),
        ], dtype=np.float32)
        views.append(look_at(eye, np.array([0.0, 0.0, 0.0], dtype=np.float32)))
    return views


def render_dataset(
    gltf_path: Path,
    out_dir: Path,
    num_views: int = 300,
    radius: float = 4.0,
    elevation_deg: float = 20.0,
    width: int = 800,
    height: int = 800,
    focal_px: float | None = None,
) -> None:
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    scene = load_gltf_scene(gltf_path)
    if focal_px is None:
        focal_px = 0.95 * width
    camera = pyrender.IntrinsicsCamera(
        fx=focal_px,
        fy=focal_px,
        cx=width * 0.5,
        cy=height * 0.5,
    )
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    frames = []
    for i, pose in enumerate(sample_orbit_views(num_views, radius, elevation_deg)):
        cam_node = scene.add(camera, pose=pose)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        scene.remove_node(cam_node)
        img_name = f"view_{i:04d}.png"
        from imageio.v2 import imwrite
        imwrite(out_images / img_name, color)
        frames.append({
            "file_path": str(Path("images") / img_name),
            "transform_matrix": pose.tolist(),
        })
    transforms = {
        "w": width,
        "h": height,
        "fl_x": float(focal_px),
        "fl_y": float(focal_px),
        "cx": float(width * 0.5),
        "cy": float(height * 0.5),
        "frames": frames,
    }
    with open(out_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="gltf scene name under scenes/")
    parser.add_argument("--views", type=int, default=300)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--focal_px", type=float, default=None)
    args = parser.parse_args()
    gltf = Path("scenes") / args.scene / "scene.gltf"
    out_dir = Path("data") / args.scene
    render_dataset(gltf, out_dir, args.views, args.radius, args.elevation, args.width, args.height, args.focal_px)


if __name__ == "__main__":
    main()



import argparse
import json
import math
import os
from pathlib import Path


def configure_evee(width: int, height: int) -> None:
    import bpy
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.display_settings.display_device = 'sRGB'
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def clear_scene() -> None:
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)


def import_gltf(gltf_path: Path) -> None:
    import bpy
    bpy.ops.import_scene.gltf(filepath=str(gltf_path))


def ensure_camera() -> 'bpy.types.Object':
    import bpy
    cam = bpy.data.objects.get('Camera')
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
    bpy.context.scene.camera = cam
    return cam


def look_at(cam, target=(0.0, 0.0, 0.0)) -> None:
    import mathutils
    target_vec = mathutils.Vector(target)
    direction = target_vec - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()


def fibonacci_sphere(n: int) -> list[tuple[float, float, float]]:
    pts = []
    golden = (1 + 5 ** 0.5) / 2
    for i in range(n):
        t = (i + 0.5) / n
        z = 1 - 2 * t
        r = (1 - z * z) ** 0.5
        phi = 2 * math.pi * i / golden
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        pts.append((x, z, y))
    return pts


def make_intrinsics(cam, width: int, height: int, focal_mm: float | None) -> tuple[float, float, float, float]:
    import bpy
    if focal_mm is not None:
        cam.data.lens = focal_mm
    sensor_w = cam.data.sensor_width
    fx = cam.data.lens / sensor_w * width
    fy = fx
    cx = width * 0.5
    cy = height * 0.5
    return fx, fy, cx, cy


def render_dataset(scene_name: str, views: int, radii: list[float], width: int, height: int, focal_mm: float | None) -> None:
    import bpy
    clear_scene()
    gltf_path = Path('scenes') / scene_name / 'scene.gltf'
    out_dir = Path('data') / scene_name
    images_dir = out_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    configure_evee(width, height)
    import_gltf(gltf_path)
    cam = ensure_camera()
    fx, fy, cx, cy = make_intrinsics(cam, width, height, focal_mm)
    frames = []
    pts = fibonacci_sphere(views)
    idx = 0
    for r in radii:
        for p in pts:
            cam.location = (r * p[0], r * p[1], r * p[2])
            look_at(cam)
            bpy.context.scene.render.filepath = str(images_dir / f'view_{idx:04d}.png')
            bpy.ops.render.render(write_still=True)
            pose = cam.matrix_world
            frames.append({
                'file_path': str(Path('images') / f'view_{idx:04d}.png'),
                'transform_matrix': [list(row) for row in pose],
            })
            idx += 1
    transforms = {
        'w': width,
        'h': height,
        'fl_x': float(fx),
        'fl_y': float(fy),
        'cx': float(cx),
        'cy': float(cy),
        'frames': frames,
    }
    with open(Path('data') / scene_name / 'transforms.json', 'w') as f:
        json.dump(transforms, f, indent=2)


def main() -> None:
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--views', type=int, default=200)
    parser.add_argument('--radii', type=str, default='3.0,4.5')
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--focal_mm', type=float, default=None)
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)
    radii = [float(x) for x in args.radii.split(',') if x]
    render_dataset(args.scene, args.views, radii, args.width, args.height, args.focal_mm)


if __name__ == '__main__':
    main()



#!/usr/bin/env python3
"""Render RGB/depth/normal outputs for splatdb scenes using Blender."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import bpy  # type: ignore
    from mathutils import Matrix  # type: ignore
except ImportError as exc:  # pragma: no cover - only available inside Blender
    raise SystemExit(
        "viewer/blender_render.py must be executed from within Blender (bpy module not found)."
    ) from exc


@dataclass
class RenderOptions:
    scene_root: Path
    transforms_path: Path
    gltf_path: Path
    enable_depth: bool
    enable_normals: bool
    engine: str
    samples: int
    sun_energy: float
    sun_azimuth: float
    sun_elevation: float
    env_strength: float
    env_color: tuple[float, float, float]
    hdri_path: Path | None
    enable_denoise: bool


def _parse_argv() -> list[str]:
    argv = sys.argv
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--gltf", type=Path, required=True)
    parser.add_argument("--enable-depth", action="store_true")
    parser.add_argument("--enable-normals", action="store_true")
    parser.add_argument(
        "--engine",
        type=str,
        default="BLENDER_EEVEE",
        choices=["BLENDER_EEVEE", "CYCLES"],
        help="Render engine to use (default: BLENDER_EEVEE)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Sample count for the selected render engine (Eevee TAA or Cycles path samples)",
    )
    parser.add_argument("--sun-energy", type=float, default=5.0)
    parser.add_argument("--sun-azimuth", type=float, default=45.0, help="Degrees")
    parser.add_argument("--sun-elevation", type=float, default=35.0, help="Degrees above horizon")
    parser.add_argument("--env-strength", type=float, default=0.5)
    parser.add_argument(
        "--env-color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(0.6, 0.62, 0.65),
        help="Environment tint color (if no HDRI is provided)",
    )
    parser.add_argument("--hdri", type=Path, default=None, help="Optional HDRI environment texture")
    parser.add_argument("--enable-denoise", action="store_true", help="Enable denoising (Cycles only)")
    return parser


def _collect_options() -> RenderOptions:
    parser = _build_parser()
    args = parser.parse_args(_parse_argv())
    env_color = tuple(float(c) for c in args.env_color)
    hdri = args.hdri.resolve() if args.hdri else None
    return RenderOptions(
        scene_root=args.scene_root.resolve(),
        transforms_path=args.transforms.resolve(),
        gltf_path=args.gltf.resolve(),
        enable_depth=bool(args.enable_depth),
        enable_normals=bool(args.enable_normals),
        engine=args.engine,
        samples=max(1, int(args.samples)),
        sun_energy=float(args.sun_energy),
        sun_azimuth=float(args.sun_azimuth),
        sun_elevation=float(args.sun_elevation),
        env_strength=float(args.env_strength),
        env_color=(env_color[0], env_color[1], env_color[2]),
        hdri_path=hdri,
        enable_denoise=bool(args.enable_denoise),
    )


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in bpy.data.collections:
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def _load_gltf(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"GLTF file not found: {path}")
    bpy.ops.import_scene.gltf(filepath=str(path))


def _setup_render_engine(opts: RenderOptions) -> None:
    scene = bpy.context.scene
    scene.render.engine = opts.engine
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_file_extension = True
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    if opts.engine == "BLENDER_EEVEE":
        scene.eevee.taa_samples = opts.samples
        scene.eevee.taa_render_samples = opts.samples
        scene.eevee.use_gtao = True
        scene.eevee.use_bloom = False
        scene.eevee.shadow_cube_size = "1024"
        scene.eevee.shadow_cascade_size = "2048"
    else:  # Cycles
        scene.cycles.samples = opts.samples
        scene.cycles.device = "CPU"
        scene.cycles.use_denoising = opts.enable_denoise
        scene.cycles.preview_samples = max(1, opts.samples // 2)
        scene.cycles.use_preview_denoising = opts.enable_denoise


def _add_environment(opts: RenderOptions) -> None:
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    nodes = tree.nodes
    links = tree.links

    nodes.clear()
    background = nodes.new("ShaderNodeBackground")
    background.location = (0, 0)
    background.inputs[1].default_value = opts.env_strength

    if opts.hdri_path is not None:
        env_tex = nodes.new("ShaderNodeTexEnvironment")
        env_tex.location = (-300, 0)
        env_tex.image = bpy.data.images.load(filepath=str(opts.hdri_path))
        links.new(env_tex.outputs["Color"], background.inputs["Color"])
    else:
        background.inputs["Color"].default_value = (
            float(opts.env_color[0]),
            float(opts.env_color[1]),
            float(opts.env_color[2]),
            1.0,
        )

    world_output = nodes.new("ShaderNodeOutputWorld")
    world_output.location = (200, 0)
    links.new(background.outputs["Background"], world_output.inputs["Surface"])


def _add_sun_light(opts: RenderOptions) -> None:
    bpy.ops.object.light_add(type="SUN", radius=1.0, location=(0.0, 0.0, 0.0))
    sun_obj = bpy.context.active_object
    sun_obj.name = "KeySun"
    sun_obj.data.energy = opts.sun_energy
    az = math.radians(opts.sun_azimuth)
    el = math.radians(opts.sun_elevation)
    sun_obj.rotation_euler = (math.pi / 2 - el, 0.0, az)


def _ensure_camera() -> bpy.types.Object:
    cam_data = bpy.data.cameras.new("SplatDB_Camera")
    cam_obj = bpy.data.objects.new("SplatDB_Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def _setup_compositor(opts: RenderOptions) -> None:
    base = opts.scene_root
    (base / "images").mkdir(parents=True, exist_ok=True)
    if opts.enable_depth:
        (base / "depth").mkdir(parents=True, exist_ok=True)
    if opts.enable_normals:
        (base / "normals").mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    nodes.clear()

    render_layers = nodes.new("CompositorNodeRLayers")
    render_layers.location = (0, 200)

    composite = nodes.new("CompositorNodeComposite")
    composite.location = (400, 200)
    links.new(render_layers.outputs["Image"], composite.inputs["Image"])

    file_output = nodes.new("CompositorNodeOutputFile")
    file_output.location = (400, 0)
    file_output.base_path = str(base)
    file_output.format.file_format = "PNG"
    file_output.format.color_mode = "RGB"
    file_output.format.color_depth = "8"
    file_output.format.compression = 15

    rgb_slot = file_output.file_slots[0]
    rgb_slot.path = "images/view_"
    rgb_slot.use_node_format = True
    links.new(render_layers.outputs["Image"], file_output.inputs[0])

    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = opts.enable_depth
    view_layer.use_pass_normal = opts.enable_normals

    if opts.enable_depth:
        file_output.file_slots.new("depth")
        depth_layer = file_output.file_slots[-1]
        depth_layer.path = "depth/view_"
        depth_layer.use_node_format = False
        depth_layer.format.file_format = "OPEN_EXR"
        depth_layer.format.color_mode = "RGB"
        depth_layer.format.color_depth = "32"
        depth_layer.format.exr_codec = "ZIP"
        depth_input = file_output.inputs[-1]
        links.new(render_layers.outputs["Depth"], depth_input)

    if opts.enable_normals:
        file_output.file_slots.new("normal")
        normal_layer = file_output.file_slots[-1]
        normal_layer.path = "normals/view_"
        normal_layer.use_node_format = False
        normal_layer.format.file_format = "OPEN_EXR"
        normal_layer.format.color_mode = "RGB"
        normal_layer.format.color_depth = "16"
        normal_layer.format.exr_codec = "ZIP"
        normal_input = file_output.inputs[-1]
        links.new(render_layers.outputs["Normal"], normal_input)


def _load_transforms(path: Path) -> dict:
    with path.open("r", encoding="utf8") as stream:
        return json.load(stream)


def _render_frames(opts: RenderOptions, transforms: dict) -> None:
    scene = bpy.context.scene
    cam_obj = _ensure_camera()
    cam_data = cam_obj.data

    frames = transforms.get("frames", [])
    if not frames:
        raise SystemExit("No frames found in transforms.json; nothing to render.")

    scene.frame_start = 0
    scene.frame_end = max(len(frames) - 1, 0)

    for idx, frame in enumerate(frames):
        pose = Matrix(frame["transform_world_from_cam"])
        cam_obj.matrix_world = pose.inverted()

        intr = frame.get("intrinsics", {})
        width = int(intr.get("width", scene.render.resolution_x))
        height = int(intr.get("height", scene.render.resolution_y))
        fy = float(intr.get("fy", intr.get("fx", 1.0)))  # assume square pixels if fy missing
        vfov = 2.0 * math.atan(height / (2.0 * fy))
        cam_data.type = "PERSP"
        cam_data.lens_unit = "FOV"
        cam_data.sensor_fit = "VERTICAL"
        cam_data.angle_y = vfov
        cam_data.clip_start = float(frame.get("near", 0.01))
        cam_data.clip_end = float(frame.get("far", 1000.0))

        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100

        scene.frame_set(idx)
        bpy.ops.render.render(write_still=False, animation=False)
        print(f"Rendered view_{idx:04d}")


def main() -> None:
    opts = _collect_options()
    _clear_scene()
    _load_gltf(opts.gltf_path)
    _setup_render_engine(opts)
    _add_environment(opts)
    _add_sun_light(opts)
    _setup_compositor(opts)

    transforms = _load_transforms(opts.transforms_path)
    _render_frames(opts, transforms)
    print("Rendering complete.")


if __name__ == "__main__":
    main()

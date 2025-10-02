#!/usr/bin/env python3
"""GLTF harvester, dataset builder, and preview generator for splatdb scenes."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

try:
    import imageio.v2 as iio
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "imageio is required to run this script. Install it with 'pip install imageio'."
    ) from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "numpy is required to run this script. Install it with 'pip install numpy'."
    ) from exc

try:
    import trimesh
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "trimesh is required to run this script. Install it with 'pip install trimesh'."
    ) from exc

try:
    from pygltflib import GLTF2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pygltflib is required to run this script. Install it with 'pip install pygltflib'."
    ) from exc


@dataclass
class FrameReference:
    index: int
    rel_path: Path
    abs_path: Path
    splits: set[str] = field(default_factory=set)


@dataclass
class ViewCandidate:
    index: int
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    coverage: float
    distance: float
    transform: List[List[float]]
    timestamp: float


# GLTF utilities --------------------------------------------------------------


def _load_gltf(path: Path) -> GLTF2:
    try:
        gltf = GLTF2().load(str(path))
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Failed to load GLTF '{path}': {exc}") from exc
    return gltf


def summarize_gltf(gltf_path: Path) -> Mapping[str, int]:
    gltf = _load_gltf(gltf_path)
    return {
        "nodes": len(gltf.nodes or []),
        "meshes": len(gltf.meshes or []),
        "materials": len(gltf.materials or []),
        "textures": len(gltf.textures or []),
        "images": len(gltf.images or []),
        "animations": len(gltf.animations or []),
        "skins": len(gltf.skins or []),
        "cameras": len(gltf.cameras or []),
    }


def copy_gltf_assets(source_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, dest_dir / path.name)
        elif path.is_dir() and path.name.lower() in {"textures", "images"}:
            shutil.copytree(path, dest_dir / path.name, dirs_exist_ok=True)


# View planning ---------------------------------------------------------------


def fibonacci_sphere(samples: int) -> np.ndarray:
    if samples <= 0:
        return np.zeros((0, 3))
    idx = np.arange(samples, dtype=np.float64)
    phi = math.pi * (1 + 5 ** 0.5)
    theta = phi * idx
    z = 1 - (2 * idx + 1) / samples
    radius = np.sqrt(1 - z * z)
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    return np.stack([x, z, y], axis=1)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    norm_forward = np.linalg.norm(forward)
    if norm_forward < 1e-9:
        raise ValueError("Eye and target coincide; cannot build look-at matrix")
    forward = forward / norm_forward
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-9:
        raise ValueError("Up vector has zero length")
    up = up / up_norm
    side = np.cross(forward, up)
    side_norm = np.linalg.norm(side)
    if side_norm < 1e-9:
        if abs(up[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([0.0, 1.0, 0.0])
        up = up / np.linalg.norm(up)
        side = np.cross(forward, up)
        side /= np.linalg.norm(side)
    else:
        side = side / side_norm
    up = np.cross(side, forward)
    mat = np.eye(4, dtype=np.float64)
    mat[0, :3] = side
    mat[1, :3] = up
    mat[2, :3] = -forward
    mat[:3, 3] = eye
    return mat


def _contains_points(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    try:
        return mesh.contains(points)
    except Exception:
        centroid = mesh.bounds.mean(axis=0)
        radius = np.linalg.norm(mesh.bounds[1] - centroid)
        dist = np.linalg.norm(points - centroid, axis=1)
        return dist < radius * 0.95


def _camera_coverage(
    mesh: trimesh.Trimesh,
    eye: np.ndarray,
    sample_ids: np.ndarray,
    min_dot: float = 0.1,
) -> float:
    verts = mesh.vertices[sample_ids]
    norms = mesh.vertex_normals[sample_ids]
    to_cam = eye - verts
    lengths = np.linalg.norm(to_cam, axis=1, keepdims=True)
    lengths = np.clip(lengths, 1e-6, None)
    directions = to_cam / lengths
    dot = np.sum(directions * norms, axis=1)
    visible = dot > min_dot
    return float(np.mean(visible))


def plan_views(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    radius: float,
    target_views: int,
    width: int,
    height: int,
    vertical_fov_deg: float,
    up: np.ndarray,
    margin: float = 1.25,
    coverage_threshold: float = 0.18,
    max_trials: int = 5,
) -> List[ViewCandidate]:
    samples = np.arange(len(mesh.vertices))
    if len(samples) == 0:
        raise SystemExit("Mesh has no vertices; cannot plan views")
    sample_ids = np.random.choice(samples, size=min(1024, len(samples)), replace=False)

    views: List[ViewCandidate] = []
    trials = 0
    offset = 0
    while len(views) < target_views and trials < max_trials:
        needed = target_views - len(views)
        seed_count = max(needed * 3, target_views)
        seeds = fibonacci_sphere(seed_count + offset)
        offset += seed_count
        distance_scale = margin + trials * 0.1
        candidate_pos = center + seeds * radius * distance_scale
        inside_mask = _contains_points(mesh, candidate_pos)
        for idx, inside in enumerate(inside_mask):
            if inside:
                continue
            eye = candidate_pos[idx]
            coverage = _camera_coverage(mesh, eye, sample_ids)
            if coverage < coverage_threshold:
                continue
            t = len(views) / max(1, target_views - 1)
            transform = look_at(eye, center, up)
            views.append(
                ViewCandidate(
                    index=len(views),
                    eye=eye,
                    target=center,
                    up=up,
                    coverage=coverage,
                    distance=float(np.linalg.norm(eye - center)),
                    transform=transform.tolist(),
                    timestamp=float(t),
                )
            )
            if len(views) >= target_views:
                break
        trials += 1

    if len(views) < target_views:
        print(
            f"Warning: requested {target_views} views but only {len(views)} usable candidates found",
            file=sys.stderr,
        )
    return views


# SQLite ----------------------------------------------------------------------


class SceneDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scenes (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                scene_root TEXT NOT NULL,
                gltf_path TEXT,
                schema_version TEXT,
                coord_system TEXT,
                handedness TEXT,
                units TEXT,
                up TEXT,
                bbox_min TEXT,
                bbox_max TEXT,
                bounding_radius REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scene_stats (
                scene_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (scene_id, key),
                FOREIGN KEY (scene_id) REFERENCES scenes(id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS views (
                scene_id INTEGER NOT NULL,
                view_index INTEGER NOT NULL,
                pose TEXT NOT NULL,
                coverage REAL,
                distance REAL,
                fx REAL,
                fy REAL,
                cx REAL,
                cy REAL,
                width INTEGER,
                height INTEGER,
                near REAL,
                far REAL,
                exposure REAL,
                timestamp REAL,
                split TEXT,
                frame_path TEXT,
                PRIMARY KEY (scene_id, view_index),
                FOREIGN KEY (scene_id) REFERENCES scenes(id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS previews (
                scene_id INTEGER NOT NULL,
                split TEXT NOT NULL,
                video_path TEXT NOT NULL,
                fps REAL NOT NULL,
                frame_count INTEGER NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (scene_id, split),
                FOREIGN KEY (scene_id) REFERENCES scenes(id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_details (
                scene_id INTEGER NOT NULL,
                view_index INTEGER NOT NULL,
                json_path TEXT NOT NULL,
                PRIMARY KEY (scene_id, view_index),
                FOREIGN KEY (scene_id, view_index) REFERENCES views(scene_id, view_index) ON DELETE CASCADE
            );
            """
        )
        self._conn.commit()

    def upsert_scene(
        self,
        *,
        name: str,
        scene_root: Path,
        gltf_path: Path,
        schema_version: str,
        coord_system: str,
        handedness: str,
        units: str,
        up: Iterable[float],
        bbox_min: Iterable[float],
        bbox_max: Iterable[float],
        bounding_radius: float,
    ) -> int:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO scenes (
                name, scene_root, gltf_path, schema_version, coord_system, handedness,
                units, up, bbox_min, bbox_max, bounding_radius
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                scene_root=excluded.scene_root,
                gltf_path=excluded.gltf_path,
                schema_version=excluded.schema_version,
                coord_system=excluded.coord_system,
                handedness=excluded.handedness,
                units=excluded.units,
                up=excluded.up,
                bbox_min=excluded.bbox_min,
                bbox_max=excluded.bbox_max,
                bounding_radius=excluded.bounding_radius
            """,
            (
                name,
                str(scene_root),
                str(gltf_path),
                schema_version,
                coord_system,
                handedness,
                units,
                json.dumps(list(up)),
                json.dumps(list(bbox_min)),
                json.dumps(list(bbox_max)),
                bounding_radius,
            ),
        )
        self._conn.commit()
        cur.execute("SELECT id FROM scenes WHERE name=?", (name,))
        scene_id = cur.fetchone()[0]
        return scene_id

    def set_scene_stats(self, scene_id: int, stats: Mapping[str, object]) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM scene_stats WHERE scene_id=?", (scene_id,))
        cur.executemany(
            "INSERT INTO scene_stats (scene_id, key, value) VALUES (?, ?, ?)",
            (
                (scene_id, key, json.dumps(value))
                for key, value in sorted(stats.items())
            ),
        )
        self._conn.commit()

    def set_views(
        self,
        scene_id: int,
        views: Iterable[Mapping[str, object]],
    ) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM views WHERE scene_id=?", (scene_id,))
        cur.executemany(
            """
            INSERT INTO views (
                scene_id, view_index, pose, coverage, distance,
                fx, fy, cx, cy, width, height, near, far,
                exposure, timestamp, split, frame_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    scene_id,
                    view["index"],
                    json.dumps(view["pose"]),
                    view.get("coverage"),
                    view.get("distance"),
                    view.get("fx"),
                    view.get("fy"),
                    view.get("cx"),
                    view.get("cy"),
                    view.get("width"),
                    view.get("height"),
                    view.get("near"),
                    view.get("far"),
                    view.get("exposure"),
                    view.get("timestamp"),
                    view.get("split"),
                    view.get("frame_path"),
                )
                for view in views
            ),
        )
        self._conn.commit()

    def record_preview(
        self,
        scene_id: int,
        split: str,
        video_path: Path,
        fps: float,
        frame_count: int,
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO previews(scene_id, split, video_path, fps, frame_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(scene_id, split) DO UPDATE SET
                video_path=excluded.video_path,
                fps=excluded.fps,
                frame_count=excluded.frame_count,
                created_at=datetime('now')
            """,
            (scene_id, split, str(video_path), fps, frame_count),
        )
        self._conn.commit()

    def clear_previews(self, scene_id: int) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM previews WHERE scene_id=?", (scene_id,))
        self._conn.commit()

    def record_frame_detail(self, scene_id: int, view_index: int, json_path: Path) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO frame_details(scene_id, view_index, json_path)
            VALUES (?, ?, ?)
            ON CONFLICT(scene_id, view_index) DO UPDATE SET json_path=excluded.json_path
            """,
            (scene_id, view_index, str(json_path)),
        )
        self._conn.commit()


# Dataset writing -------------------------------------------------------------


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as stream:
        json.dump(payload, stream, indent=2)


def generate_transforms(
    scene_root: Path,
    views: List[ViewCandidate],
    width: int,
    height: int,
    vfov_deg: float,
    near: float,
    far: float,
    exposure_ev: float,
    schema_version: str,
    coord_system: str,
    handedness: str,
    units: str,
    up: Iterable[float],
) -> Mapping:
    vfov_rad = math.radians(vfov_deg)
    fy = 0.5 * height / math.tan(vfov_rad * 0.5)
    fx = fy
    cx = width * 0.5
    cy = height * 0.5

    frames = []
    for view in views:
        frame_name = f"images/view_{view.index:04d}.png"
        frames.append(
            {
                "file_path": frame_name,
                "transform_world_from_cam": view.transform,
                "intrinsics": {
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "width": width,
                    "height": height,
                },
                "distortion": {
                    "k1": 0.0,
                    "k2": 0.0,
                    "p1": 0.0,
                    "p2": 0.0,
                    "k3": 0.0,
                },
                "near": near,
                "far": far,
                "exposure_ev": exposure_ev,
                "timestamp": view.timestamp,
            }
        )
    transforms = {
        "schema": schema_version,
        "camera_model": "pinhole",
        "coord_system": coord_system,
        "handedness": handedness,
        "up": list(up),
        "units": units,
        "frames": frames,
    }
    _write_json(scene_root / "metadata" / "transforms.json", transforms)
    return transforms


def write_conventions(
    scene_root: Path,
    *,
    coord_system: str,
    handedness: str,
    units: str,
    up: Iterable[float],
    depth_space: str,
) -> None:
    payload = {
        "coord_system": coord_system,
        "handedness": handedness,
        "units": units,
        "up": list(up),
        "depth_space": depth_space,
    }
    _write_json(scene_root / "metadata" / "conventions.json", payload)


def write_provenance(scene_root: Path, *, source_gltf: Path) -> None:
    payload = {
        "source_gltf": str(source_gltf),
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
    }
    _write_json(scene_root / "metadata" / "provenance.json", payload)


def write_stats(scene_root: Path, stats: Mapping[str, object]) -> None:
    _write_json(scene_root / "metadata" / "stats.json", stats)


def write_manifest(scene_root: Path, total_frames: int) -> Mapping[str, List[str]]:
    paths = [f"images/view_{idx:04d}.png" for idx in range(total_frames)]
    payload = {"all": paths}
    _write_json(scene_root / "manifest.json", payload)
    _write_json(scene_root / "metadata" / "manifest.json", payload)
    return payload


def write_exposures(scene_root: Path, total_frames: int, exposure_ev: float) -> None:
    exposure_dir = scene_root / "exposure"
    exposure_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"exposure_ev": exposure_ev})
    for idx in range(total_frames):
        (exposure_dir / f"{idx:06d}.json").write_text(payload)


def write_frame_detail(scene_root: Path, transforms: Mapping, view_idx: int) -> Path:
    frames = transforms.get("frames", [])
    if not 0 <= view_idx < len(frames):
        raise SystemExit(f"Frame index {view_idx} out of range (0..{len(frames) - 1})")
    frame = frames[view_idx]
    payload = {
        "frame_index": view_idx,
        "file_path": frame.get("file_path"),
        "transform_world_from_cam": frame.get("transform_world_from_cam"),
        "intrinsics": frame.get("intrinsics"),
        "distortion": frame.get("distortion"),
        "near": frame.get("near"),
        "far": frame.get("far"),
        "exposure_ev": frame.get("exposure_ev"),
        "timestamp": frame.get("timestamp"),
    }
    output_path = scene_root / "metadata" / "frames" / f"frame_{view_idx:04d}.json"
    _write_json(output_path, payload)
    return output_path


# Harvest ---------------------------------------------------------------------


def harvest_scene(args: argparse.Namespace) -> None:
    gltf_dir = args.gltf.resolve()
    if gltf_dir.is_dir():
        gltf_file = gltf_dir / "scene.gltf"
    else:
        gltf_file = gltf_dir
        gltf_dir = gltf_file.parent
    if not gltf_file.exists():
        raise SystemExit(f"GLTF file not found: {gltf_file}")

    scene_root = args.output.resolve()
    scene_root.mkdir(parents=True, exist_ok=True)
    (scene_root / "images").mkdir(exist_ok=True)

    copy_gltf_assets(gltf_dir, scene_root / "gltf")
    write_provenance(scene_root, source_gltf=gltf_file)

    stats = summarize_gltf(gltf_file)

    mesh = trimesh.load(str(gltf_file), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    bounds = mesh.bounds
    bbox_min = bounds[0].tolist()
    bbox_max = bounds[1].tolist()
    center = mesh.bounding_box.centroid
    radius = float(np.max(np.linalg.norm(mesh.vertices - center, axis=1)))
    up = np.array(args.up, dtype=np.float64)

    views = plan_views(
        mesh,
        center=center,
        radius=radius,
        target_views=args.views,
        width=args.width,
        height=args.height,
        vertical_fov_deg=args.vertical_fov,
        up=up,
        margin=args.margin,
        coverage_threshold=args.coverage,
    )

    transforms = generate_transforms(
        scene_root,
        views,
        width=args.width,
        height=args.height,
        vfov_deg=args.vertical_fov,
        near=args.near,
        far=args.far,
        exposure_ev=args.exposure_ev,
        schema_version=args.schema,
        coord_system=args.coord_system,
        handedness=args.handedness,
        units=args.units,
        up=args.up,
    )

    write_conventions(
        scene_root,
        coord_system=args.coord_system,
        handedness=args.handedness,
        units=args.units,
        up=args.up,
        depth_space="ray",
    )
    write_stats(
        scene_root,
        {
            **stats,
            "planned_views": len(views),
            "bounding_box_min": bbox_min,
            "bounding_box_max": bbox_max,
            "bounding_radius": radius,
            "coverage_mean": float(np.mean([v.coverage for v in views]) if views else 0.0),
        },
    )
    write_exposures(scene_root, len(views), args.exposure_ev)
    manifest = write_manifest(scene_root, len(views))

    for obsolete in [scene_root / "splits.json", scene_root / "metadata" / "splits.json"]:
        if obsolete.exists():
            obsolete.unlink()

    db = SceneDatabase(args.sqlite.resolve())
    scene_id = db.upsert_scene(
        name=args.scene_name,
        scene_root=scene_root,
        gltf_path=gltf_file,
        schema_version=args.schema,
        coord_system=args.coord_system,
        handedness=args.handedness,
        units=args.units,
        up=args.up,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bounding_radius=radius,
    )
    db.set_scene_stats(scene_id, stats)

    vfov_rad = math.radians(args.vertical_fov)
    fy = 0.5 * args.height / math.tan(vfov_rad * 0.5)
    fx = fy
    cx = args.width * 0.5
    cy = args.height * 0.5

    view_rows = []
    for v in views:
        frame_path = f"images/view_{v.index:04d}.png"
        view_rows.append(
            {
                "index": v.index,
                "pose": v.transform,
                "coverage": v.coverage,
                "distance": v.distance,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": args.width,
                "height": args.height,
                "near": args.near,
                "far": args.far,
                "exposure": args.exposure_ev,
                "timestamp": v.timestamp,
                "split": "all",
                "frame_path": frame_path,
            }
        )
    db.set_views(scene_id, view_rows)

    (scene_root / "version.txt").write_text(f"{args.schema}\n")

    print(
        f"Harvest complete: {len(views)} planned views written to {scene_root / 'metadata' / 'transforms.json'}"
    )


# Preview ---------------------------------------------------------------------


def _load_json(path: Path) -> Mapping:
    try:
        with path.open("r", encoding="utf8") as stream:
            return json.load(stream)
    except FileNotFoundError as exc:
        raise SystemExit(f"Required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def _resolve_frames(scene_root: Path, transforms: Mapping) -> List[FrameReference]:
    frames_data = transforms.get("frames", [])
    if not frames_data:
        raise SystemExit("No frames found in transforms.json")

    resolved: List[FrameReference] = []
    for idx, frame in enumerate(frames_data):
        file_path_value = frame.get("file_path")
        if not file_path_value:
            print(f"Warning: frame {idx} missing file_path; skipping", file=sys.stderr)
            continue
        rel_path = Path(file_path_value)
        abs_path = rel_path if rel_path.is_absolute() else scene_root / rel_path
        if not abs_path.exists():
            print(
                f"Warning: frame {idx} image not found at {abs_path}; skipping",
                file=sys.stderr,
            )
            continue
        resolved.append(
            FrameReference(
                index=idx,
                rel_path=rel_path,
                abs_path=abs_path,
            )
        )
    if not resolved:
        raise SystemExit("No valid frames resolved; nothing to do.")
    return resolved


def _tag_frames_with_manifest(frames: List[FrameReference], manifest: Mapping[str, List[str]]) -> None:
    paths = manifest.get("all", [])
    lookup = {fr.rel_path.as_posix(): fr for fr in frames}
    for entry in paths:
        entry_key = Path(entry).as_posix()
        target = lookup.get(entry_key)
        if target is not None:
            target.splits.add("all")


def _write_video(frame_paths: Iterable[FrameReference], output_path: Path, fps: int) -> int:
    selection = list(frame_paths)
    if not selection:
        print(f"Skipping {output_path.name}: no frames in selection", file=sys.stderr)
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    try:
        with iio.get_writer(str(output_path), fps=fps) as writer:
            for frame in selection:
                try:
                    image = iio.imread(frame.abs_path)
                except Exception as exc:  # pragma: no cover
                    print(
                        f"Warning: failed to read {frame.abs_path}: {exc}; skipping frame",
                        file=sys.stderr,
                    )
                    continue
                writer.append_data(image)
                frame_count += 1
    except Exception as exc:  # pragma: no cover
        message = str(exc)
        if "Could not find a backend" in message:
            raise SystemExit(
                "Video backend unavailable. Install ffmpeg support via 'pip install imageio-ffmpeg' "
                "or provide an ffmpeg binary on PATH."
            ) from exc
        raise SystemExit(f"Failed to write {output_path}: {exc}") from exc

    try:
        summary_path = output_path.relative_to(output_path.parent.parent)
    except ValueError:
        summary_path = output_path
    print(f"Wrote {summary_path} with {frame_count} frames")
    return frame_count


def build_previews(scene_root: Path, output_dir: Path, fps: int, db_path: Optional[Path]) -> None:
    transforms_path = scene_root / "metadata" / "transforms.json"
    transforms = _load_json(transforms_path)
    frames = _resolve_frames(scene_root, transforms)

    manifest_path = scene_root / "metadata" / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
    else:
        manifest = {"all": [fr.rel_path.as_posix() for fr in frames]}
    _tag_frames_with_manifest(frames, manifest)

    selection = [fr for fr in frames if "all" in fr.splits] or frames
    if not selection:
        print("Nothing to render", file=sys.stderr)
        return

    db = SceneDatabase(db_path) if db_path else None
    scene_id = None
    if db:
        scene_name = scene_root.name
        scene_id = db.upsert_scene(
            name=scene_name,
            scene_root=scene_root,
            gltf_path=scene_root / "gltf" / "scene.gltf",
            schema_version=transforms.get("schema", "unknown"),
            coord_system=transforms.get("coord_system", "unknown"),
            handedness=transforms.get("handedness", "unknown"),
            units=transforms.get("units", "unknown"),
            up=transforms.get("up", [0, 1, 0]),
            bbox_min=[0, 0, 0],
            bbox_max=[0, 0, 0],
            bounding_radius=0.0,
        )
        db.clear_previews(scene_id)

    target = output_dir / "preview_all.mp4"
    frames_written = _write_video(selection, target, fps)
    if db and scene_id is not None:
        db.record_preview(scene_id, "all", target, fps, frames_written)


def export_frame_detail_command(args: argparse.Namespace) -> None:
    scene_root = args.scene_root.resolve()
    transforms = _load_json(scene_root / "metadata" / "transforms.json")
    output_path = write_frame_detail(scene_root, transforms, args.frame_index)
    if args.sqlite:
        db = SceneDatabase(args.sqlite.resolve())
        scene_id = db.upsert_scene(
            name=scene_root.name,
            scene_root=scene_root,
            gltf_path=scene_root / "gltf" / "scene.gltf",
            schema_version=transforms.get("schema", "unknown"),
            coord_system=transforms.get("coord_system", "unknown"),
            handedness=transforms.get("handedness", "unknown"),
            units=transforms.get("units", "unknown"),
            up=transforms.get("up", [0, 1, 0]),
            bbox_min=[0, 0, 0],
            bbox_max=[0, 0, 0],
            bounding_radius=0.0,
        )
        db.record_frame_detail(scene_id, args.frame_index, output_path)
    print(f"Frame {args.frame_index} metadata written to {output_path}")


# CLI -------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GLTF harvester and preview generator")
    sub = parser.add_subparsers(dest="command", required=True)

    harvest = sub.add_parser("harvest", help="Harvest a GLTF scene into splatdb layout")
    harvest.add_argument("scene_name", type=str, help="Identifier for the scene (used in DB)")
    harvest.add_argument("--gltf", type=Path, required=True, help="Path to directory or gltf/glb")
    harvest.add_argument("--output", type=Path, required=True, help="Destination dataset root")
    harvest.add_argument("--views", type=int, default=180)
    harvest.add_argument("--width", type=int, default=1024)
    harvest.add_argument("--height", type=int, default=1024)
    harvest.add_argument("--vertical-fov", type=float, default=60.0)
    harvest.add_argument("--near", type=float, default=0.01)
    harvest.add_argument("--far", type=float, default=1000.0)
    harvest.add_argument("--exposure-ev", type=float, default=0.0)
    harvest.add_argument("--coord-system", type=str, default="blender")
    harvest.add_argument("--handedness", type=str, default="right")
    harvest.add_argument("--units", type=str, default="meter")
    harvest.add_argument("--up", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    harvest.add_argument("--schema", type=str, default="splatdb-0.3.0")
    harvest.add_argument("--margin", type=float, default=1.25)
    harvest.add_argument("--coverage", type=float, default=0.18)
    harvest.add_argument("--sqlite", type=Path, default=Path("metadata/splatdb_meta.db"))
    harvest.set_defaults(func=harvest_scene)

    preview = sub.add_parser("preview", help="Generate MP4 previews for a dataset")
    preview.add_argument("scene_root", type=Path, help="Path to the scene directory")
    preview.add_argument("--output", type=Path, default=None)
    preview.add_argument("--fps", type=int, default=24)
    preview.add_argument("--sqlite", type=Path, default=None)
    preview.set_defaults(func=preview_command)

    frame = sub.add_parser("frame-detail", help="Export metadata for a single frame")
    frame.add_argument("scene_root", type=Path, help="Path to the scene directory")
    frame.add_argument("frame_index", type=int, help="Frame index to export")
    frame.add_argument("--sqlite", type=Path, default=None)
    frame.set_defaults(func=export_frame_detail_command)

    return parser


def preview_command(args: argparse.Namespace) -> None:
    scene_root = args.scene_root.resolve()
    if not scene_root.exists():
        print(f"Scene directory not found: {scene_root}", file=sys.stderr)
        raise SystemExit(1)
    output_dir = (args.output or (scene_root / "previews")).resolve()
    build_previews(scene_root, output_dir, fps=args.fps, db_path=args.sqlite)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

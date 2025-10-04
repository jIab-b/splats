# Godot 3DGS Capture Template

## Setup
1. Copy `godot_stuff/` to your Godot project's `res://addons/splat_capture/` (or directly to `res://godot_stuff/`).
2. In Project Settings > Plugins, enable "SplatCapture" (if using addons structure; otherwise, manually instance plugin.gd).
3. Create a new 3D project or open existing.
4. Ensure Input Map has actions: ui_left/right/up/down (WASD), ui_cancel (Esc/Right-click for mouse capture).
5. Add key 6 for capture (manual: Project Settings > Input Map > Add Action "capture_key" bound to 6, but script uses KEY_6 directly).

## Editor Usage (Plugin)
1. In Editor, click "Load & Setup Capture Scene" button (top toolbar).
2. Select a .gltf or .glb file.
3. Godot auto-imports, sets up scene with FlyCamera, positions at bounds.
4. Enter 3D viewport, right-click to capture mouse.
5. Move: WASD (forward/back/left/right), mouse look. Scroll wheel: adjust speed (log scale 0.1-1000).
6. Press **6** to capture current view: Saves PNG to `data/{scene_name}/images/view_{idx:04d}.png` and JSON to `metadata/frames/frame_{idx:04d}.json`.
7. Press **7** to save/update `transforms.json` (accumulates all frames).
8. Output uses fixed 1024x1024, 60° VFOV; adjust in capture_manager.gd if needed.
9. Scene saved as `capture_{scene_name}.tscn` for reload.

## Runtime Usage (Auto-Launch Scene)
For programmatic launch without editor (e.g., for chinch GLTF):
1. Set `auto_capture.tscn` as the main scene: Project Settings > Application > Run > Main Scene = res://godot_stuff/auto_capture.tscn.
2. Ensure GLTF exists at `res://scenes/chinch/scene.gltf` (relative to project root).
3. Run the project (F5 or play button): Automatically loads GLTF, sets up camera at bounds, initializes capture for 'chinch'.
4. Immediately ready: Right-click for mouse lock, WASD/scroll to navigate, spam **6** to capture views.
5. Press **7** to save `transforms.json`.
6. Outputs to project root's `data/chinch/`.
7. Console: "Auto-capture setup complete for chinch."

## Notes
- Godot coord: Y-up by default; adjust "up" in transforms if Blender Z-up needed.
- No depth/normals yet; extend capture_view() with SubViewport + shaders.
- For production: Add custom importer for auto-setup on GLTF import.
- Test: Load simple GLTF (e.g., cube), capture 5 views, check outputs match splatdb format.
- Runtime vs Editor: Auto_capture.gd works in play mode; plugin for editor fly-through. For other GLTFs, modify path/name in script.

## Files
- `plugin.gd`: Editor integration, GLTF loader/setup.
- `fly_camera.gd`: Movement script.
- `capture_manager.gd`: Keybind capture + JSON/PNG export (as child in runtime).
- `auto_capture.gd`: Programmatic runtime scene for chinch GLTF (set as main scene).
- `auto_capture.tscn`: Pre-made .tscn with auto_capture.gd attached (set as main scene for instant launch).

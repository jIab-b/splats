extends Node

var scene_name: String = ""
var frame_idx: int = 0
var data_dir: String = ""
var viewport_size: Vector2i = Vector2i(1024, 1024)  # Fixed for captures
var vfov_deg: float = 60.0
var near: float = 0.01
var far: float = 1000.0

func _ready():
    # Ensure input handling
    set_process_input(true)

func set_scene_name(name: String):
    scene_name = name
    data_dir = "data/" + scene_name
    DirAccess.make_dir_absolute_and_recurse(data_dir)
    DirAccess.make_dir_absolute_and_recurse(data_dir + "/images")
    DirAccess.make_dir_absolute_and_recurse(data_dir + "/metadata/frames")
    frame_idx = 0
    print("Capture setup for: " + scene_name)

func _input(event):
    if event is InputEventKey and event.pressed:
        if event.keycode == KEY_6:  # Press 6 to capture
            capture_view()
            accept_event()

func capture_view():
    var camera = get_viewport().get_camera_3d()
    if not camera:
        push_error("No active camera for capture")
        return
    
    # Get transform: world_from_cam (inverse of camera global_transform)
    var cam_transform = camera.global_transform
    var world_from_cam = cam_transform.affine_inverse()
    var matrix_4x4 = []  # List of lists
    for i in range(4):
        var row = []
        for j in range(4):
            row.append(world_from_cam[i][j])
        matrix_4x4.append(row)
    
    # Intrinsics (fixed for simplicity)
    var vfov_rad = deg_to_rad(vfov_deg)
    var fy = viewport_size.y / (2.0 * tan(vfov_rad / 2.0))
    var fx = fy  # Square pixels
    var cx = viewport_size.x / 2.0
    var cy = viewport_size.y / 2.0
    var intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": viewport_size.x,
        "height": viewport_size.y
    }
    
    # Capture RGB
    # Temporarily set viewport size for capture (restore after)
    var original_size = get_viewport().size
    get_viewport().size = viewport_size
    get_viewport().render_target_update_mode = Viewport.UPDATE_ONCE
    
    await get_tree().process_frame  # Wait for render
    
    var img = get_viewport().get_texture().get_image()
    var png_path = data_dir + "/images/view_%04d.png" % frame_idx
    img.save_png(png_path)
    
    # Restore viewport
    get_viewport().size = original_size
    get_viewport().render_target_update_mode = Viewport.UPDATE_ALWAYS
    
    # Frame JSON
    var frame_data = {
        "frame_index": frame_idx,
        "file_path": "images/view_%04d.png" % frame_idx,
        "transform_world_from_cam": matrix_4x4,
        "intrinsics": intrinsics,
        "distortion": {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        "near": near,
        "far": far,
        "exposure_ev": 0.0,
        "timestamp": float(frame_idx) / 60.0  # Dummy timestamp
    }
    
    var json_path = data_dir + "/metadata/frames/frame_%04d.json" % frame_idx
    var file = FileAccess.open(json_path, FileAccess.WRITE)
    if file:
        file.store_string(JSON.stringify(frame_data, "  ", false))
        file.close()
    
    print("Captured frame %d: %s and %s" % [frame_idx, png_path.get_file(), json_path.get_file()])
    frame_idx += 1

# Optional: Save transforms.json on demand (e.g., key 7)
func _input(event):
    # ... existing ...
    if event is InputEventKey and event.pressed:
        if event.keycode == KEY_7:
            save_transforms()
            accept_event()

func save_transforms():
    if frame_idx == 0:
        return
    
    var transforms = {
        "schema": "splatdb-0.3.0",
        "camera_model": "pinhole",
        "coord_system": "blender",  # Adjust if needed
        "handedness": "right",
        "up": [0.0, 0.0, 1.0],  # Z-up for Godot?
        "units": "meter",
        "frames": []
    }
    
    # Load existing frames if any
    for i in range(frame_idx):
        var json_path = data_dir + "/metadata/frames/frame_%04d.json" % i
        if FileAccess.file_exists(json_path):
            var file = FileAccess.open(json_path, FileAccess.READ)
            var json_str = file.get_as_text()
            file.close()
            var parsed = JSON.parse_string(json_str)
            transforms.frames.append(parsed)
    
    var transforms_path = data_dir + "/metadata/transforms.json"
    var file = FileAccess.open(transforms_path, FileAccess.WRITE)
    if file:
        file.store_string(JSON.stringify(transforms, "  ", false))
        file.close()
        print("Saved transforms.json with %d frames" % frame_idx)

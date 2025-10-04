extends Node3D

@onready var camera: Camera3D
var fly_camera_script = preload("res://godot_stuff/fly_camera.gd")
var capture_manager_script = preload("res://godot_stuff/capture_manager.gd")

func _ready():
    # Load GLTF programmatically
    var gltf_path = "res://scenes/chinch/scene.gltf"
    var scene_resource = ResourceLoader.load(gltf_path)
    if not scene_resource:
        push_error("Failed to load GLTF at " + gltf_path)
        return
    
    var gltf_instance: Node3D
    if scene_resource is PackedScene:
        gltf_instance = scene_resource.instantiate()
    elif scene_resource is Node3D:
        gltf_instance = scene_resource
    else:
        push_error("GLTF resource is not a scene or Node3D")
        return
    
    add_child(gltf_instance)
    
    # Compute scene bounds
    var aabb = get_scene_aabb(gltf_instance)
    var center = aabb.position + aabb.size / 2.0
    var radius = aabb.get_longest_axis_size() / 2.0
    var cam_pos = center + Vector3(0, 0, radius * 2.0)  # Offset along -Z (Godot forward)
    
    # Add FlyCamera
    camera = Camera3D.new()
    camera.set_script(fly_camera_script)
    camera.global_position = cam_pos
    camera.look_at(center, Vector3.UP)
    add_child(camera)
    camera.make_current()
    
    # Add CaptureManager as child (for runtime; assumes no AutoLoad)
    var capture_manager = Node.new()
    capture_manager.set_script(capture_manager_script)
    add_child(capture_manager)
    capture_manager.set_scene_name("chinch")
    
    # Input setup for runtime (mouse capture, etc.)
    Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
    
    print("Auto-capture setup complete for chinch. Use WASD/mouse to navigate, press 6 to capture views.")

func get_scene_aabb(node: Node3D) -> AABB:
    var aabb = AABB()
    if node is MeshInstance3D:
        var mesh = node.mesh
        if mesh:
            var local_aabb = mesh.get_aabb()
            # Transform to global
            var global_aabb = node.global_transform * local_aabb
            aabb = aabb.merge(global_aabb)
    # Recurse children
    for child in node.get_children():
        if child is Node3D:
            aabb = aabb.merge(get_scene_aabb(child))
    return aabb

extends EditorPlugin

const FLY_CAMERA_SCRIPT = preload("res://godot_stuff/fly_camera.gd")
const CAPTURE_MANAGER_SCRIPT = preload("res://godot_stuff/capture_manager.gd")

var tool_button: Button
var importer: EditorImportPlugin

func _enter_tree():
    # Add custom importer if needed, but for simplicity, use manual load
    importer = preload("res://godot_stuff/custom_importer.gd").new() if ResourceLoader.exists("res://godot_stuff/custom_importer.gd") else null
    if importer:
        add_import_plugin(importer)
    
    # Tool button for loading GLTF
    tool_button = Button.new()
    tool_button.text = "Load & Setup Capture Scene"
    tool_button.pressed.connect(_on_load_pressed)
    get_editor_interface().get_editor_main_screen().add_child(tool_button)
    
    # Add CaptureManager as temporary AutoLoad for editor session
    add_autoload_singleton("CaptureManager", "res://godot_stuff/capture_manager.gd")

func _exit_tree():
    if tool_button:
        tool_button.queue_free()
    if importer:
        remove_import_plugin(importer)
    remove_autoload_singleton("CaptureManager")

func _on_load_pressed():
    var file_dialog = EditorFileDialog.new()
    file_dialog.file_mode = EditorFileDialog.FILE_MODE_OPEN_FILE
    file_dialog.access = EditorFileDialog.ACCESS_FILESYSTEM
    file_dialog.filters = ["*.gltf ; GLTF Scene", "*.glb ; GLTF Binary"]
    file_dialog.file_selected.connect(_on_gltf_selected)
    get_editor_interface().get_base_control().add_child(file_dialog)
    file_dialog.popup_centered_ratio(0.75)

func _on_gltf_selected(path: String):
    var scene = load(path) as PackedScene
    if not scene:
        # Try to load as resource and instance
        var resource = ResourceLoader.load(path)
        if resource is Node3D:
            scene = PackedScene.new()
            scene.pack(resource)
        else:
            push_error("Failed to load GLTF as scene")
            return
    
    # Create new scene
    var editor_interface = get_editor_interface()
    var new_scene = editor_interface.get_edited_scene_root() or Node3D.new()
    if not new_scene.get_parent():
        editor_interface.make_scene_current(new_scene)
    
    # Clear existing children if needed
    for child in new_scene.get_children():
        child.queue_free()
    
    # Instance GLTF
    var gltf_instance = scene.instantiate()
    new_scene.add_child(gltf_instance)
    gltf_instance.owner = new_scene
    
    # Compute bounds for camera position
    var aabb = get_scene_aabb(gltf_instance)
    var center = aabb.position + aabb.size / 2
    var radius = aabb.get_longest_axis_size() / 2
    var cam_pos = center + Vector3(0, 0, radius * 2)  # Offset along Z
    
    # Add FlyCamera
    var camera = Camera3D.new()
    camera.set_script(FLY_CAMERA_SCRIPT)
    camera.global_position = cam_pos
    camera.look_at(center, Vector3.UP)
    new_scene.add_child(camera)
    camera.owner = new_scene
    camera.make_current()
    
    # Set scene name for capture
    var scene_name = path.get_file().get_basename()
    CaptureManager.set_scene_name(scene_name)
    
    # Save as capture scene
    var save_path = "res://capture_" + scene_name + ".tscn"
    var saved_scene = PackedScene.new()
    saved_scene.pack(new_scene)
    ResourceSaver.save(saved_scene, save_path)
    editor_interface.open_scene(save_path)
    
    print("Capture scene setup complete for: " + scene_name)

func get_scene_aabb(node: Node3D) -> AABB:
    var aabb = AABB()
    if node is MeshInstance3D:
        var mesh = node.mesh
        if mesh:
            var instance_aabb = mesh.get_aabb()
            var transform_aabb = node.global_transform * instance_aabb
            aabb = aabb.merge(transform_aabb)
    for child in node.get_children():
        if child is Node3D:
            aabb = aabb.merge(get_scene_aabb(child))
    return aabb

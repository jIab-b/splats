extends Camera3D

var speed: float = 10.0  # Initial speed
var min_speed: float = 0.1
var max_speed: float = 1000.0
var sensitivity: float = 0.005  # Mouse look sensitivity
var velocity: Vector3 = Vector3.ZERO
var yaw: float = 0.0
var pitch: float = 0.0

func _ready():
    make_current()
    Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)

func _input(event):
    # Mouse look
    if Input.is_action_just_pressed("ui_cancel"):  # Right-click or Esc
        if Input.get_mouse_mode() == Input.MOUSE_MODE_VISIBLE:
            Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
        else:
            Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
    
    if event is InputEventMouseMotion and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
        yaw -= event.relative.x * sensitivity
        pitch -= event.relative.y * sensitivity
        pitch = clamp(pitch, -PI/2 + 0.01, PI/2 - 0.01)
        global_transform.basis = Basis.from_euler(Vector3(pitch, yaw, 0))
    
    # Speed adjustment with wheel (log scale)
    if event is InputEventMouseButton:
        if event.button_index == MOUSE_BUTTON_WHEEL_UP:
            speed = min(max_speed, speed * 2.0)  # Exponential increase
            print("Speed: " + str(speed))
        elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
            speed = max(min_speed, speed / 2.0)  # Exponential decrease
            print("Speed: " + str(speed))

func _unhandled_input(event):
    # Pass wheel to parent if needed, but handle here
    accept_event()

func _physics_process(delta):
    # WASD movement (editor-like: forward/back/left/right in camera plane)
    var input_dir = Input.get_vector("ui_left", "ui_right", "ui_up", "ui_down")
    if input_dir != Vector2.ZERO:
        var cam_basis = global_transform.basis
        var move_dir = (cam_basis.z * input_dir.y + cam_basis.x * input_dir.x).normalized()
        velocity = move_dir * speed
    else:
        velocity = velocity.lerp(Vector3.ZERO, 5.0 * delta)  # Dampen
    
    global_position += velocity * delta

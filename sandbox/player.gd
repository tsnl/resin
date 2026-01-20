extends Node3D

var movement_speed: float = 5.0


func _ready() -> void:
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _process(delta: float) -> void:
	if Input.is_key_pressed(Key.KEY_W):
		self.translate_object_local(Vector3(0, 0, -1) * delta * movement_speed)
	
	if Input.is_key_pressed(Key.KEY_S):
		self.translate_object_local(Vector3(0, 0, 1) * delta * movement_speed)

	if Input.is_key_pressed(Key.KEY_A):
		self.translate_object_local(Vector3(-1, 0, 0) * delta * movement_speed)
	
	if Input.is_key_pressed(Key.KEY_D):
		self.translate_object_local(Vector3(1, 0, 0) * delta * movement_speed)

	var mouse_motion = Input.get_last_mouse_velocity()
	self.rotate_y(-mouse_motion.x * delta * 0.002)
	self.get_child(1).rotate_x(-mouse_motion.y * delta * 0.002)

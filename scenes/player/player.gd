extends Node3D


@export var avatar_node: Node3D
@export var camera_node: Camera3D


func _ready() -> void:
	pass # Replace with function body.


func _process(delta: float) -> void:
	if Input.is_key_pressed(Key.KEY_W):
		self.avatar_node.translate_object_local(Vector3(0, 0, -1) * delta * 5.0)

	if Input.is_key_pressed(Key.KEY_S):
		self.avatar_node.translate_object_local(Vector3(0, 0, 1) * delta * 5.0)

extends Camera3D

@export var follow_target: Node3D

var follow_transform: Transform3D


func _ready() -> void:
	follow_transform = self.get_global_transform() * follow_target.get_global_transform().affine_inverse()
	

func _process(delta: float) -> void:
	self.set_global_transform(follow_transform * follow_target.get_global_transform())
